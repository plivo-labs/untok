"""Append-only extensions of the immutable original Nemotron tokenizer.

Profile filtering and quality cleanup apply only to additions. Every original
piece message and all original normalization metadata are preserved exactly.
"""
from __future__ import annotations

from importlib import resources
import json
import math
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from .unigram import NativeTokenizerAdapter, _digest, _load, _vocabulary, native_id_map, validate_native_prefix

ALGORITHM = "native_sentencepiece_unigram_preserved_v3"
POLICY_SHA256 = "ab80a9e5104ee13b8f7ced2c47f0ec49a45a33ab9b1a4b0030f596ba8106292c"


def _json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _extension_policy():
    root = resources.files("untok").joinpath("data", "extension-v3")
    raw = root.joinpath("policy.json").read_bytes()
    if _digest(raw) != POLICY_SHA256:
        raise ValueError("Pinned append-only extension policy changed")
    return json.loads(raw)


def _dominated(pieces):
    scores = {p.piece: p.score for p in pieces if p.type == pb.ModelProto.SentencePiece.NORMAL}
    result = {}
    for piece, score in scores.items():
        if len(piece) < 2:
            continue
        best = [-math.inf] * (len(piece) + 1)
        best[0] = 0.0
        paths = [[] for _ in best]
        for end in range(1, len(piece) + 1):
            for start in range(end):
                if start == 0 and end == len(piece):
                    continue
                part = piece[start:end]
                candidate = best[start] + scores.get(part, -math.inf)
                if candidate > best[end]:
                    best[end] = candidate
                    paths[end] = paths[start] + [part]
        if best[-1] > score + 1e-7:
            result[piece] = {"reason": "strictly_dominated", "piece_score": score,
                             "replacement_score": best[-1], "replacement": paths[-1]}
    return result


def _artifacts(base_bytes: bytes, full_bytes: bytes, profile: str):
    from .bundles import PROFILES, character_allowed

    if profile not in PROFILES:
        raise ValueError("Unknown clean profile")
    source = validate_native_prefix(base_bytes, full_bytes)
    full = _load(full_bytes)
    policy = _extension_policy()
    if source["base_tokenizer_sha256"] != policy["source_base_sha256"]:
        raise ValueError("Preserved v3 requires the pinned original native base")
    if source["tokenizer_sha256"] != policy["source_full_sha256"]:
        raise ValueError("Preserved v3 source vocabulary differs from its versioned recipe")
    base_size = source["native_entries_preserved"]
    target = pb.ModelProto()
    target.CopyFrom(full)
    normalizer = spm.SentencePieceNormalizer(
        model_proto=target.SerializeToString(), add_dummy_prefix=False,
        escape_whitespaces=True, remove_extra_whitespaces=False)
    removed = []
    retained = []
    for index, p in enumerate(full.pieces):
        reason = None
        if index >= base_size and p.type == pb.ModelProto.SentencePiece.NORMAL:
            if not all(character_allowed(c, profile) for c in p.piece):
                reason = {"reason": "outside_profile"}
            else:
                normalized = normalizer.normalize(p.piece.replace("▁", " "))
                if normalized != p.piece:
                    reason = {"reason": "normalization_changes_piece", "normalized": normalized}
        if reason:
            removed.append({"full_native_id": index, "piece": p.piece, **reason})
        else:
            retained.append((index, p))
    dominated = _dominated([p for _, p in retained])
    for index, p in retained:
        if index >= base_size and p.piece in dominated:
            removed.append({"full_native_id": index, "piece": p.piece, **dominated[p.piece]})
    retained = [(i, p) for i, p in retained if i < base_size or p.piece not in dominated]
    target.ClearField("pieces")
    reverse = []
    for index, piece in retained:
        target.pieces.add().CopyFrom(piece)
        reverse.append(index)
    added = []
    # Coverage is added after the immutable original bank, using its normalizer.
    for char in sorted(policy["coverage_additions"]):
        if character_allowed(char, profile) and char not in {p.piece for p in target.pieces}:
            if normalizer.normalize(char) != char:
                raise ValueError("Coverage addition is unstable under the original normalizer")
            target.pieces.add(piece=char, score=-32.0, type=pb.ModelProto.SentencePiece.NORMAL)
            reverse.append(None)
            added.append({"piece": char, "score": -32.0, "reason": "unicode_case_decomposition_coverage"})
    size = len(target.pieces)
    full_map = [None] * (len(full.pieces) + 1)
    for new, old in enumerate(reverse):
        if old is not None:
            full_map[old] = new
    full_map[-1] = size
    target.trainer_spec.vocab_size = size
    # Original special IDs and even implicit/default protobuf fields stay exact.
    model_bytes = target.SerializeToString()
    preservation = validate_native_prefix(base_bytes, model_bytes)
    dominated_after = _dominated(target.pieces)
    if any(p.piece in dominated_after for p in target.pieces[base_size:]):
        raise ValueError("Cleanup retained a strictly dominated addition")
    for p in target.pieces[base_size:]:
        if p.type == pb.ModelProto.SentencePiece.NORMAL and normalizer.normalize(p.piece.replace("▁", " ")) != p.piece:
            raise ValueError("Cleanup retained a normalization-changing addition")
    if full_map[:base_size] != list(range(base_size)):
        raise ValueError("Original Nemotron text IDs changed")
    mapping = {
        "schema_version": 1, "layout": "preserved_v3_original_text_prefix_native_blank_last",
        "source_native_to_target_native": full_map[:base_size] + [size],
        "full_native_to_target_native": full_map,
        "target_native_to_full_native": reverse + [len(full.pieces)],
        "source_blank_id": base_size, "full_blank_id": len(full.pieces), "target_blank_id": size,
        "removed_ids_require_retokenization": True,
    }
    public = native_id_map(base_bytes, model_bytes)
    cleanup = {"profile": profile, "removed": sorted(removed, key=lambda x: x["full_native_id"]),
               "added": added, "retained_piece_scores": "unchanged", "global_score_refit": False,
               "original_native_entries_preserved": base_size,
               "original_piece_ids_scores_types_preserved": True,
               "original_normalizer_and_metadata_preserved": True,
               "profile_scope": "Filter additions only; the complete original multilingual bank is always retained",
               "additions_normalization_stable": True, "strictly_dominated_additions_remaining": 0,
               "inherited_strictly_dominated_pieces": sorted(p.piece for p in target.pieces[:base_size] if p.piece in dominated_after),
               "inherited_normalization_changing_pieces": [p.piece for p in target.pieces[:base_size]
                   if p.type == pb.ModelProto.SentencePiece.NORMAL and normalizer.normalize(p.piece.replace("▁", " ")) != p.piece],
               "locale_tag_types": "preserved from original", "extension_policy": policy}
    files = {"base-tokenizer.model": base_bytes, "full-tokenizer.model": full_bytes,
             "tokenizer.model": model_bytes, "native-row-map.json": _json(mapping),
             "nemo-id-map.json": _json(public.to_dict()), "vocabulary.json": _json(_vocabulary(target, public)),
             "cleanup.json": _json(cleanup)}
    manifest = {
        "schema_version": 1, "algorithm": ALGORITHM, "profile": profile,
        "tokenizer_version": 3,
        "status": "preserved_tokenizer_candidate", "structural_passed": True,
        "checkpoint_validated": False, "asr_validated": False, "requires_retokenized_training_labels": True,
        "native_vocabulary_size": size, "native_blank_id": size,
        "original_native_entries_preserved": base_size, "original_native_text_ids_unchanged": True,
        "original_normalizer_unchanged": True, "native_prefix_preserved": True,
        "original_native_blank_id": base_size, "native_blank_id_unchanged": size == base_size,
        "public_pad_id": public.hf_pad_id, "public_blank_id": public.hf_blank_id,
        "base_tokenizer_sha256": _digest(base_bytes), "full_tokenizer_sha256": _digest(full_bytes),
        "tokenizer_sha256": _digest(model_bytes), "normalizer_sha256": preservation["normalizer_sha256"],
        "files": {name: _digest(raw) for name, raw in files.items()},
    }
    return files, manifest, mapping, public


def build_clean_bundles(bundle: str | Path, output: str | Path):
    """Create three append-only profiles preserving the complete original bank."""
    from .bundles import load_tokenizer_bundle, PROFILES
    from .export_notices import source_notice_files, write_export_notices

    source = load_tokenizer_bundle(bundle)
    if not isinstance(source, NativeTokenizerAdapter) or not hasattr(source, "base_model_bytes"):
        raise ValueError("Expected the original full bundle")
    # Reduced and clean bundles carry the original full model, so they can
    # rebuild any profile without chaining lossy vocabulary transformations.
    full_bytes = getattr(source, "full_model_bytes", source.model_bytes)
    validate_native_prefix(source.base_model_bytes, full_bytes)
    notices = source_notice_files(bundle) if isinstance(bundle, Path) or bundle not in PROFILES else {}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty destination")
    builds = {p: _artifacts(source.base_model_bytes, full_bytes, p) for p in PROFILES}
    result = {}
    for profile, (files, manifest, _, _) in builds.items():
        destination = output / profile
        destination.mkdir(parents=True, exist_ok=True)
        for name, raw in files.items():
            (destination / name).write_bytes(raw)
        (destination / "manifest.json").write_bytes(_json(manifest))
        write_export_notices(destination, notices)
        result[profile] = manifest
    return result


class CleanTokenizerAdapter(NativeTokenizerAdapter):
    """Load a reproducible extension and enforce the complete original prefix."""

    def __init__(self, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("algorithm") != ALGORITHM:
            raise ValueError("Expected a preserved v3 bundle")
        profile = manifest.get("profile", "")
        names = {"base-tokenizer.model", "full-tokenizer.model", "tokenizer.model", "native-row-map.json",
                 "nemo-id-map.json", "vocabulary.json", "cleanup.json"}
        if set(manifest.get("files", {})) != names:
            raise ValueError("Incomplete preserved v3 manifest")
        actual = {name: (directory / name).read_bytes() for name in names}
        if any(_digest(raw) != manifest["files"][name] for name, raw in actual.items()):
            raise ValueError("Preserved v3 file hash mismatch")
        expected, expected_manifest, mapping, public = _artifacts(
            actual["base-tokenizer.model"], actual["full-tokenizer.model"], profile)
        if actual != expected or manifest != expected_manifest:
            raise ValueError("Preserved v3 artifacts differ from the pinned cleanup recipe")
        self.base_model_bytes = actual["base-tokenizer.model"]
        self.full_model_bytes = actual["full-tokenizer.model"]
        self.model_bytes = actual["tokenizer.model"]
        self.source_native_to_target_native = tuple(mapping["source_native_to_target_native"])
        self.full_native_to_subset_native = tuple(mapping["full_native_to_target_native"])
        self.subset_native_to_full_native = tuple(mapping["target_native_to_full_native"])
        self.id_map = public
        self.profile = profile
        self.backend = spm.SentencePieceProcessor(model_proto=self.model_bytes)
        self.tokenizer = self
        self.vocab_size = self.backend.get_piece_size()
        self.blank_id = public.model_blank_id
        self.pad_id = self.blank_id
        self.unk_id, self.bos_id, self.eos_id = self.backend.unk_id(), self.backend.bos_id(), self.backend.eos_id()
        self.vocab = self.get_vocab()
