"""Reproducible original, script-restricted and expanded native tokenizers."""
from __future__ import annotations

from importlib import resources
import json
import math
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from .unigram import NativeTokenizerAdapter, _digest, _load, _vocabulary, native_id_map, validate_native_prefix

ALGORITHM = "native_sentencepiece_unigram_profiles_v4"
LEGACY_ALGORITHM = "native_sentencepiece_unigram_preserved_v3"
POLICY_SHA256 = "ab80a9e5104ee13b8f7ced2c47f0ec49a45a33ab9b1a4b0030f596ba8106292c"
SELECTION_SHA256 = "2acb9490e1a710ceac79f6d34b7eac1e72d5d6179ace72753d6405032456190e"


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


def _preserved_v3_artifacts(base_bytes: bytes, full_bytes: bytes, profile: str):
    """Frozen v3 recipe, retained only to validate existing v3 bundles."""
    from .bundles import character_allowed

    if profile not in ("latin", "latin-indic", "full"):
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
        "schema_version": 1, "algorithm": LEGACY_ALGORITHM, "profile": profile,
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


def _artifacts(base_bytes: bytes, full_bytes: bytes, profile: str):
    """Build v4 from the pinned source, preserving retained piece messages."""
    from .bundles import PROFILES, script_policy
    from .profile_policy import piece_allowed, indic_addition_allowed, profile_policy
    from .runtime import IdMap

    if profile not in PROFILES:
        raise ValueError("Unknown tokenizer profile")
    source_policy = _extension_policy()
    if (_digest(base_bytes) != source_policy["source_base_sha256"]
            or _digest(full_bytes) != source_policy["source_full_sha256"]):
        raise ValueError("V4 profiles require the pinned original Nemotron base and Untok source inventory")
    # Reuse the frozen source cleanup and coverage recipe, then apply the v4
    # inventory contract to both the base and extensions. No scores are fitted.
    base, full = _load(base_bytes), _load(full_bytes)
    base_size, full_size = len(base.pieces), len(full.pieces)
    if profile == "original":
        validate_native_prefix(base_bytes, full_bytes)
        candidate = full
        candidate_map = {"target_native_to_full_native": list(range(full_size))}
        previous_cleanup = {"removed": [], "added": []}
        additions = {}
    else:
        candidate_files, _, candidate_map, _ = _preserved_v3_artifacts(base_bytes, full_bytes, "full")
        candidate = _load(candidate_files["tokenizer.model"])
        previous_cleanup = json.loads(candidate_files["cleanup.json"])
        selection_bytes = resources.files("untok").joinpath("data", "source", "selection.json").read_bytes()
        if _digest(selection_bytes) != SELECTION_SHA256:
            raise ValueError("Pinned profile source selection changed")
        additions = {entry["piece"]: entry for entry in json.loads(selection_bytes)["additions"]}
    removed = list(previous_cleanup["removed"])
    selected = []
    for index, piece in enumerate(candidate.pieces):
        old = candidate_map["target_native_to_full_native"][index]
        if profile == "original":
            keep = old is not None and old < base_size
        elif profile == "full":
            # Newly generated Latin case/decomposition coverage has no source
            # row. Full is specifically Nemotron plus source Indic additions.
            keep = old is not None and (old < base_size or indic_addition_allowed(piece, additions.get(piece.piece)))
        elif profile == "latin":
            keep = old is not None and old < base_size and piece_allowed(piece, profile)
        else:
            keep = (old is not None and (old < base_size or indic_addition_allowed(piece, additions.get(piece.piece)))
                    and piece_allowed(piece, profile))
        if keep:
            selected.append((old, piece))
        elif old is not None:
            removed.append({"full_native_id": old, "piece": piece.piece, "reason": "outside_profile"})

    target = pb.ModelProto()
    target.CopyFrom(base if profile == "original" else candidate)
    if profile != "original":
        target.ClearField("pieces")
        for _, piece in selected:
            target.pieces.add().CopyFrom(piece)
        target.trainer_spec.vocab_size = len(selected)
        if profile in {"latin", "latin-indic"}:
            # Remove stale trainer declarations for excluded language symbols.
            retained_strings = {p.piece for _, p in selected}
            for field in ("user_defined_symbols", "control_symbols"):
                values = [value for value in getattr(target.trainer_spec, field) if value in retained_strings]
                target.trainer_spec.ClearField(field)
                getattr(target.trainer_spec, field).extend(values)
            indices = {p.piece: i for i, (_, p) in enumerate(selected)}
            for field in ("unk_id", "bos_id", "eos_id", "pad_id"):
                old_id = getattr(candidate.trainer_spec, field)
                if old_id >= 0:
                    spelling = candidate.pieces[old_id].piece
                    if spelling not in indices:
                        raise ValueError(f"Profile removed required trainer special ID: {field}")
                    if indices[spelling] != old_id:
                        setattr(target.trainer_spec, field, indices[spelling])

    model_bytes = base_bytes if profile == "original" else target.SerializeToString()
    _load(model_bytes)
    size = len(target.pieces)
    reverse = [old for old, _ in selected]
    forward = [None] * (full_size + 1)
    for new, old in enumerate(reverse):
        if old is not None:
            forward[old] = new
    forward[-1] = size
    prefix_preserved = profile in {"original", "full"}
    if prefix_preserved:
        validate_native_prefix(base_bytes, model_bytes)
        public = native_id_map(base_bytes, model_bytes)
    else:
        public = IdMap(tuple(range(size)) + (None, size), tuple(range(size)) + (size + 1,),
                       size, size + 1, size, _digest(model_bytes), _digest(base_bytes))
    base_scores = [p.score for p in base.pieces if p.type == pb.ModelProto.SentencePiece.NORMAL]
    scores = [p.score for p in target.pieces if p.type == pb.ModelProto.SentencePiece.NORMAL]
    if (min(scores), max(scores)) != (min(base_scores), max(base_scores)):
        raise ValueError("Profile changed native NORMAL score extrema")
    mapping = {
        "schema_version": 1, "layout": "profiles_v4_dense_native_blank_last",
        "source_native_to_target_native": forward[:base_size] + [size],
        "full_native_to_target_native": forward,
        "target_native_to_full_native": reverse + [full_size],
        "source_blank_id": base_size, "full_blank_id": full_size, "target_blank_id": size,
        "removed_ids_require_retokenization": not prefix_preserved,
    }
    policy = profile_policy(profile)
    if profile in {"latin", "latin-indic"}:
        policy["unicode_script_policy"] = script_policy(profile)
        policy["unicode_script_policy"]["non_normal_pieces"] = policy["special_tokens"]
    retained_strings = {p.piece for p in target.pieces}
    retained_base = [p for old, p in selected if old is not None and old < base_size]
    extension_pieces = [p for old, p in selected if old is None or old >= base_size]
    normalizer = spm.SentencePieceNormalizer(model_proto=model_bytes, add_dummy_prefix=False,
                                           escape_whitespaces=True, remove_extra_whitespaces=False)
    dominated = _dominated(target.pieces)
    if any(p.piece in dominated for p in extension_pieces):
        raise ValueError("Profile retained a strictly dominated addition")
    if any(normalizer.normalize(p.piece.replace("▁", " ")) != p.piece for p in extension_pieces):
        raise ValueError("Profile retained a normalization-changing addition")
    cleanup = {
        "profile": profile, "removed": sorted(removed, key=lambda item: item["full_native_id"]),
        "added": [item for item in previous_cleanup["added"] if item["piece"] in retained_strings],
        "retained_piece_scores": "unchanged", "global_score_refit": False,
        "original_native_entries_preserved": len(retained_base),
        "original_piece_ids_scores_types_preserved": prefix_preserved,
        "retained_piece_messages_preserved": True,
        "original_normalizer_and_metadata_preserved": prefix_preserved,
        "original_normalizer_unchanged": True, "profile_scope": policy["scope"],
        "additions_normalization_stable": True, "strictly_dominated_additions_remaining": 0,
        "inherited_strictly_dominated_pieces": sorted(p.piece for p in retained_base if p.piece in dominated),
        "inherited_normalization_changing_pieces": [p.piece for p in retained_base
            if p.type == pb.ModelProto.SentencePiece.NORMAL and normalizer.normalize(p.piece.replace("▁", " ")) != p.piece],
        "locale_tag_types": "retained source tags preserve their original types",
        "extension_policy": {"source_base_sha256": _digest(base_bytes),
            "source_full_sha256": _digest(full_bytes), "source_selection_sha256": SELECTION_SHA256 if profile != "original" else None,
            "new_latin_pieces": False, "unicode_case_decomposition_additions": False,
            "addition_policy": "Indic and shared support additions only, excluding the rare Latin inventory"},
        "profile_policy": policy,
    }
    files = {"base-tokenizer.model": base_bytes, "full-tokenizer.model": full_bytes,
             "tokenizer.model": model_bytes, "native-row-map.json": _json(mapping),
             "nemo-id-map.json": _json(public.to_dict()), "vocabulary.json": _json(_vocabulary(target, public)),
             "cleanup.json": _json(cleanup)}
    manifest = {
        "schema_version": 1, "algorithm": ALGORITHM, "profile": profile, "tokenizer_version": 4,
        "status": "tokenizer_profile_candidate", "structural_passed": True,
        "checkpoint_validated": False, "asr_validated": False,
        "requires_retokenized_training_labels": profile != "original",
        "native_vocabulary_size": size, "native_blank_id": size,
        "original_native_vocabulary_size": base_size,
        "original_native_entries_preserved": len(retained_base),
        "original_native_text_ids_unchanged": prefix_preserved,
        "original_normalizer_unchanged": True, "native_prefix_preserved": prefix_preserved,
        "original_model_byte_identical": model_bytes == base_bytes,
        "original_native_blank_id": base_size, "native_blank_id_unchanged": size == base_size,
        "public_pad_id": public.hf_pad_id, "public_blank_id": public.hf_blank_id,
        "base_tokenizer_sha256": _digest(base_bytes), "full_tokenizer_sha256": _digest(full_bytes),
        "tokenizer_sha256": _digest(model_bytes), "normalizer_sha256": _digest(base.normalizer_spec.SerializeToString()),
        "files": {name: _digest(raw) for name, raw in files.items()},
    }
    return files, manifest, mapping, public


def build_clean_bundles(bundle: str | Path, output: str | Path):
    """Create the four profiles from their immutable source inventory."""
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
    """Validate v4 profiles or historical v3 bundles against their exact recipe."""

    def __init__(self, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("algorithm") not in {ALGORITHM, LEGACY_ALGORITHM}:
            raise ValueError("Expected a versioned native tokenizer profile")
        profile = manifest.get("profile", "")
        names = {"base-tokenizer.model", "full-tokenizer.model", "tokenizer.model", "native-row-map.json",
                 "nemo-id-map.json", "vocabulary.json", "cleanup.json"}
        if set(manifest.get("files", {})) != names:
            raise ValueError("Incomplete tokenizer profile manifest")
        actual = {name: (directory / name).read_bytes() for name in names}
        if any(_digest(raw) != manifest["files"][name] for name, raw in actual.items()):
            raise ValueError("Tokenizer profile file hash mismatch")
        build = _preserved_v3_artifacts if manifest["algorithm"] == LEGACY_ALGORITHM else _artifacts
        expected, expected_manifest, mapping, public = build(
            actual["base-tokenizer.model"], actual["full-tokenizer.model"], profile)
        if actual != expected or manifest != expected_manifest:
            raise ValueError("Tokenizer profile artifacts differ from the pinned cleanup recipe")
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
