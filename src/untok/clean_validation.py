"""Validate append-only cleanup while preserving the original Nemotron model."""
from __future__ import annotations

import hashlib
from importlib import resources
import json
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from .clean import ALGORITHM, CleanTokenizerAdapter
from .unigram import native_id_map, validate_native_prefix
from .unigram_validation import (
    _CORPUS_LIMITS, _contained, _corpus_metrics, _inventory_quality,
    _json, _policy, _sha, _training_usage, _witness,
)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _source_selection(manifest: dict) -> tuple[dict, str]:
    """Bind retained fitting roles to the original packaged selection."""
    source = resources.files("untok").joinpath("data", "source")
    source_manifest = json.loads(source.joinpath("manifest.json").read_text(encoding="utf-8"))
    raw = source.joinpath("selection.json").read_bytes()
    digest = _digest(raw)
    if (source_manifest.get("tokenizer_sha256") != manifest["full_tokenizer_sha256"]
            or source_manifest.get("base_tokenizer_sha256") != manifest["base_tokenizer_sha256"]
            or source_manifest.get("files", {}).get("selection.json") != digest):
        raise ValueError("Clean bundle does not match the packaged original selection")
    selection = json.loads(raw)
    if selection.get("base_tokenizer_sha256") != manifest["base_tokenizer_sha256"]:
        raise ValueError("Original selection base hash disagrees with clean bundle")
    return selection, digest


def _roles(source: dict, adapter, cleanup: dict) -> tuple[dict, dict]:
    from .bundles import character_allowed

    proc = adapter.backend
    normalizer = spm.SentencePieceNormalizer(
        model_proto=adapter.model_bytes, add_dummy_prefix=False,
        escape_whitespaces=True, remove_extra_whitespaces=False,
    )
    retained = set(adapter.vocab)
    optional, required, required_text = set(), set(), {}
    changed_roles = []
    for entry in source["additions"]:
        piece = entry["piece"]
        canonical = normalizer.normalize(piece.replace("▁", " "))
        if entry.get("required_reasons"):
            if not canonical or not all(character_allowed(c, adapter.profile) for c in canonical):
                continue
            required_text[canonical] = proc.unk_id() not in proc.encode(canonical.replace("▁", " "))
            if canonical in retained:
                required.add(canonical)
            if canonical != piece or canonical not in retained:
                changed_roles.append({"source_piece": piece, "canonical": canonical,
                                      "retained_whole_piece": canonical in retained})
        elif piece in retained:
            optional.add(piece)
    # New case-preserving Latin character coverage is
    # required even when the old frozen training mix never contained it.
    required.update(entry["piece"] for entry in cleanup["added"])
    optional -= required
    selection = {"data_manifest_sha256": source.get("data_manifest_sha256"),
                 "additions": [{"piece": piece, "required_reasons": [{"kind": "clean_coverage"}]} for piece in sorted(required)]
                 + [{"piece": piece} for piece in sorted(optional)]}
    return selection, {"required_piece_witnesses": {piece: _witness(proc, piece) for piece in sorted(required)},
                       "canonical_required_text": required_text,
                       "canonicalized_required_roles": changed_roles,
                       "required_piece_count": len(required), "optional_piece_count": len(optional),
                       "classification": "Retained exact optional fitted strings require fresh usage; canonical required strings require coverage, and retained required pieces require witnesses. Original upstream rows are not required to occur in the Indic training mix."}


def validate_clean_tokenizer(
    bundle_path: str | Path,
    policy_path: str | Path,
    corpus_manifest_path: str | Path | None = None,
    *,
    phase: str = "dev",
    selection_receipt_path: str | Path | None = None,
    max_examples: int = 0,
) -> dict:
    """Verify native identity, addition quality and fresh target corpus evidence.

    Every original Nemotron piece message and model setting is immutable except
    the enlarged vocabulary length. Inherited inventory limitations are reported
    independently; additions must pass every quality gate. Training text is opened
    for usage checks, and reserve access requires an exact artifact receipt.
    """
    if phase not in {"dev", "reserve"} or isinstance(max_examples, bool) or not isinstance(max_examples, int) or max_examples < 0:
        raise ValueError("Use phase dev/reserve and a nonnegative integer max_examples")
    bundle, policy_file = Path(bundle_path), Path(policy_path)
    adapter = CleanTokenizerAdapter(bundle)
    manifest = _json(bundle / "manifest.json")
    policy = _policy(policy_file)
    for field in ("base_tokenizer_sha256", "tokenizer_sha256", "normalizer_sha256"):
        if field in policy and policy[field] != manifest[field]:
            raise ValueError(f"Policy {field} disagrees with clean bundle")
    proc = adapter.backend
    source_proc = spm.SentencePieceProcessor(model_proto=adapter.full_model_bytes)
    model, source_base = pb.ModelProto(), pb.ModelProto()
    model.ParseFromString(adapter.model_bytes)
    source_base.ParseFromString(adapter.base_model_bytes)
    prefix = validate_native_prefix(adapter.base_model_bytes, adapter.model_bytes)
    base_size = prefix["native_entries_preserved"]
    source_normalizer_sha = _digest(source_base.normalizer_spec.SerializeToString())
    cleanup = _json(bundle / "cleanup.json")
    source_selection, selection_sha = _source_selection(manifest)
    selection, roles = _roles(source_selection, adapter, cleanup)
    quality = _inventory_quality(adapter.model_bytes, set(range(base_size, len(model.pieces))))
    inherited_quality = _inventory_quality(adapter.model_bytes, set(range(base_size)))
    original_quality = _inventory_quality(adapter.base_model_bytes)
    coverage = {lang: {"script": entry.get("script"), "required_characters": len(entry["characters"]),
                       "missing": [char for char in entry["characters"] if proc.unk_id() in proc.encode(char)]}
                for lang, entry in policy["profiles"].items()}
    probes = [{"input": text, "source": source_proc.normalize(text), "candidate": proc.normalize(text),
               "changed": source_proc.normalize(text) != proc.normalize(text)}
              for text in policy.get("normalizer_probes", [])]
    expected_mapping = native_id_map(adapter.base_model_bytes, adapter.model_bytes)
    structural_gates = {"reproducible_clean_artifacts": True,
                        "all_original_piece_messages_preserved": True,
                        "all_original_model_metadata_preserved": True,
                        "original_normalizer_unchanged": manifest["normalizer_sha256"] == source_normalizer_sha
                        and not any(probe["changed"] for probe in probes),
                        "required_alphabets": not any(row["missing"] for row in coverage.values()),
                        "public_layout": adapter.id_map == expected_mapping
                        and adapter.id_map.hf_pad_id == base_size
                        and adapter.id_map.hf_blank_id == base_size + 1}
    report = {"algorithm": ALGORITHM, "tokenizer_version": 3, "phase": phase,
              "profile": adapter.profile, "tokenizer_sha256": manifest["tokenizer_sha256"],
              "base_tokenizer_sha256": manifest["base_tokenizer_sha256"],
              "source_tokenizer_sha256": manifest["full_tokenizer_sha256"],
              "normalizer_sha256": manifest["normalizer_sha256"],
              "source_normalizer_sha256": source_normalizer_sha,
              "policy_sha256": _sha(policy_file), "selection_sha256": selection_sha,
              "source_selection_sha256": selection_sha,
              "bundle_manifest_sha256": _sha(bundle / "manifest.json"),
              "sentencepiece_version": spm.__version__, "alphabet_coverage": coverage,
              "normalization_controls": probes, "inventory_quality": quality,
              "inventory_quality_scope": "Only appended entries are release-gated; original Nemotron entries remain byte-identical.",
              "inherited_inventory_quality": inherited_quality,
              "original_base_inventory_quality": original_quality,
              "inherited_inventory_policy": "Report inherited normalization aliases and dominated pieces without changing original IDs, scores, types or normalization.",
              "native_prefix": prefix,
              "selection_quality": roles, "gates": structural_gates,
              "structural_passed": all(structural_gates.values()),
              "native_vocabulary_size": proc.get_piece_size(), "native_blank_id": adapter.blank_id,
              "public_pad_id": adapter.id_map.hf_pad_id, "public_blank_id": adapter.id_map.hf_blank_id,
              "corpus_status": "incomplete", "corpus_gates": {}, "corpora": {},
              "missing_profiles": sorted(policy["profiles"]), "empty_profiles": [],
              "old_ID_compatibility": True,
              "old_ID_compatibility_scope": "Original Nemotron text IDs only; expanded v1/v2 additions and native RNNT blank require their explicit mappings.",
              "original_text_id_range": [0, base_size - 1],
              "original_public_pad_and_blank_preserved": True,
              "requires_retokenized_training_labels": True,
              "training_label_scope": "Original Nemotron text labels retain their meanings; regenerate labels from text to use the new segmentation. Older expanded bundle labels require migration or retokenization."}
    manifest_path = files = None
    if corpus_manifest_path is not None:
        manifest_path = Path(corpus_manifest_path)
        corpus = _json(manifest_path)
        if corpus.get("native_base_sha256") != manifest["base_tokenizer_sha256"]:
            raise ValueError("Corpus manifest native base hash disagrees with source bundle")
        if corpus.get("normalizer_sha256") != source_normalizer_sha:
            raise ValueError("Corpus manifest normalizer hash disagrees with source bundle")
        files = corpus.get("files")
        if not isinstance(files, dict):
            raise ValueError("Corpus manifest must declare relative file SHA256 digests")
        report["data_manifest_sha256"] = _sha(manifest_path)
        if phase == "reserve":
            if selection_receipt_path is None:
                raise ValueError("Reserve evaluation requires a selection receipt before reading data")
            receipt_path = Path(selection_receipt_path)
            receipt = _json(receipt_path)
            for field in ("tokenizer_sha256", "data_manifest_sha256", "bundle_manifest_sha256", "selection_sha256", "policy_sha256"):
                if receipt.get(field) != report[field]:
                    raise ValueError(f"Selection receipt does not bind current artifact: {field}")
            report["selection_receipt_sha256"] = _sha(receipt_path)
            report["receipt_scope"] = "Binds current v3 artifact identity; does not independently prove creation chronology."
        for name in files:
            if not isinstance(name, str):
                raise ValueError("Corpus paths must be strings")
            _contained(manifest_path.parent, name)
        inputs, missing = {}, []
        for lang in sorted(policy["profiles"]):
            name = f"{phase}/{lang}.jsonl"
            path = _contained(manifest_path.parent, name)
            if name not in files or not path.is_file():
                missing.append(lang)
                continue
            if not isinstance(files[name], str) or _sha(path) != files[name]:
                raise ValueError(f"Corpus integrity failure: {name}")
            inputs[lang] = path
        report["corpora"] = {lang: _corpus_metrics(path, source_proc, proc, model,
                             0 if phase == "reserve" else max_examples, lang, expected_processor=proc)
                             for lang, path in inputs.items()}
        empty = [lang for lang, metrics in report["corpora"].items()
                 if not metrics["records"] or not metrics["nonempty_normalized_records"]]
        settings = policy.get("corpus_thresholds", {}).get(phase, {})
        checks = {}
        for lang, metrics in report["corpora"].items():
            limits = {**settings.get("default", {}), **settings.get("profiles", {}).get(lang, {})}
            checks[lang] = {name: {"actual": metrics[_CORPUS_LIMITS[name]], "maximum": limit,
                                  "passed": metrics[_CORPUS_LIMITS[name]] is not None
                                  and metrics[_CORPUS_LIMITS[name]] <= limit}
                            for name, limit in limits.items()}
        gates = {"all_profiles_have_text": not missing and not empty,
                 "all_representable_roundtrips_pass": not any(v["roundtrip_failures"] for v in report["corpora"].values()),
                 "all_corpus_normalizers_match_target": not any(v["normalizer_failures"] for v in report["corpora"].values()),
                 "configured_release_thresholds": all(check["passed"] for limits in checks.values() for check in limits.values())}
        report.update({"corpus_threshold_checks": checks, "corpus_gates": gates,
                       "missing_profiles": missing, "empty_profiles": empty,
                       "corpus_status": "incomplete" if missing or empty else ("passed" if all(gates.values()) else "failed")})
    usage = _training_usage(selection, proc, policy["profiles"], manifest_path, files,
                            report.get("data_manifest_sha256"))
    for row in usage["additions"]:
        if row["required"] and row["training_occurrences"] and roles["required_piece_witnesses"][row["piece"]] is None:
            roles["required_piece_witnesses"][row["piece"]] = {
                "kind": "verified_training_usage", "native_id": row["native_id"],
                "occurrences": row["training_occurrences"], "data_manifest_sha256": usage["data_manifest_sha256"]}
    roles["required_unwitnessed"] = sorted(piece for piece, witness in roles["required_piece_witnesses"].items() if witness is None)
    roles["uncovered_required_text"] = sorted(piece for piece, covered in roles["canonical_required_text"].items() if not covered)
    quality_gates = {**quality["gates"], "required_piece_witnesses": not roles["required_unwitnessed"],
                     "canonical_required_text_coverage": not roles["uncovered_required_text"],
                     "optional_training_usage": None if usage["status"] == "incomplete" else usage["status"] == "passed"}
    quality_status = "failed" if False in quality_gates.values() else usage["status"]
    statuses = ["passed" if report["structural_passed"] else "failed", report["corpus_status"], quality_status]
    passed = all(status == "passed" for status in statuses)
    report.update({"training_usage": usage, "training_usage_status": usage["status"],
                   "selection_quality_gates": quality_gates, "selection_quality_status": quality_status,
                   "passed": passed, "status": "failed" if "failed" in statuses else ("passed" if passed else "incomplete"),
                   "checkpoint_validated": False, "asr_validated": False,
                   "scope": "V3 text-tokenizer evidence with unchanged original Nemotron IDs, piece messages and normalization. Appended pieces can change segmentation; no acoustic accuracy or completed checkpoint validation claim.",
                   "corpus_unknown_policy": "Explicit per-phase policy thresholds govern unknowns and token efficiency; unconfigured metrics are reported only.",
                   "validator_sha256": _sha(Path(__file__)),
                   "shared_validator_sha256": _sha(Path(__file__).with_name("unigram_validation.py"))})
    return report
