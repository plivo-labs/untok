"""Pinned, language-level text diagnostics for the three shipped vocabularies.

No fitting, transcript cleanup, audio download, or claim of donor-training
independence is performed here. Corpus selection does not load a tokenizer.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import unicodedata
import urllib.request

from sentencepiece import sentencepiece_model_pb2 as pb

from .bundles import PROFILES, load_tokenizer
from .unigram_validation import _expected


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _policy(path: Path) -> dict:
    value = _read(path)
    if value.get("schema_version") != 1 or not isinstance(value.get("profiles"), dict) or not value["profiles"]:
        raise ValueError("Language policy requires schema_version=1 and profiles")
    if not re.fullmatch(r"[a-f0-9]{40}", value.get("fleurs_revision", "")):
        raise ValueError("FLEURS revision must be an immutable 40-character commit")
    count = value.get("records_per_language")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("records_per_language must be positive")
    for language, spec in value["profiles"].items():
        if not re.fullmatch(r"[a-z]{2,3}", language):
            raise ValueError("Invalid language profile")
        if not isinstance(spec, dict) or not spec.get("expected_bundles") or not set(spec["expected_bundles"]) <= set(PROFILES):
            raise ValueError("Invalid bundle membership")
        if spec.get("source_kind") not in {"fleurs_dev", "frozen_indic"}:
            raise ValueError("Unknown language evidence source")
        if spec["source_kind"] == "fleurs_dev":
            if not re.fullmatch(r"[a-z0-9_]+", spec.get("fleurs_config", "")):
                raise ValueError("Invalid FLEURS configuration")
            if any(not re.fullmatch(r"[a-f0-9]{64}", spec.get(field, "")) for field in ("source_sha256", "selected_sha256")):
                raise ValueError("Each FLEURS source and selected corpus needs a SHA256 pin")
    return value


def select_fleurs_records(raw: bytes, language: str, count: int) -> tuple[list[dict], dict]:
    """Choose distinct publisher-normalized texts before loading any model.

    Raw and normalized transcript fields are preserved byte-for-byte after TSV
    decoding. NFC/whitespace collapsing is used only as a deduplication key.
    """
    unique = {}
    total = empty = 0
    for line, row in enumerate(csv.reader(io.StringIO(raw.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE), 1):
        total += 1
        if len(row) != 7:
            raise ValueError(f"Unexpected FLEURS TSV schema at row {line}")
        record_id, filename, raw_text, normalized, _, samples, gender = row
        if Path(filename).name != filename or not samples.isdecimal():
            raise ValueError(f"Malformed FLEURS metadata at row {line}")
        key = " ".join(unicodedata.normalize("NFC", normalized).split())
        if not key or not raw_text.strip():
            empty += 1
            continue
        identity = _sha(key.encode())
        record = {"record_id": f"{record_id}/{filename}", "source_row": line, "language": language,
                  "text": raw_text, "publisher_normalized_text": normalized, "dedup_sha256": identity}
        # Choosing a representative and sample order uses source identity/text
        # only, never unknown counts, segmentation, or the candidate normalizer.
        if identity not in unique or record["record_id"] < unique[identity]["record_id"]:
            unique[identity] = record
    if len(unique) < count:
        raise ValueError(f"{language}: {len(unique)} distinct transcripts, need {count}")
    selected = [unique[key] for key in sorted(unique)[:count]]
    return selected, {"source_rows": total, "empty_rows": empty, "distinct_publisher_texts": len(unique),
                      "selected_records": count, "selection": "First SHA256-ranked distinct NFC/whitespace publisher-normalized texts; source-ID representative"}


def _write_immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to replace different frozen data: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def prepare_fleurs_corpus(policy_path: str | Path, cache: str | Path, output: str | Path, *, download: bool = True) -> dict:
    """Fetch only SHA-pinned public TSV metadata and freeze balanced text rows."""
    policy_path, cache, output = Path(policy_path), Path(cache), Path(output)
    policy = _policy(policy_path)
    manifest = {"schema_version": 1, "policy_sha256": _sha(policy_path.read_bytes()),
                "dataset": "google/fleurs", "revision": policy["fleurs_revision"], "split": "dev",
                "license": "CC-BY-4.0", "files": {}, "languages": {}}
    for language, spec in sorted(policy["profiles"].items()):
        if spec["source_kind"] != "fleurs_dev":
            continue
        config = spec["fleurs_config"]
        url = f"https://huggingface.co/datasets/google/fleurs/resolve/{policy['fleurs_revision']}/data/{config}/dev.tsv"
        source_path = cache / f"{config}.tsv"
        if source_path.is_file():
            raw = source_path.read_bytes()
        elif download:
            with urllib.request.urlopen(url, timeout=45) as response:
                raw = response.read(10_000_001)
            if len(raw) > 10_000_000:
                raise ValueError("Oversized FLEURS metadata response")
        else:
            raise ValueError(f"Missing pinned metadata: {source_path}")
        if _sha(raw) != spec["source_sha256"]:
            raise ValueError(f"FLEURS source integrity failure: {config}")
        _write_immutable(source_path, raw)
        rows, sampling = select_fleurs_records(raw, language, policy["records_per_language"])
        relative = f"fleurs/{language}.jsonl"
        data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
        if _sha(data) != spec["selected_sha256"]:
            raise ValueError(f"Frozen selection integrity failure: {language}")
        _write_immutable(output / relative, data)
        manifest["files"][relative] = _sha(data)
        manifest["languages"][language] = {"path": relative, "source_url": url,
            "source_sha256": _sha(raw), "source_bytes": len(raw), "sampling": sampling}
    _write_immutable(output / "manifest.json", _bytes(manifest))
    return manifest


def _rows(root: Path, relative: str, digest: str, language: str) -> list[dict]:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Corpus paths must remain relative and contained")
    data = path.read_bytes()
    if _sha(data) != digest:
        raise ValueError(f"Corpus integrity failure: {relative}")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    for row in rows:
        if not isinstance(row, dict) or row.get("language") != language or not isinstance(row.get("text"), str):
            raise ValueError(f"Invalid corpus record in {relative}")
    return rows


def text_metrics(texts: list[str], reference, processor, reference_proto) -> dict:
    """Counts are normalized text diagnostics; unknown rows cannot round-trip."""
    totals = Counter({key: 0 for key in ("records", "tokens", "normalized_characters", "unknown_records",
                     "unknown_tokens", "unknown_codepoint_occurrences", "roundtrip_checked", "roundtrip_failures", "normalizer_failures")})
    unknown = Counter()
    ratios = []
    lengths = []
    for text in texts:
        normal = reference.normalize(text)
        ids = processor.encode(text)
        n_unknown = ids.count(processor.unk_id())
        totals.update(records=1, tokens=len(ids), normalized_characters=len(normal),
                      unknown_records=int(n_unknown > 0), unknown_tokens=n_unknown,
                      normalizer_failures=int(processor.normalize(text) != normal))
        lengths.append(len(ids))
        ratios.append(len(ids) / max(1, len(normal)))
        if n_unknown:
            for piece in processor.encode_as_immutable_proto(text).pieces:
                if piece.id == processor.unk_id():
                    unknown.update(piece.piece)
                    totals["unknown_codepoint_occurrences"] += len(piece.piece)
        else:
            totals["roundtrip_checked"] += 1
            totals["roundtrip_failures"] += processor.decode(ids) != _expected(reference, reference_proto, text)
    quantile = lambda values, q: sorted(values)[max(0, math.ceil(q * len(values)) - 1)] if values else None
    return {**totals, "unknown_record_rate": totals["unknown_records"] / max(1, totals["records"]),
            "unknown_token_rate": totals["unknown_tokens"] / max(1, totals["tokens"]),
            "unknown_codepoints": {f"U+{ord(char):04X}": count for char, count in sorted(unknown.items())},
            "mean_tokens": sum(lengths) / len(lengths) if lengths else None,
            "tokens_p95": quantile(lengths, .95), "tokens_p99": quantile(lengths, .99),
            "mean_tokens_per_normalized_character": sum(ratios) / len(ratios) if ratios else None,
            "tokens_per_normalized_character_p95": quantile(ratios, .95),
            "tokens_per_normalized_character_p99": quantile(ratios, .99)}


def validate_language_coverage(policy_path: str | Path, corpus_manifest_path: str | Path,
                               frozen_indic_manifest_path: str | Path | None = None) -> dict:
    """Evaluate every supplied language against all bundles, preserving scope.

    Original Indic development/reserve data remain a separate evidence family;
    their hashes must match the original freeze pin. No raw text is reported.
    """
    policy_path, corpus_path = Path(policy_path), Path(corpus_manifest_path)
    policy, corpus = _policy(policy_path), _read(corpus_path)
    if corpus.get("policy_sha256") != _sha(policy_path.read_bytes()):
        raise ValueError("Prepared corpus does not bind the current language policy")
    adapters = {name: load_tokenizer(name) for name in PROFILES}
    reference = adapters["full"].backend
    proto = pb.ModelProto()
    proto.ParseFromString(adapters["full"].model_bytes)
    frozen_path = Path(frozen_indic_manifest_path) if frozen_indic_manifest_path else None
    frozen = None
    if frozen_path is not None:
        if _sha(frozen_path.read_bytes()) != policy.get("frozen_indic_manifest_sha256"):
            raise ValueError("Indic corpus does not match the original frozen manifest")
        frozen = _read(frozen_path)
    report = {"schema_version": 1, "policy_sha256": _sha(policy_path.read_bytes()),
              "corpus_manifest_sha256": _sha(corpus_path.read_bytes()),
              "frozen_indic_manifest_sha256": _sha(frozen_path.read_bytes()) if frozen_path else None,
              "model_sha256": {name: _sha(adapter.model_bytes) for name, adapter in adapters.items()},
              "languages": {}, "missing_profiles": [], "coverage_gap_profiles": [],
              "roundtrip_or_normalizer_failure_profiles": [], "scope": policy["scope"]}
    english = []
    if "en" in corpus.get("languages", {}):
        relative = corpus["languages"]["en"]["path"]
        english = _rows(corpus_path.parent, relative, corpus["files"][relative], "en")
    for language, spec in sorted(policy["profiles"].items()):
        evidence = []
        diagnostic_rows = []
        inputs = []
        if spec["source_kind"] == "fleurs_dev" and language in corpus.get("languages", {}):
            entry = corpus["languages"][language]
            if entry.get("source_sha256") != spec["source_sha256"]:
                raise ValueError(f"Prepared source pin mismatch: {language}")
            relative = entry["path"]
            if corpus["files"].get(relative) != spec["selected_sha256"]:
                raise ValueError(f"Frozen selection pin mismatch: {language}")
            rows = _rows(corpus_path.parent, relative, corpus["files"][relative], language)
            if len(rows) != policy["records_per_language"] or len({r.get("dedup_sha256") for r in rows}) != len(rows):
                raise ValueError(f"Unbalanced or duplicate prepared rows: {language}")
            inputs = [("fleurs_dev", "raw_transcription", rows, "text"),
                      ("fleurs_dev", "publisher_normalized_transcription", rows, "publisher_normalized_text")]
            diagnostic_rows = rows
        elif spec["source_kind"] == "frozen_indic" and frozen is not None:
            for phase in ("dev", "reserve"):
                relative = f"{phase}/{language}.jsonl"
                if relative in frozen["files"]:
                    rows = _rows(frozen_path.parent, relative, frozen["files"][relative], language)
                    inputs.append((f"original_indic_{phase}", "source_annotation_cleaned_lexical_text", rows, "text"))
                    if phase == "dev":
                        diagnostic_rows = rows
        if not inputs or not all(rows for _, _, rows, _ in inputs):
            report["missing_profiles"].append(language)
        for source, view, rows, field in inputs:
            if any(not isinstance(row.get(field), str) for row in rows):
                raise ValueError(f"Missing transcript field: {language}/{view}")
            metrics = {name: text_metrics([row[field] for row in rows], reference, adapter.backend, proto)
                       for name, adapter in adapters.items()}
            evidence.append({"source": source, "text_view": view, "bundle_metrics": metrics})
        if language != "en" and diagnostic_rows and english:
            count = min(policy.get("synthetic_code_switch_records", 10), len(diagnostic_rows), len(english))
            texts = [diagnostic_rows[i]["text"] + " " + english[i]["text"] for i in range(count)]
            evidence.append({"source": "synthetic_code_switch", "text_view": "unaltered reference concatenation with English",
                "natural_code_switch_evidence": False,
                "bundle_metrics": {name: text_metrics(texts, reference, adapter.backend, proto) for name, adapter in adapters.items()}})
        in_scope = [item["bundle_metrics"][name] for item in evidence for name in spec["expected_bundles"]]
        if any(m["unknown_records"] for m in in_scope):
            report["coverage_gap_profiles"].append(language)
        if any(m["roundtrip_failures"] or m["normalizer_failures"] for m in in_scope):
            report["roundtrip_or_normalizer_failure_profiles"].append(language)
        report["languages"][language] = {"name": spec["name"], "expected_bundles": spec["expected_bundles"],
            "source_kind": spec["source_kind"], "limitations": spec.get("limitations", []), "evidence": evidence}
    report["status"] = ("failed" if report["roundtrip_or_normalizer_failure_profiles"] else
        "incomplete" if report["missing_profiles"] else
        "complete_with_coverage_gaps" if report["coverage_gap_profiles"] else "passed_on_supplied_text")
    report["passed"] = report["status"] == "passed_on_supplied_text"
    return report
