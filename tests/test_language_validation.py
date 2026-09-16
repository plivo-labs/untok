import hashlib
import json
from pathlib import Path

import pytest

from untok.language_validation import prepare_fleurs_corpus, select_fleurs_records, validate_language_coverage


def sha(data):
    return hashlib.sha256(data).hexdigest()


def selected_sha(data):
    rows, _ = select_fleurs_records(data, "en", 2)
    return sha("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def corpus(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    # Quotes are literal FLEURS text, not CSV field quoting. Repeated speech
    # recordings must not inflate the number of distinct text examples.
    data = '1\ta.wav\thello "world"\thello world\t0\t16000\tF\n2\tb.wav\thello again\thello again\t0\t16000\tM\n3\tc.wav\thello "world"\thello world\t0\t16000\tM\n'.encode()
    (cache / "en_us.tsv").write_bytes(data)
    policy = tmp_path / "policy.json"
    write(policy, {"schema_version": 1, "fleurs_revision": "a" * 40, "records_per_language": 2,
                   "scope": "test text diagnostics", "profiles": {
        "en": {"name": "English", "source_kind": "fleurs_dev", "fleurs_config": "en_us",
               "source_sha256": sha(data), "selected_sha256": selected_sha(data), "expected_bundles": ["latin", "latin-indic", "full"]}}})
    return policy, cache, tmp_path / "prepared", data


def test_sampling_preserves_literals_and_deduplicates_recordings(corpus):
    _, _, _, raw = corpus
    rows, result = select_fleurs_records(raw, "en", 2)
    assert len(rows) == 2 and result["source_rows"] == 3
    assert result["distinct_publisher_texts"] == 2
    assert any(row["text"] == 'hello "world"' for row in rows)
    assert len({row["dedup_sha256"] for row in rows}) == 2
    with pytest.raises(ValueError, match="distinct transcripts"):
        select_fleurs_records(raw, "en", 3)


def test_corrupt_source_rejected_before_preparing_any_text(corpus):
    policy, cache, output, _ = corpus
    (cache / "en_us.tsv").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="source integrity"):
        prepare_fleurs_corpus(policy, cache, output, download=False)
    assert not output.exists()


def test_prepared_data_is_immutable_and_does_not_need_network(corpus, monkeypatch):
    policy, cache, output, _ = corpus
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("Unexpected network request"))
    first = prepare_fleurs_corpus(policy, cache, output, download=False)
    assert prepare_fleurs_corpus(policy, cache, output, download=False) == first
    (output / "fleurs/en.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="Refusing to replace"):
        prepare_fleurs_corpus(policy, cache, output, download=False)


def test_evaluation_reports_three_bundles_without_transcript_leakage(corpus):
    policy, cache, output, _ = corpus
    prepare_fleurs_corpus(policy, cache, output, download=False)
    report = validate_language_coverage(policy, output / "manifest.json")
    assert report["status"] == "passed_on_supplied_text"
    evidence = report["languages"]["en"]["evidence"]
    assert len(evidence) == 2  # Two text forms, not independent corpora.
    assert set(evidence[0]["bundle_metrics"]) == {"latin", "latin-indic", "full"}
    for item in evidence:
        for metrics in item["bundle_metrics"].values():
            assert metrics["records"] == metrics["roundtrip_checked"] == 2
            assert metrics["unknown_records"] == metrics["roundtrip_failures"] == 0
    assert "hello" not in json.dumps(report)


def test_unknowns_are_visible_coverage_gaps_not_false_pass(corpus):
    policy, cache, output, _ = corpus
    raw = '1\ta.wav\thello 🦄\thello 🦄\t0\t16000\tF\n2\tb.wav\thello again\thello again\t0\t16000\tM\n'.encode()
    (cache / "en_us.tsv").write_bytes(raw)
    value = json.loads(policy.read_text())
    value["profiles"]["en"]["source_sha256"] = sha(raw)
    value["profiles"]["en"]["selected_sha256"] = selected_sha(raw)
    write(policy, value)
    prepare_fleurs_corpus(policy, cache, output, download=False)
    report = validate_language_coverage(policy, output / "manifest.json")
    assert report["status"] == "complete_with_coverage_gaps" and not report["passed"]
    assert report["coverage_gap_profiles"] == ["en"]
    metrics = report["languages"]["en"]["evidence"][0]["bundle_metrics"]["full"]
    assert metrics["unknown_records"] == metrics["roundtrip_checked"] == 1
    assert metrics["unknown_record_rate"] == .5
    assert "U+1F984" in metrics["unknown_codepoints"]


def test_missing_original_indic_evidence_is_not_silently_validated(corpus):
    policy, cache, output, _ = corpus
    value = json.loads(policy.read_text())
    value["profiles"]["hi"] = {"name": "Hindi", "source_kind": "frozen_indic", "expected_bundles": ["latin-indic", "full"]}
    write(policy, value)
    prepare_fleurs_corpus(policy, cache, output, download=False)
    report = validate_language_coverage(policy, output / "manifest.json")
    assert report["status"] == "incomplete" and report["missing_profiles"] == ["hi"]


def test_prepared_hash_tampering_and_policy_change_fail(corpus):
    policy, cache, output, _ = corpus
    prepare_fleurs_corpus(policy, cache, output, download=False)
    target = output / "fleurs/en.jsonl"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="Corpus integrity"):
        validate_language_coverage(policy, output / "manifest.json")
    value = json.loads(policy.read_text())
    value["scope"] = "changed policy"
    write(policy, value)
    with pytest.raises(ValueError, match="bind"):
        validate_language_coverage(policy, output / "manifest.json")


def test_packaged_policy_separates_inherited_and_original_indic_evidence():
    policy = json.loads((Path(__file__).parents[1] / "configs/language-coverage.json").read_text())
    assert len(policy["profiles"]) == 56
    assert sum(p["source_kind"] == "fleurs_dev" for p in policy["profiles"].values()) == 34
    assert sum(p["source_kind"] == "frozen_indic" for p in policy["profiles"].values()) == 22
    assert any("Nynorsk" in text for text in policy["profiles"]["no"]["limitations"])
