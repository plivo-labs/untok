"""Training-text policy: no tags; Indic without punctuation and with spelled-out numbers; English unchanged."""
import json

import pytest

from untok.text import normalize_manifest, normalize_text, spell_numbers


def test_tags_go_everywhere_and_english_keeps_case_punctuation_and_digits():
    assert normalize_text("Hello, World 42! <en-US>", "en-US") == ("Hello, World 42!", 0, 0)
    assert normalize_text("<unintelligible> வணக்கம்", "ta-IN")[0] == "வணக்கம்"


def test_indic_punctuation_is_removed():
    assert normalize_text("नमस्ते, दोस्त। कैसे हो? <hi-IN>", "hi-IN")[0] == "नमस्ते दोस्त कैसे हो"


def test_digits_are_spelled_in_the_rows_language():
    pytest.importorskip("indic_numtowords")
    text, spelled, left = normalize_text("मेरे पास 2,025 रुपये हैं।", "hi-IN")
    assert (spelled, left) == (1, 0) and "दो हज़ार पच्चीस" in text and "2" not in text
    native, spelled, _ = normalize_text("௨௦ பேர்", "ta-IN")  # native Tamil digits
    assert spelled == 1 and native.endswith("பேர்") and not any(c.isdigit() for c in native)
    assert "ఏడు" in normalize_text("007", "te-IN")[0]
    phone, spelled, left = normalize_text("मेरा नंबर 9876543210 है", "hi-IN")  # read digit by digit
    assert (spelled, left) == (1, 0) and "नौ आठ सात" in phone


def test_numbers_outside_the_library_keep_their_digits():
    assert spell_numbers("12 abc", "xx") == ("12 abc", 0, 1)


def test_manifest_normalisation_counts_and_drops_empty_rows(tmp_path):
    source, output = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    rows = [{"audio_filepath": "a.flac", "duration": 1.0, "text": "சரி. <ta-IN>", "target_lang": "ta-IN"},
            {"audio_filepath": "b.flac", "duration": 1.0, "text": "<unintelligible>", "target_lang": "ta-IN"},
            {"audio_filepath": "c.flac", "duration": 1.0, "text": "Okay.", "target_lang": "en-US"}]
    source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts = normalize_manifest(source, output)
    written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in written] == ["சரி", "Okay."]
    assert counts["rows"] == 3 and counts["dropped_empty"] == 1 and counts["changed"] == 1
    with pytest.raises(ValueError):
        normalize_manifest(source, output)
