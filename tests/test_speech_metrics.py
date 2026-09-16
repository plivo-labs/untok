"""Scoring contracts used by native checkpoint diagnostics."""
import pytest

from untok.speech_metrics import edit_distance, normalize_for_scoring


def test_scoring_only_collapses_whitespace():
    assert normalize_for_scoring(" \tA  e\u0301!\nक्\u200dष क्\u200cष ") == "A e\u0301! क्\u200dष क्\u200cष"
    assert normalize_for_scoring(" \t\n") == ""


@pytest.mark.parametrize("reference,hypothesis,errors", [
    ("one one two".split(), "one two".split(), 1),
    ("one two".split(), "one one two".split(), 1),
    ("one two".split(), "one three".split(), 1),
    ("क ख", "क ग", 1),
    ("", "ab", 2),
    ("ab", "", 2),
    ("", "", 0),
])
def test_word_and_character_edits_preserve_repetitions_and_insertions(reference, hypothesis, errors):
    assert edit_distance(reference, hypothesis) == errors
