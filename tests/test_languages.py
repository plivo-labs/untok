"""Every Indic language of latin-indic has a script, a prompt slot and (for new scripts) its rows."""
import pytest

from untok.bundles import load_tokenizer_bundle
from untok.languages import (LANGUAGE_SCRIPT, SCRIPT_BLOCKS, new_scripts, original_size, piece_script,
                             registry_languages, resolve_languages, script_rows)
from untok.prompts import profile_registry


@pytest.fixture(scope="module")
def adapter():
    return load_tokenizer_bundle("latin-indic")


@pytest.fixture(scope="module")
def registry():
    return profile_registry("latin-indic")


def test_script_rows_match_the_profile(adapter):
    counts = {s: len(script_rows(adapter, s)) for s in SCRIPT_BLOCKS}
    assert counts["Tamil"] == counts["Ol_Chiki"] == 500
    for script in ("Bengali", "Gurmukhi", "Gujarati", "Oriya", "Telugu", "Kannada", "Malayalam", "Meetei_Mayek"):
        assert counts[script] == 499
    devanagari = script_rows(adapter, "Devanagari")
    assert len(devanagari) == 1528 and sum(i < original_size(adapter) for i in devanagari) == 196


def test_new_scripts_are_the_ones_nvidia_never_wrote(adapter):
    assert set(new_scripts(adapter)) == {"Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu", "Kannada",
                                         "Malayalam", "Ol_Chiki", "Meetei_Mayek"}


def test_every_registry_language_has_a_script_and_slot(registry):
    languages = registry_languages(registry)
    assert len(languages) == 22
    assert {info["language"] for info in languages.values()} == set(LANGUAGE_SCRIPT)
    assert languages["ta-IN"]["prompt_slot"] == 39 and languages["te-IN"]["prompt_slot"] == 40
    assert languages["hi-IN"]["prompt_slot"] == 6 and languages["pa-IN"]["script"] == "Gurmukhi"
    assert len({info["prompt_slot"] for info in languages.values()}) == 22


def test_language_requests_resolve_to_locales(registry):
    assert resolve_languages(registry, "all") == tuple(registry_languages(registry))
    assert resolve_languages(registry, "ta, te-IN,bn,ta") == ("ta-IN", "te-IN", "bn-IN")
    with pytest.raises(ValueError):
        resolve_languages(registry, "fr")


def test_piece_script_ignores_word_boundary_and_latin():
    assert piece_script("▁தமிழ்") == "Tamil"
    assert piece_script("‍లో") == "Telugu"
    assert piece_script("▁") is None and piece_script("hello") is None
