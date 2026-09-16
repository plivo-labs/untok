"""The fixed checkpoint aliases and added identities must never be reallocated."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from untok.prompts import prompt_registry


@pytest.fixture
def pinned():
    return json.loads((Path(__file__).parents[1] / "src/untok/data/prompt-registry.json").read_text())


@pytest.mark.parametrize("profile,count", [("original", 121), ("latin", 57), ("latin-indic", 81), ("full", 133)])
def test_released_prompt_assignments_are_unchanged(pinned, profile, count):
    registry = prompt_registry(SimpleNamespace(**pinned["source"]), profile)
    assert registry == pinned["profiles"][profile]
    dictionary = registry["prompt_dictionary"]
    assert len(dictionary) == count
    assert dictionary["auto"] == 101
    assert dictionary["en"] == dictionary["en-US"] == 0
    assert dictionary["enGB"] == dictionary["en-GB"] == 1
    if profile != "latin":
        assert dictionary["hi"] == dictionary["hi-IN"] == dictionary["hi-HI"] == 6
    for name, slot in pinned["source"]["prompt_dictionary"].items():
        if name in dictionary:
            assert dictionary[name] == slot
    if profile in {"latin-indic", "full"}:
        assert len(registry["target_assignments"]) == 22
        assert registry["identity_assignments"] == pinned["profiles"]["full"]["identity_assignments"]


def test_changed_source_and_unknown_profile_rejected(pinned):
    changed = copy.deepcopy(pinned["source"])
    changed["prompt_dictionary"]["hi"] = 7
    with pytest.raises(ValueError, match="differs from original"):
        prompt_registry(SimpleNamespace(**changed), "full")
    with pytest.raises(ValueError, match="Unknown"):
        prompt_registry(SimpleNamespace(**pinned["source"]), "legacy")
