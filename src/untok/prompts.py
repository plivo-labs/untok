"""Pinned language-prompt assignments for the original checkpoint and four profiles."""
from __future__ import annotations

import hashlib
from importlib import resources
import json


REGISTRY_SHA256 = "21c8ebb91030a6347c3043c9e5b1818d9b7e940aa56505248cd064902a179525"


def _load_registry():
    data = resources.files("untok").joinpath("data", "prompt-registry.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != REGISTRY_SHA256:
        raise ValueError("Pinned prompt registry changed")
    return json.loads(data)


def profile_registry(profile):
    """A profile's reviewed prompt slots, for checkpoints that were already migrated to it."""
    registry = _load_registry()
    if profile not in registry["profiles"]:
        raise ValueError("Unknown tokenizer profile for prompt registry")
    return registry["profiles"][profile]


def prompt_registry(model_defaults, profile):
    """Use reviewed names and slots without reallocating any trained prompt."""
    registry = _load_registry()
    source = {"num_prompts": int(model_defaults.num_prompts),
              "prompt_dictionary": dict(model_defaults.prompt_dictionary)}
    if source != registry["source"]:
        raise ValueError("Source prompt dictionary differs from original Nemotron")
    if profile not in registry["profiles"]:
        raise ValueError("Unknown tokenizer profile for prompt registry")
    return registry["profiles"][profile]
