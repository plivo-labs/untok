"""Pinned language-prompt assignments for the original checkpoint and four profiles."""
from __future__ import annotations

import hashlib
from importlib import resources
import json


REGISTRY_SHA256 = "b8b36265160ab64140ef6adad5582c824624916e2256596bd0113ccfe47c9ffe"


def prompt_registry(model_defaults, profile):
    """Use reviewed names and slots without reallocating any trained prompt."""
    data = resources.files("untok").joinpath("data", "prompt-registry.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != REGISTRY_SHA256:
        raise ValueError("Pinned prompt registry changed")
    registry = json.loads(data)
    source = {"num_prompts": int(model_defaults.num_prompts),
              "prompt_dictionary": dict(model_defaults.prompt_dictionary)}
    if source != registry["source"]:
        raise ValueError("Source prompt dictionary differs from original Nemotron")
    if profile not in registry["profiles"]:
        raise ValueError("Unknown tokenizer profile for prompt registry")
    return registry["profiles"][profile]
