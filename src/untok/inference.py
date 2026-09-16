"""Hash-verified audio and explicit prompt conditioning for native inference."""
from __future__ import annotations

import hashlib
from enum import Enum
import json
from pathlib import Path
from typing import Mapping


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    # OmegaConf ListConfig is not a built-in list.
    if type(value).__module__.startswith("omegaconf"):
        from omegaconf import OmegaConf
        return _plain(OmegaConf.to_container(value, resolve=True, enum_to_str=True))
    return value


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _load_audio_tensor(path: Path, expected_hash: str, sample_rate: int):
    """Decode hash-verified mono audio without resampling or changing its length."""
    import soundfile
    import torch

    if _sha256(path) != expected_hash:
        raise ValueError("Audio hash changed before transcription")
    samples, actual_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    if _sha256(path) != expected_hash:
        raise ValueError("Audio hash changed while loading transcription input")
    if actual_rate != sample_rate:
        raise ValueError(f"Audio sample rate {actual_rate} differs from model sample rate {sample_rate}; resampling is not implicit")
    if samples.ndim != 2 or samples.shape[1] != 1 or not samples.shape[0]:
        raise ValueError("Transcription requires nonempty mono audio")
    tensor = torch.from_numpy(samples[:, 0].copy())
    if not torch.isfinite(tensor).all():
        raise ValueError("Audio samples must be finite")
    return tensor


def _transcribe_with_verified_prompt(model, audio: Path, audio_sha256: str, target_lang: str):
    """Use tensor audio and verify the one-hot prompt entering the projection.

    The pinned file-path loader can select a random unified prompt. Tensor audio
    instead makes the prompt RNNT create indices from the explicit target_lang.
    Its direct self.forward() call bypasses model hooks, so inspect prompt_kernel.
    """
    import torch

    config = _plain(model.cfg)
    defaults = config.get("model_defaults", {})
    registry = defaults.get("prompt_dictionary", {})
    prompt_id = registry.get(target_lang)
    num_prompts = getattr(model, "num_prompts", None)
    encoder_hidden = defaults.get("enc_hidden")
    sample_rate = getattr(getattr(model, "preprocessor", None), "_sample_rate", None)
    if (not isinstance(prompt_id, int) or not isinstance(num_prompts, int)
            or not 0 <= prompt_id < num_prompts or not isinstance(encoder_hidden, int)
            or encoder_hidden <= 0 or not isinstance(sample_rate, int) or sample_rate <= 0):
        raise ValueError("Model lacks the expected prompt projection or audio configuration")
    kernel = getattr(model, "prompt_kernel", None)
    if not getattr(model, "concat", False) or not isinstance(kernel, torch.nn.Module):
        raise ValueError("Model has no active prompt projection to verify")
    waveform = _load_audio_tensor(Path(audio), audio_sha256, sample_rate)
    trcfg = model.get_transcribe_config()
    for key, value in {"batch_size": 1, "return_hypotheses": True, "num_workers": 0,
                       "verbose": False, "target_lang": target_lang,
                       "pad_min_duration": 0.0, "pad_direction": "right"}.items():
        setattr(trcfg, key, value)
    evidence = {"target_lang": target_lang, "prompt_id": prompt_id,
                "verified_kernel_calls": 0, "verified_frames": 0,
                "verification": "one_hot_input_to_prompt_kernel", "audio_input": "tensor",
                "sample_rate": sample_rate, "audio_samples": waveform.numel(),
                "pad_min_duration": 0.0}

    def verify_prompt(module, arguments):
        if (len(arguments) != 1 or not isinstance(arguments[0], torch.Tensor)
                or arguments[0].ndim != 3 or arguments[0].shape[0] != 1
                or arguments[0].shape[1] == 0
                or arguments[0].shape[-1] != encoder_hidden + num_prompts):
            raise ValueError("Unexpected prompt projection input shape")
        actual = arguments[0][..., encoder_hidden:]
        expected = torch.zeros_like(actual)
        expected[..., prompt_id] = 1
        if not torch.equal(actual, expected):
            raise ValueError(f"Actual conditioning prompt differs from requested {target_lang}")
        evidence["verified_kernel_calls"] += 1
        evidence["verified_frames"] += actual.shape[0] * actual.shape[1]

    handle = kernel.register_forward_pre_hook(verify_prompt)
    try:
        result = model.transcribe(audio=[waveform], batch_size=1, return_hypotheses=True,
                                  num_workers=0, verbose=False, target_lang=target_lang,
                                  override_config=trcfg)
    finally:
        handle.remove()
    if not evidence["verified_kernel_calls"]:
        raise ValueError("Prompt projection hooks did not execute; actual conditioning is unverified")
    return result, evidence
