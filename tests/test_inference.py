"""Native audio and prompt checks with explicit fake models and decoding."""
from enum import Enum
import json
import sys
import types

import pytest

from untok import inference


def test_runtime_configuration_enums_have_stable_json_names():
    class ScoreMode(Enum):
        KEEP = 1

    plain = inference._plain({"decoder": {"score_mode": ScoreMode.KEEP}, "modes": [ScoreMode.KEEP]})
    assert plain == {"decoder": {"score_mode": "KEEP"}, "modes": ["KEEP"]}
    assert json.loads(json.dumps(plain)) == plain


class FakeModel:
    def __init__(self, outputs=None):
        torch = pytest.importorskip("torch")
        self.outputs = outputs if outputs is not None else ["one one two"]
        self.calls = []
        self.cfg = {
            "model_defaults": {"prompt_dictionary": {"en-US": 0, "auto": 1}, "enc_hidden": 2},
            "decoding": {"strategy": "greedy_batch", "greedy": {"max_symbols": 10}},
            "encoder": {"att_context_size": [70, 13]},
        }
        self.tokenizer = types.SimpleNamespace(tokenizer=types.SimpleNamespace(serialized_model_proto=lambda: b"synthetic-tokenizer-unit-test"))
        self.preprocessor = types.SimpleNamespace(_sample_rate=16000)
        self.num_prompts, self.concat = 2, True
        self.prompt_kernel = torch.nn.Linear(4, 2)
        self.wrong_prompt = False
        self.skip_prompt = False

    def get_transcribe_config(self):
        return types.SimpleNamespace()

    def to(self, device):
        self.device = device
        return self

    def float(self):
        return self

    def eval(self):
        return self

    def transcribe(self, **kwargs):
        torch = pytest.importorskip("torch")
        self.calls.append(kwargs)
        assert isinstance(kwargs["audio"][0], torch.Tensor)
        assert kwargs["override_config"].target_lang == kwargs["target_lang"]
        assert kwargs["override_config"].pad_min_duration == 0
        if not self.skip_prompt:
            conditioning = torch.zeros(1, 3, 4)
            prompt_id = self.cfg["model_defaults"]["prompt_dictionary"][kwargs["target_lang"]]
            if self.wrong_prompt:
                prompt_id = 1 - prompt_id
            conditioning[..., 2 + prompt_id] = 1
            self.prompt_kernel(conditioning)
        output = self.outputs[len(self.calls) - 1]
        if isinstance(output, Exception):
            raise output
        return output if isinstance(output, list) else [types.SimpleNamespace(text=output)]


@pytest.mark.parametrize("failure", [None, "sample_rate", "stereo", "empty", "nonfinite", "changed_hash"])
def test_audio_loader_checks_decode_shape_rate_and_current_hash(tmp_path, monkeypatch, failure):
    """Mock soundfile decoding; exercise loader checks without adding test dependencies."""
    torch = pytest.importorskip("torch")
    path = tmp_path / "decoder-fixture.wav"
    path.write_bytes(b"explicit mock decoder fixture")
    digest = inference._sha256(path)
    samples = torch.zeros(320, 1)
    rate = 16000
    if failure == "sample_rate":
        rate = 8000
    elif failure == "stereo":
        samples = torch.zeros(320, 2)
    elif failure == "empty":
        samples = torch.zeros(0, 1)
    elif failure == "nonfinite":
        samples[0, 0] = float("nan")

    def read(filename, **kwargs):
        assert filename == str(path)
        assert kwargs == {"dtype": "float32", "always_2d": True}
        if failure == "changed_hash":
            path.write_bytes(b"changed during mocked decoding")
        return samples.numpy(), rate

    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(read=read))
    if failure:
        with pytest.raises(ValueError, match="sample rate|mono audio|finite|hash changed"):
            inference._load_audio_tensor(path, digest, 16000)
    else:
        tensor = inference._load_audio_tensor(path, digest, 16000)
        assert tensor.shape == (320,) and tensor.dtype == torch.float32


@pytest.mark.parametrize("target_lang,prompt_id", [("en-US", 0), ("auto", 1)])
def test_requested_prompt_reaches_actual_projection(tmp_path, monkeypatch, target_lang, prompt_id):
    torch = pytest.importorskip("torch")
    model = FakeModel()
    monkeypatch.setattr(inference, "_load_audio_tensor", lambda *args: torch.zeros(16000))
    result, evidence = inference._transcribe_with_verified_prompt(model, tmp_path / "audio.wav", "hash", target_lang)
    assert result[0].text == "one one two"
    assert evidence["prompt_id"] == prompt_id
    assert evidence["verified_kernel_calls"] == 1
    assert evidence["verified_frames"] == 3
    assert model.calls[0]["target_lang"] == target_lang
    assert not model.prompt_kernel._forward_pre_hooks


@pytest.mark.parametrize("failure", ["wrong_prompt", "skip_prompt", "transcribe_exception", "unknown_prompt"])
def test_prompt_failures_are_explicit_and_remove_hooks(tmp_path, monkeypatch, failure):
    torch = pytest.importorskip("torch")
    model = FakeModel()
    monkeypatch.setattr(inference, "_load_audio_tensor", lambda *args: torch.zeros(16000))
    target_lang = "en-US"
    error = ValueError
    pattern = "conditioning|hooks did not execute|prompt projection"
    if failure == "transcribe_exception":
        model.outputs = [RuntimeError("transcription failed")]
        error, pattern = RuntimeError, "transcription failed"
    elif failure == "unknown_prompt":
        target_lang = "missing"
    else:
        setattr(model, failure, True)
    with pytest.raises(error, match=pattern):
        inference._transcribe_with_verified_prompt(model, tmp_path / "audio.wav", "hash", target_lang)
    assert not model.prompt_kernel._forward_pre_hooks
