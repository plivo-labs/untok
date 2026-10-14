"""Calibration reads token offsets off a probe and splits data by speaker."""
import pytest

from untok.calibration import _batches, offsets_from_probe, split_rows


def test_offsets_are_blank_relative_in_raw_feature_space():
    torch = pytest.importorskip("torch")
    vocab = [11, 12]  # probe classes: 0 = blank, 1 -> token 11, 2 -> token 12
    weight = torch.tensor([[1.0, 0.0, 0.0], [3.0, 1.0, 5.0], [0.0, 2.0, 7.0]])
    bias = torch.tensor([0.5, 1.0, 2.0])
    mean, std = torch.tensor([1.0, 2.0, 0.0]), torch.tensor([2.0, 1.0, 0.005])  # last unit is dead
    offsets, bias_offsets, info = offsets_from_probe(weight, bias, mean, std, vocab, [11, 12, 13])
    effective = weight.double() / std.double()
    effective[:, 2] = 0
    effective_bias = bias.double() - effective @ mean.double()
    assert torch.allclose(offsets[0], effective[1] - effective[0])
    assert torch.allclose(offsets[1], effective[2] - effective[0])
    assert float(bias_offsets[1]) == pytest.approx(float(effective_bias[2] - effective_bias[0]))
    assert torch.allclose(offsets[2], offsets[:2].mean(0))  # unseen token: mean offset, low bias
    assert float(bias_offsets[2]) == pytest.approx(float(bias_offsets[:2].min()) - 2.0)
    assert info == {"dead_units": 1, "missing_from_probe": 1}
    with pytest.raises(ValueError):
        offsets_from_probe(weight, bias, mean, std, vocab, [13])


def test_split_holds_out_whole_speakers_and_caps_hours():
    rows = [{"duration": 10.0, "text": "x", "speaker_id": f"s{i % 200}"} for i in range(2000)]
    rows.append({"duration": 30.0, "text": "too long", "speaker_id": "s1"})
    train, dev = split_rows(rows, hours=1.0, holdout_speakers=60, seed=0)
    assert {r["speaker_id"] for r in train}.isdisjoint({r["speaker_id"] for r in dev})
    assert len({r["speaker_id"] for r in dev}) == 60
    assert sum(r["duration"] for r in train) == pytest.approx(3600.0)
    assert all(r["duration"] <= 20 for r in train + dev)


def test_batches_respect_the_padded_frame_budget():
    lengths = [5, 50, 7, 90, 30, 30]
    batches = _batches(lengths, 100)
    assert sorted(i for b in batches for i in b) == list(range(6))
    assert all(max(lengths[i] for i in b) * len(b) <= 100 for b in batches)


def test_calibrate_skips_leftovers_of_scripts_nvidia_wrote(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from types import SimpleNamespace
    import untok.calibration as calibration

    class Calibrating(Exception):
        pass

    def to(device):
        raise Calibrating

    metadata = {"languages": {"mr-IN": {"script": "Devanagari", "rows": "initialized", "calibrated": "dormant_leftovers"},
                              "ta-IN": {"script": "Tamil", "rows": "initialized", "calibrated": False}},
                "scripts": {"Devanagari": {"rows": [22], "donor_script": "Devanagari"},
                            "Tamil": {"rows": [20, 21], "donor_script": "Devanagari"}}}
    model = SimpleNamespace(cfg=OmegaConf.create({"untok_languages": metadata}), to=to)
    monkeypatch.setattr(calibration, "restore_profile_checkpoint", lambda *args: (model, None, None, None))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32)
    with pytest.raises(ValueError, match="None of the manifests"):  # Marathi alone: nothing to calibrate
        calibration.calibrate("in.nemo", tmp_path / "a.nemo", source_sha256="s", manifests={"mr-IN": "mr.jsonl"})
    with pytest.raises(Calibrating):  # Tamil goes on to calibration
        calibration.calibrate("in.nemo", tmp_path / "b.nemo", source_sha256="s",
                              manifests={"mr-IN": "mr.jsonl", "ta-IN": "ta.jsonl"})
