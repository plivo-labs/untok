"""New-language starts change only their rows and prompt columns, deterministically."""
import io
import tarfile

import pytest

from untok.checkpoint import RNNTLayout
from untok.language_init import (AUTO_SLOT, PROMPT_KEY, copy_prompt_columns, donor_statistics, identical_copies,
                                 init_language_state, place_rows, read_language_metadata, require_calibrated)

ENC, PROMPTS, HIDDEN, VOCAB = 6, 128, 5, 30
LAYOUT = RNNTLayout("decoder.embed.weight", "joint.out.weight", "joint.out.bias", VOCAB, VOCAB + 1)


def toy_state(torch):
    generator = torch.Generator().manual_seed(1)
    prompt = torch.randn(2 * ENC, ENC + PROMPTS, generator=generator) * 0.05
    prompt[:, ENC + AUTO_SLOT] = torch.randn(2 * ENC, generator=generator) * 3  # trained auto column
    prompt[:, ENC + 6] = torch.randn(2 * ENC, generator=generator) * 3  # trained hi column
    return {"decoder.embed.weight": torch.randn(VOCAB + 1, 4, generator=generator),
            "joint.out.weight": torch.randn(VOCAB + 1, HIDDEN, generator=generator),
            "joint.out.bias": torch.randn(VOCAB + 1, generator=generator),
            PROMPT_KEY: prompt, "encoder.weight": torch.randn(3, 3, generator=generator)}


PLAN = {"slots": [39, 40, 6], "scripts": {"Tamil": {"rows": [20, 21, 22], "donors": [2, 3, 4, 5, 6]},
                                          "Telugu": {"rows": [23, 24], "donors": [2, 3, 4, 5, 6]}}}


def test_rows_start_at_the_donor_mean_with_the_donor_spread():
    torch = pytest.importorskip("torch")
    state = toy_state(torch)
    weight, bias = state["joint.out.weight"].clone(), state["joint.out.bias"].clone()
    stats = donor_statistics(weight, bias, [2, 3, 4, 5, 6])
    offsets = torch.randn(3, HIDDEN, dtype=torch.float64)
    report = place_rows(weight, bias, [20, 21, 22], stats, offsets, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    new = weight[[20, 21, 22]].double()
    assert torch.allclose(new.mean(0), stats["mean"], atol=1e-5)
    assert abs(float((new - stats["mean"]).norm(dim=1).mean()) - stats["spread"]) < 1e-4
    assert abs(float(bias[[20, 21, 22]].double().mean()) - stats["bias_mean"]) < 1e-5
    assert report["rows"] == 3 and report["gamma"] > 0


def test_prompt_slots_copy_auto_but_keep_trained_slots():
    torch = pytest.importorskip("torch")
    prompt = toy_state(torch)[PROMPT_KEY]
    result = copy_prompt_columns(prompt, ENC, [39, 6])
    assert result == {39: "auto_copy", 6: "kept_trained"}
    assert torch.equal(prompt[:, ENC + 39], prompt[:, ENC + AUTO_SLOT])
    assert not torch.equal(prompt[:, ENC + 6], prompt[:, ENC + AUTO_SLOT])
    with pytest.raises(ValueError):
        copy_prompt_columns(prompt, ENC, [AUTO_SLOT])


def test_identical_copies_are_found_only_when_repeated():
    torch = pytest.importorskip("torch")
    weight = torch.randn(10, 4)
    weight[[3, 5, 7]] = weight[0] * 0 + 1.5  # three untrained copies among trained rows
    assert identical_copies(weight, [1, 3, 4, 5, 7, 9]) == [3, 5, 7]
    assert identical_copies(weight, [1, 2, 4]) == [] and identical_copies(weight, [3]) == []


def test_init_changes_only_new_rows_and_columns_and_is_deterministic():
    torch = pytest.importorskip("torch")
    state = toy_state(torch)
    rng_before = torch.random.get_rng_state()
    first, scripts, prompts = init_language_state(state, LAYOUT, ENC, PLAN, seed=3)
    second, _, _ = init_language_state(state, LAYOUT, ENC, PLAN, seed=3)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert set(scripts) == {"Tamil", "Telugu"} and prompts == {39: "auto_copy", 40: "auto_copy", 6: "kept_trained"}
    untouched = [i for i in range(VOCAB + 1) if i not in (20, 21, 22, 23, 24)]
    for key in ("decoder.embed.weight", "joint.out.weight", "joint.out.bias"):
        assert torch.equal(first[key][untouched], state[key][untouched])
        assert not torch.equal(first[key][20], state[key][20])
    assert torch.equal(first["encoder.weight"], state["encoder.weight"])
    donor_bias = float(state["joint.out.bias"][[2, 3, 4, 5, 6]].double().mean())
    donor_mean = state["joint.out.weight"][[2, 3, 4, 5, 6]].double().mean(0)
    for r in (20, 21, 22, 23, 24):  # dormant: exactly the donor mean, bias 8 lower, until calibrated
        assert abs(float(first["joint.out.bias"][r]) - (donor_bias - 8.0)) < 1e-5
        assert torch.allclose(first["joint.out.weight"][r].double(), donor_mean, atol=1e-6)
    assert torch.equal(state[PROMPT_KEY][:, ENC + 39], toy_state(torch)[PROMPT_KEY][:, ENC + 39])  # input untouched


def _nemo_with(tmp_path, text):
    path = tmp_path / "model.nemo"
    with tarfile.open(path, "w") as archive:
        data = text.encode()
        info = tarfile.TarInfo("./model_config.yaml")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return path


def test_training_refuses_uncalibrated_new_script_languages(tmp_path):
    pytest.importorskip("yaml")
    path = _nemo_with(tmp_path, """untok_languages:
  languages:
    ta-IN: {script: Tamil, rows: initialized, prompt_slot: 39, prompt_init: auto_copy, calibrated: false}
    mr-IN: {script: Devanagari, rows: trained, prompt_slot: 41, prompt_init: auto_copy, calibrated: not_required}
    te-IN: {script: Telugu, rows: initialized, prompt_slot: 40, prompt_init: auto_copy, calibrated: true}
    hi-IN: {script: Devanagari, rows: initialized, prompt_slot: 6, prompt_init: kept_trained, calibrated: dormant_leftovers}
""")
    metadata = read_language_metadata(path)
    require_calibrated(metadata, ["te-IN", "mr-IN", "hi-IN"])  # dormant leftovers learn from training targets
    with pytest.raises(ValueError, match="ta-IN"):
        require_calibrated(metadata, ["ta-IN", "te-IN"])
    with pytest.raises(ValueError, match="bn-IN"):
        require_calibrated(metadata, ["bn-IN"])
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="init-languages"):
        require_calibrated(read_language_metadata(_nemo_with(other, "a: 1\n")), ["ta-IN"])


def test_rerun_keeps_scripts_recorded_earlier():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from untok.language_init import build_plan
    import untok.language_init as language_init

    weight = torch.randn(40, 4)
    weight[[20, 21, 22]] = 1.0  # Tamil copies
    adapter = SimpleNamespace()
    rows = {"Tamil": [20, 21, 22], "Devanagari": [2, 3, 4]}
    registry = {"target_assignments": [{"language": "ta", "locale": "ta-IN", "prompt_id": 39},
                                       {"language": "hi", "locale": "hi-IN", "prompt_id": 6}]}
    original = (language_init.script_rows, language_init.original_size)
    language_init.script_rows = lambda _, script: rows.get(script, [])
    language_init.original_size = lambda _: 10
    try:
        fresh = build_plan(adapter, registry, ["ta-IN"], weight)
        assert fresh["scripts"]["Tamil"]["rows"] == [20, 21, 22] and fresh["languages"]["ta-IN"]["rows"] == "initialized"
        again = build_plan(adapter, registry, ["ta-IN"], weight, recorded_scripts={"Tamil"})
        assert again["scripts"] == {} and again["languages"]["ta-IN"]["rows"] == "initialized"
    finally:
        language_init.script_rows, language_init.original_size = original


def test_init_marks_leftovers_of_scripts_nvidia_wrote_dormant(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from types import SimpleNamespace
    import untok.language_init as language_init

    model = SimpleNamespace(cfg=OmegaConf.create({"model_defaults": {"enc_hidden": ENC}}),
                            state_dict=lambda: {LAYOUT.output_weight_key: torch.zeros(VOCAB + 1, HIDDEN)},
                            load_state_dict=lambda state, strict: None)
    plan = {"languages": {"ta-IN": {"prompt_slot": 39, "script": "Tamil", "rows": "initialized"},
                          "mr-IN": {"prompt_slot": 41, "script": "Devanagari", "rows": "initialized"},
                          "ur-PK": {"prompt_slot": 50, "script": "Arabic", "rows": "initialized"}},
            "scripts": {"Tamil": {"rows": [20, 21], "donors": [2, 3], "donor_script": "Devanagari"},
                        "Devanagari": {"rows": [22], "donors": [2, 3], "donor_script": "Devanagari"},
                        "Arabic": {"rows": [23, 24], "donors": [4, 5], "donor_script": "Arabic"}},
            "slots": [39, 41, 50]}
    saved = {}

    def save(model, metadata, output, tokenizer_dir, adapter):
        saved.update(metadata)
        output.write_bytes(b"checkpoint")

    adapter = SimpleNamespace(profile="latin-indic", tokenizer_sha256="t")
    monkeypatch.setattr(language_init, "restore_profile_checkpoint", lambda *args: (model, adapter, {}, LAYOUT))
    monkeypatch.setattr(language_init, "resolve_languages", lambda registry, languages: list(plan["languages"]))
    monkeypatch.setattr(language_init, "build_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(language_init, "init_language_state", lambda state, layout, enc, plan, **kwargs: (
        state, {s: {"rows": len(v["rows"])} for s, v in plan["scripts"].items()}, dict.fromkeys(plan["slots"], "auto_copy")))
    monkeypatch.setattr(language_init, "save_with_metadata", save)
    report = language_init.init_languages("in.nemo", tmp_path / "out.nemo", source_sha256="s")
    assert {k: v["calibrated"] for k, v in report["languages"].items()} == {
        "ta-IN": False, "mr-IN": "dormant_leftovers", "ur-PK": "dormant_leftovers"}
    assert saved["languages"] == report["languages"]
    require_calibrated(saved, ["mr-IN", "ur-PK"])
    with pytest.raises(ValueError, match="ta-IN"):
        require_calibrated(saved, ["ta-IN", "mr-IN"])
