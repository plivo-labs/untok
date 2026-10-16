"""Public commands: bundle inspection, native migration and Indic-language preparation."""
import json

import pytest

from untok.cli import main


def test_check_reports_native_ids(capsys):
    assert main(["check", "--bundle", "latin-indic"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["native_vocabulary_size"] == result["native_blank_id"] == 20360
    assert result["active_vocabulary_size"] == 10382


def test_migration_requires_hash_and_forwards_initialization_settings(monkeypatch, capsys):
    import untok.native_checkpoint as native
    calls = []
    monkeypatch.setattr(native, "migrate_native_checkpoint", lambda *a, **k: calls.append((a, k)) or {"ok": True})
    argv = ["migrate", "--source", "source.nemo", "--bundle", "latin-indic", "--output", "target.nemo"]
    with pytest.raises(SystemExit):
        main(argv)
    capsys.readouterr()
    assert main([*argv, "--source-sha256", "a" * 64]) == 0
    assert calls == [(("source.nemo", "latin-indic", "target.nemo"), {
        "expected_source_sha256": "a" * 64, "seed": 0, "max_new_mass_ratio": .05})]


@pytest.mark.parametrize("command", ["build", "clean", "package", "validate", "check-unigram"])
def test_retired_commands_are_not_exposed(command):
    with pytest.raises(SystemExit) as error:
        main([command])
    assert error.value.code == 2


def test_check_rejects_a_missing_bundle(tmp_path, capsys):
    assert main(["check", "--bundle", str(tmp_path / "missing")]) == 1
    assert capsys.readouterr().err


def test_init_languages_defaults_to_every_indic_language(monkeypatch, capsys):
    import untok.language_init as language_init
    calls = []
    monkeypatch.setattr(language_init, "init_languages", lambda *a, **k: calls.append((a, k)) or {"ok": True})
    assert main(["init-languages", "--source", "s.nemo", "--source-sha256", "a" * 64, "--output", "o.nemo"]) == 0
    assert calls == [(("s.nemo", "o.nemo"), {"source_sha256": "a" * 64, "languages": "all", "bundle": "latin-indic",
                                             "seed": 0, "force_prompt_slots": False})]


def test_calibrate_takes_one_manifest_per_language(monkeypatch, capsys):
    import untok.calibration as calibration
    calls = []
    monkeypatch.setattr(calibration, "calibrate", lambda *a, **k: calls.append(k["manifests"]) or {"ok": True})
    base = ["calibrate", "--source", "s.nemo", "--source-sha256", "a" * 64, "--output", "o.nemo"]
    assert main([*base, "--manifest", "ta-IN=ta.jsonl", "--manifest", "bn-IN=bn.jsonl"]) == 0
    assert calls == [{"ta-IN": "ta.jsonl", "bn-IN": "bn.jsonl"}]
    capsys.readouterr()
    assert main([*base, "--manifest", "ta.jsonl"]) == 1
    assert main([*base, "--manifest", "ta-IN=a", "--manifest", "ta-IN=b"]) == 1
