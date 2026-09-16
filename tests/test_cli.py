"""The CLI must not turn incomplete or failed evidence into success."""
import json
from types import SimpleNamespace

import pytest

from untok.cli import main


@pytest.mark.parametrize("command", ["build", "build-unigram"])
def test_build_and_alias_use_native_builder(monkeypatch, capsys, command):
    import untok.unigram as native

    calls = []

    def build(*args):
        calls.append(args)
        return {"structural_passed": True, "checkpoint_validated": False, "asr_validated": False}

    monkeypatch.setattr(native, "build_native_tokenizer", build)
    assert main([command, "--base", "base.model", "--selection", "selected.json", "--output", "bundle"]) == 0
    assert calls == [("base.model", "selected.json", "bundle")]
    assert json.loads(capsys.readouterr().out)["structural_passed"]


@pytest.mark.parametrize("command", ["check", "check-unigram"])
def test_check_and_alias_verify_native_bundle(monkeypatch, capsys, command):
    import untok.bundles as native

    calls = []

    def adapter(bundle):
        calls.append(bundle)
        return SimpleNamespace(vocab_size=20, blank_id=20, id_map=SimpleNamespace(tokenizer_sha256="verified"))

    monkeypatch.setattr(native, "load_tokenizer_bundle", adapter)
    assert main([command, "--bundle", "candidate"]) == 0
    assert calls == ["candidate"]
    assert json.loads(capsys.readouterr().out) == {
        "structural_passed": True, "native_vocabulary_size": 20, "native_blank_id": 20,
        "tokenizer_sha256": "verified", "checkpoint_validated": False, "asr_validated": False,
    }


@pytest.mark.parametrize("command", ["validate", "validate-unigram"])
@pytest.mark.parametrize("state,expected", [("passed", 0), ("incomplete", 2), ("failed", 2)])
def test_native_validation_dispatch_and_evidence_gate(monkeypatch, tmp_path, capsys, command, state, expected):
    import untok.unigram_validation as native

    calls = []
    report = {"passed": state == "passed", "status": state, "structural_passed": state != "failed",
              "corpus_status": state}

    def validate(*args, **kwargs):
        calls.append((args, kwargs))
        return report

    monkeypatch.setattr(native, "validate_native_tokenizer", validate)
    output = tmp_path / "report.json"
    assert main([command, "--bundle", "candidate", "--policy", "policy.json", "--corpora", "corpora.json",
                 "--phase", "reserve", "--selection-receipt", "receipt.json", "--output", str(output)]) == expected
    assert calls == [(("candidate", "policy.json", "corpora.json"),
                     {"phase": "reserve", "selection_receipt_path": "receipt.json", "max_examples": 0})]
    assert json.loads(output.read_text()) == report
    assert not json.loads(capsys.readouterr().out)["checkpoint_validated"]


@pytest.mark.parametrize("command", ["build", "build-unigram", "check", "check-unigram", "validate", "validate-unigram"])
def test_native_commands_require_explicit_inputs(command):
    with pytest.raises(SystemExit) as error:
        main([command])
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["build", "build-unigram"])
def test_native_build_failure_returns_error(monkeypatch, capsys, command):
    import untok.unigram as native

    def reject(*args):
        raise ValueError("Selection was not fitted against this native base hash")

    monkeypatch.setattr(native, "build_native_tokenizer", reject)
    assert main([command, "--base", "wrong.model", "--selection", "selection.json", "--output", "bundle"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "native base hash" in captured.err


def test_invalid_native_bundle_returns_error(monkeypatch, capsys):
    import untok.bundles as native

    def reject(bundle):
        raise ValueError("Native bundle file hash mismatch")

    monkeypatch.setattr(native, "load_tokenizer_bundle", reject)
    assert main(["check", "--bundle", "changed-bundle"]) == 1
    assert "hash mismatch" in capsys.readouterr().err


def test_package_dispatches_all_four_profiles(monkeypatch, capsys):
    import untok.bundles as bundles

    calls = []

    def package(*args, **kwargs):
        calls.append((args, kwargs))
        return {"packaged": True}

    monkeypatch.setattr(bundles, "package_tokenizer_bundles", package)
    assert main(["package", "--bundle", "candidate", "--output", "dist"]) == 0
    assert calls == [(("candidate", "dist"), {"profiles": ("original", "latin", "latin-indic", "full")})]
    assert json.loads(capsys.readouterr().out)["packaged"]


def test_clean_command_builds_versioned_profiles(monkeypatch, capsys):
    import untok.clean as clean

    calls = []
    def build(*args):
        calls.append(args)
        return {"full": {"tokenizer_version": 4}}

    monkeypatch.setattr(clean, "build_clean_bundles", build)
    assert main(["clean", "--bundle", "source", "--output", "cleaned"]) == 0
    assert calls == [("source", "cleaned")]
    assert json.loads(capsys.readouterr().out)["full"]["tokenizer_version"] == 4


def test_native_migration_requires_source_pin_and_dispatches_native(monkeypatch, capsys):
    import untok.native_checkpoint as native

    calls = []

    def migrate(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "migrated_weights_verified"}

    monkeypatch.setattr(native, "migrate_native_checkpoint", migrate)
    flags = ["migrate", "--source", "source.nemo", "--bundle", "native", "--output", "new.nemo"]
    with pytest.raises(SystemExit) as error:
        main(flags)
    assert error.value.code == 2
    assert not calls
    assert main(flags + ["--source-sha256", "a" * 64]) == 0
    assert calls == [(("source.nemo", "native", "new.nemo"), {
        "expected_source_sha256": "a" * 64, "seed": 0,
        "max_new_mass_ratio": 0.05,
        "model_config": None, "training_template": None, "training_overrides": None,
    })]


def test_native_migration_forwards_mass_ratio_and_native_configuration(monkeypatch, capsys):
    import untok.native_checkpoint as native

    calls = []

    def migrate(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "migrated_weights_verified"}

    monkeypatch.setattr(native, "migrate_native_checkpoint", migrate)
    assert main([
        "migrate", "--source", "source.nemo", "--bundle", "native", "--output", "new.nemo",
        "--source-sha256", "a" * 64, "--seed", "17",
        "--max-new-mass-ratio", "0.03", "--model-config", "model.yaml",
        "--training-template", "nvidia.yaml", "--training-overrides", "recipe.yaml",
    ]) == 0
    assert calls == [(("source.nemo", "native", "new.nemo"), {
        "expected_source_sha256": "a" * 64, "seed": 17,
        "max_new_mass_ratio": 0.03,
        "model_config": "model.yaml", "training_template": "nvidia.yaml",
        "training_overrides": "recipe.yaml",
    })]
    assert json.loads(capsys.readouterr().out)["status"] == "migrated_weights_verified"


@pytest.mark.parametrize("initialization", ["blank", "text-donor"])
def test_migration_rejects_removed_initialization_option(monkeypatch, capsys, initialization):
    import untok.native_checkpoint as native

    monkeypatch.setattr(native, "migrate_native_checkpoint", lambda *args, **kwargs: pytest.fail("Invalid flags must not migrate"))
    with pytest.raises(SystemExit) as error:
        main(["migrate", "--source", "source.nemo", "--bundle", "native", "--output", "new.nemo",
              "--source-sha256", "a" * 64, "--initialization", initialization])
    assert error.value.code == 2
    assert "unrecognized arguments: --initialization" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["legacy-bpe", "fetch", "fetch-corpora", "scope-corpora", "preflight",
                                    "id-map", "prompts", "verify-checkpoint", "infer", "evaluate"])
def test_removed_legacy_commands_are_rejected(command):
    with pytest.raises(SystemExit) as error:
        main([command])
    assert error.value.code == 2


def test_help_explains_native_migration_and_text_donor_default(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "native SentencePiece Unigram" in text
    assert "Checkpoint migration requires a compatible NVIDIA NeMo runtime" in text
    assert "legacy-bpe" not in text
    assert "evaluate" not in text
    assert "build-unigram" in text
    with pytest.raises(SystemExit) as error:
        main(["migrate", "--help"])
    assert error.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "--initialization" not in text
    assert "Initial text-donor added-output mass bound (default: 0.05)" in text
