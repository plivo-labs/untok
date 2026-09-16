"""Hash-pinned native Unigram RNNT migration, preserving every original model row."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .checkpoint import RNNTLayout, _sha256, _torch, inspect_nemo_layout
from .native_donors import initialize_text_donor_rows
from .native_runtime import get_native_nemo_model_class, native_bundle_config, verify_native_output_mask


def retained_row_pairs(old_layout, new_layout, source_to_target):
    """Keep every original text row at its ID and relocate only RNNT blank."""
    expected = (*range(old_layout.blank_id), new_layout.blank_id)
    if (old_layout.row_keys != new_layout.row_keys
            or old_layout.blank_id != old_layout.output_size - 1
            or new_layout.blank_id != new_layout.output_size - 1
            or old_layout.blank_id < 1 or new_layout.blank_id < old_layout.blank_id
            or any(type(index) is not int for index in source_to_target)
            or tuple(source_to_target) != expected):
        raise ValueError("Migration must retain every original text ID and relocate only blank")
    return tuple(range(old_layout.output_size)), expected


def _validate_state_shapes(source, target, old_layout, new_layout):
    torch = _torch()
    if set(source) != set(target):
        raise ValueError("State keys differ across native migration")
    if old_layout.row_keys != new_layout.row_keys or not set(old_layout.row_keys) <= set(source):
        raise ValueError("Native vocabulary parameter layout changed or is incomplete")
    for key, old in source.items():
        new = target[key]
        if not isinstance(old, torch.Tensor) or not isinstance(new, torch.Tensor):
            raise ValueError(f"Unsupported non-tensor state: {key}")
        if old.dtype != new.dtype:
            raise ValueError(f"Unexpected dtype change: {key}")
        if key in old_layout.row_keys:
            if (old.ndim < 1 or new.ndim < 1 or old.shape[0] != old_layout.output_size
                    or new.shape[0] != new_layout.output_size or old.shape[1:] != new.shape[1:]):
                raise ValueError(f"Invalid vocabulary tensor shape: {key}")
        elif old.shape != new.shape:
            raise ValueError(f"Unapproved non-vocabulary shape change: {key}")


def transfer_native_state_dict(source, initialized_target, old_layout, new_layout, source_to_target):
    """Copy every source tensor and relocate its vocabulary blank row."""
    torch = _torch()
    old_ids, new_ids = retained_row_pairs(old_layout, new_layout, source_to_target)
    _validate_state_shapes(source, initialized_target, old_layout, new_layout)
    result = {}
    for key, old in source.items():
        target = initialized_target[key]
        if key in old_layout.row_keys:
            selected = old.index_select(0, torch.tensor(old_ids, device=old.device))
            copied = target.detach().clone()
            copied.index_copy_(0, torch.tensor(new_ids, device=copied.device), selected.to(copied.device))
            result[key] = copied
        else:
            result[key] = old.detach().to(target.device).clone()
    return result


def verify_native_state_transfer(source, migrated, old_layout, new_layout, source_to_target):
    """Independently compare every source value, including the relocated blank."""
    torch = _torch()
    old_ids, new_ids = retained_row_pairs(old_layout, new_layout, source_to_target)
    _validate_state_shapes(source, migrated, old_layout, new_layout)
    failures, preserved, omitted = [], 0, 0
    for key, source_tensor in source.items():
        expected, actual = source_tensor, migrated[key]
        if key in old_layout.row_keys:
            expected = expected.index_select(0, torch.tensor(old_ids, device=expected.device))
            actual = actual.index_select(0, torch.tensor(new_ids, device=actual.device))
            omitted += source_tensor.numel() - expected.numel()
        if not torch.equal(expected.cpu(), actual.cpu()):
            failures.append(key)
        else:
            preserved += expected.numel()
    if failures:
        raise ValueError("Retained learned state changed: " + ", ".join(failures))
    return {"passed": True, "comparison": "exact_tensor_equality", "tensors_checked": len(source),
            "learned_values_preserved": preserved, "learned_values_omitted": omitted,
            "retained_source_rows_including_blank": len(old_ids),
            "removed_source_text_rows": old_layout.output_size - len(old_ids),
            "all_source_values_preserved": omitted == 0}


def _source_native_bytes(model):
    tokenizer = model.tokenizer
    for candidate in (getattr(tokenizer, "backend", None), getattr(tokenizer, "tokenizer", None)):
        serialize = getattr(candidate, "serialized_model_proto", None)
        if callable(serialize):
            data = serialize()
            if isinstance(data, bytes) and data:
                return data
    raise ValueError("Source tokenizer must expose its actual native SentencePiece model")


def validate_source_native_tokenizer(model, base_bytes):
    import sentencepiece as spm

    actual = _source_native_bytes(model)
    if actual != base_bytes:
        raise ValueError("Restored source tokenizer differs from the bundle's pinned native base")
    processor = spm.SentencePieceProcessor(model_proto=base_bytes)
    if int(model.tokenizer.vocab_size) != processor.get_piece_size():
        raise ValueError("Source tokenizer vocabulary size disagrees with its native model")
    for index in range(processor.get_piece_size()):
        if model.tokenizer.ids_to_tokens([index]) != [processor.id_to_piece(index)]:
            raise ValueError(f"Source native token ID changed: {index}")
    probes = ["", "hello", " hello  world ", "नमस्ते", "അവന്‍", "a\u200cb", "Ａ ﬁ", "<unk>"]
    for text in probes:
        if model.tokenizer.text_to_ids(text) != processor.encode(text, out_type=int):
            raise ValueError("Source wrapper encoding disagrees with native SentencePiece")
    return {"native_tokenizer_sha256": hashlib.sha256(actual).hexdigest(),
            "piece_ids_checked": processor.get_piece_size(), "encoding_probes_checked": len(probes)}


def migrate_native_checkpoint(source, bundle, output, *, expected_source_sha256, seed=0,
                              max_new_mass_ratio=0.05,
                              model_config=None, training_template=None, training_overrides=None):
    """Migrate the original NVIDIA checkpoint to one of the four current profiles.

    Every source weight survives exactly. New classes and filtered outputs can
    change recognition; tensor equality does not establish speech accuracy.
    """
    from importlib import resources
    from .bundles import PROFILES, load_tokenizer_bundle

    if training_overrides is not None and training_template is None:
        raise ValueError("Training overrides require a training template")
    torch = _torch()
    if isinstance(bundle, str) and bundle in PROFILES:
        bundle = resources.files("untok").joinpath("data", bundle)
    source, bundle, output = Path(source), Path(bundle).resolve(), Path(output)
    if not source.is_file() or source.suffix != ".nemo":
        raise ValueError("Provide a complete local source .nemo checkpoint")
    if (not isinstance(expected_source_sha256, str) or len(expected_source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_source_sha256)
            or _sha256(source) != expected_source_sha256):
        raise ValueError("Source checkpoint SHA256 differs from its explicit pin")
    report_path = output.with_suffix(".migration.json")
    training_path = output.with_suffix(".train.yaml") if training_template is not None else None
    if (output.suffix != ".nemo" or output.exists() or output.resolve() == source.resolve()
            or report_path.exists() or training_path is not None and training_path.exists()):
        raise ValueError("Checkpoint and migration report require new output paths")
    adapter = load_tokenizer_bundle(bundle)
    tokenizer_cfg = native_bundle_config(bundle)
    base_to_target = tuple(adapter.source_native_to_target_native)
    requires_retokenized_labels = adapter.profile != "original"
    target_sha = hashlib.sha256(adapter.model_bytes).hexdigest()
    base_sha = hashlib.sha256(adapter.base_model_bytes).hexdigest()
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
    from omegaconf import OmegaConf, open_dict

    native_class = get_native_nemo_model_class()
    original = ASRModel.restore_from(str(source), map_location="cpu")
    if type(original) not in {EncDecRNNTBPEModelWithPrompt, native_class}:
        raise ValueError("Unsupported source model class for native migration")
    source_tokenizer_check = validate_source_native_tokenizer(original, adapter.base_model_bytes)
    source_inventory, source_to_target = "original_native_base", base_to_target
    old_layout = inspect_nemo_layout(original)
    if old_layout.output_size != len(source_to_target):
        raise ValueError("Source acoustic vocabulary disagrees with the selected pinned tokenizer inventory")
    cfg = OmegaConf.create(OmegaConf.to_container(original.cfg, resolve=True))
    from .prompts import prompt_registry
    registry = prompt_registry(cfg.model_defaults, adapter.profile)
    with open_dict(cfg):
        cfg.tokenizer = tokenizer_cfg
        cfg.target = "untok.native_runtime.NativeNemotronRNNTModel"
        cfg.model_defaults.prompt_dictionary = registry["prompt_dictionary"]
        cfg.untok_native_migration = {"source_checkpoint_sha256": expected_source_sha256,
                                      "source_inventory": source_inventory,
                                      "source_tokenizer_sha256": source_tokenizer_check["native_tokenizer_sha256"],
                                      "base_tokenizer_sha256": base_sha,
                                      "tokenizer_sha256": target_sha,
                                      "requires_retokenized_training_labels": requires_retokenized_labels,
                                      "prompt_registry": registry}
        for split in ("train_ds", "validation_ds", "test_ds"):
            cfg[split] = None
    settings = None
    if model_config is not None:
        from .nemo_config import apply_model_config
        cfg, settings = apply_model_config(cfg, model_config)

    def verify_settings(model):
        if settings is not None:
            actual = OmegaConf.to_container(OmegaConf.create({"model": model.cfg}), resolve=True)["model"]
            if any(actual.get(key) != value for key, value in settings["effective"].items()):
                raise ValueError("Native model settings changed during construction or checkpoint reload")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".untok-native-migration-", dir=output.parent) as staging:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            expanded = native_class(cfg=cfg, trainer=None)
        verify_settings(expanded)
        new_layout = inspect_nemo_layout(expanded)
        retained_row_pairs(old_layout, new_layout, source_to_target)
        if expanded.tokenizer.model_bytes != adapter.model_bytes or new_layout.blank_id != adapter.blank_id:
            raise ValueError("Constructed model disagrees with the target native tokenizer")
        verify_native_output_mask(expanded)
        transferred = transfer_native_state_dict(
            original.state_dict(), expanded.state_dict(), old_layout, new_layout, source_to_target)
        transferred, policy = initialize_text_donor_rows(
            transferred, new_layout, adapter, source_to_target, max_new_mass_ratio=max_new_mass_ratio)
        added = policy["new_model_rows"]
        expected_added = {key: transferred[key][added].detach().clone() for key in new_layout.row_keys}
        expanded.load_state_dict(transferred, strict=True)
        del transferred
        expanded.eval()
        before = verify_native_state_transfer(original.state_dict(), expanded.state_dict(), old_layout, new_layout, source_to_target)
        output_mask_before = verify_native_output_mask(expanded)
        if any(not torch.equal(expanded.state_dict()[key][added], value) for key, value in expected_added.items()):
            raise ValueError("Added native row initialization changed during transfer")
        policy["verified_before_save"] = True
        staged = Path(staging) / output.name
        expanded.save_to(str(staged))
        del expanded
        restored = native_class.restore_from(str(staged), map_location="cpu")
        verify_settings(restored)
        if inspect_nemo_layout(restored) != new_layout:
            raise ValueError("Native acoustic layout changed after checkpoint reload")
        output_mask_after = verify_native_output_mask(restored)
        if output_mask_after != output_mask_before:
            raise ValueError("Native inactive-output mask changed after checkpoint reload")
        after = verify_native_state_transfer(original.state_dict(), restored.state_dict(), old_layout, new_layout, source_to_target)
        if (restored.tokenizer.model_bytes != adapter.model_bytes
                or restored.tokenizer.base_model_bytes != adapter.base_model_bytes
                or tuple(restored.tokenizer.source_native_to_target_native) != base_to_target
                or restored.native_bundle_manifest_sha256 != tokenizer_cfg["bundle_manifest_sha256"]):
            raise ValueError("Native tokenizer artifacts or row mapping changed after checkpoint reload")
        if dict(restored.cfg.model_defaults.prompt_dictionary) != registry["prompt_dictionary"]:
            raise ValueError("Prompt assignments changed after native checkpoint reload")
        saved_metadata = OmegaConf.to_container(restored.cfg.untok_native_migration, resolve=True)
        if saved_metadata["prompt_registry"] != registry:
            raise ValueError("Prompt registry provenance changed after checkpoint reload")
        if (saved_metadata.get("source_inventory") != source_inventory
                or saved_metadata.get("source_tokenizer_sha256") != source_tokenizer_check["native_tokenizer_sha256"]
                or saved_metadata.get("requires_retokenized_training_labels") != requires_retokenized_labels):
            raise ValueError("Source tokenizer inventory or label policy changed after checkpoint reload")
        if any(not torch.equal(restored.state_dict()[key][added], value) for key, value in expected_added.items()):
            raise ValueError("Added native rows changed after checkpoint reload")
        policy["verified_after_reload"] = True
        if _sha256(source) != expected_source_sha256 or native_bundle_config(bundle) != tokenizer_cfg:
            raise ValueError("Pinned migration inputs changed during execution")
        if settings is not None and _sha256(settings["source"]) != settings["sha256"]:
            raise ValueError("Model settings changed during migration")
        training = None
        if training_path is not None:
            from .nemo_config import native_training_config
            training_cfg = native_training_config(restored.cfg, output.resolve(), training_template, training_overrides)
            staged_training = Path(staging) / training_path.name
            OmegaConf.save(training_cfg, staged_training)
            training = {"path": str(training_path.resolve()), "sha256": _sha256(staged_training)}
        report = {"schema_version": 1, "status": "native_migrated_weights_verified",
                  "source_checkpoint_sha256": expected_source_sha256, "checkpoint_sha256": _sha256(staged),
                  "source_inventory": source_inventory,
                  "source_tokenizer_sha256": source_tokenizer_check["native_tokenizer_sha256"],
                  "tokenizer_sha256": target_sha, "base_tokenizer_sha256": base_sha,
                  "bundle_manifest_sha256": tokenizer_cfg["bundle_manifest_sha256"],
                  "seed": seed, "old_layout": asdict(old_layout), "new_layout": asdict(new_layout),
                  "old_model_to_new_model": list(source_to_target),
                  "source_remapping": "source_native_to_target_native",
                  "requires_retokenized_training_labels": requires_retokenized_labels,
                  "source_tokenizer_check": source_tokenizer_check,
                  "before_save": before, "after_reload": after,
                  "initialization": "text-donor", "new_row_initialization": policy, "prompt_registry": registry,
                  "model_settings": settings, "training_config": training,
                  "inactive_output_mask_before_save": output_mask_before,
                  "inactive_output_mask_after_reload": output_mask_after,
                  "native_tokenizer_verified_before_save_and_after_reload": True,
                  "asr_accuracy_evaluated": False, "training_forward_evaluated": False,
                  "logit_parity_evaluated": False,
                  "remaining_gates": (["retokenize_training_labels"] if requires_retokenized_labels else [])
                                     + ["training_forward_backward", "retained_logit_and_audio_controls",
                                        "unmasked_audio_regression"]
                                     + ([] if adapter.profile in {"original", "latin"}
                                        else ["new_language_fine_tuning_and_evaluation"])}
        staged_report = Path(staging) / report_path.name
        staged_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        publications = [(staged, output), (staged_report, report_path)]
        if training_path is not None:
            publications.append((staged_training, training_path))
        published = []
        try:
            for temporary, destination in publications:
                os.link(temporary, destination)
                published.append((temporary, destination))
        except OSError:
            # Remove only links created by this call; never overwrite another artifact.
            for temporary, destination in reversed(published):
                if destination.exists() and os.path.samestat(destination.stat(), temporary.stat()):
                    destination.unlink()
            raise
    return report
