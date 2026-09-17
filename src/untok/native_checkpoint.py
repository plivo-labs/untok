"""Hash-pinned native Unigram RNNT migration, preserving every original model row."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .checkpoint import _sha256, _torch, inspect_nemo_layout
from .native_donors import initialize_text_donor_rows


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


def migrate_native_checkpoint(source, bundle, output, *, expected_source_sha256,
                              seed=0, max_new_mass_ratio=0.05):
    """Prepare native Nemotron weights and ordinary SentencePiece files once.

    Training configuration belongs to NVIDIA's config-first training entry point.
    This function neither generates training YAML nor changes training settings.
    """
    if not 0 < max_new_mass_ratio < 1:
        raise ValueError("New-output mass ratio must be finite and between zero and one")
    from .bundles import load_tokenizer_bundle
    from .nemo import register
    from .prompts import prompt_registry
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
    from omegaconf import OmegaConf, open_dict

    source, output = Path(source), Path(output).resolve()
    if source.suffix != ".nemo" or _sha256(source) != expected_source_sha256:
        raise ValueError("Provide the original .nemo checkpoint and its matching SHA256")
    tokenizer_dir, report_path = output.with_suffix(".tokenizer"), output.with_suffix(".migration.json")
    if output.suffix != ".nemo" or any(p.exists() for p in (output, tokenizer_dir, report_path)):
        raise ValueError("Checkpoint, tokenizer directory and report require new output paths")
    adapter = load_tokenizer_bundle(bundle)
    register()
    original = ASRModel.restore_from(str(source), map_location="cpu")
    if type(original) is not EncDecRNNTBPEModelWithPrompt:
        raise ValueError("Source must be NVIDIA's native prompted RNNT checkpoint")
    source_check = validate_source_native_tokenizer(original, adapter.base_model_bytes)
    old_layout = inspect_nemo_layout(original)
    mapping = adapter.source_native_to_target_native
    cfg = OmegaConf.create(OmegaConf.to_container(original.cfg, resolve=True))
    registry = prompt_registry(cfg.model_defaults, adapter.profile)
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer.model").write_bytes(adapter.model_bytes)
    # NeMo gets IDs from the model; this required sidecar lists physical slots.
    pieces = [adapter.backend.id_to_piece(i) for i in range(adapter.vocab_size)]
    (tokenizer_dir / "vocab.txt").write_text("\n".join(pieces) + "\n", encoding="utf-8")
    with open_dict(cfg):
        cfg.tokenizer = {"dir": str(tokenizer_dir), "type": "bpe"}
        cfg.model_defaults.prompt_dictionary = registry["prompt_dictionary"]
        if adapter.inactive_native_ids:
            cfg.joint._target_ = "untok.nemo.MaskedRNNTJoint"
            cfg.joint.mask_unused_tokens = True
        for split in ("train_ds", "validation_ds", "test_ds"):
            cfg[split] = None
    torch = _torch()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        expanded = EncDecRNNTBPEModelWithPrompt(cfg=cfg, trainer=None)
    new_layout = inspect_nemo_layout(expanded)
    if new_layout.blank_id != adapter.blank_id or _source_native_bytes(expanded) != adapter.model_bytes:
        raise ValueError("Native model disagrees with the selected tokenizer or blank")
    state = transfer_native_state_dict(original.state_dict(), expanded.state_dict(), old_layout, new_layout, mapping)
    state, policy = initialize_text_donor_rows(state, new_layout, adapter, mapping,
                                              max_new_mass_ratio=max_new_mass_ratio)
    expanded.load_state_dict(state, strict=True)
    preservation = verify_native_state_transfer(original.state_dict(), expanded.state_dict(), old_layout, new_layout, mapping)
    del state, original
    expanded.eval()
    expanded.save_to(str(output))
    # Save/restore and acoustic checks are integration tests, not a second model
    # implementation hidden inside the preparation command.
    report = {"source_checkpoint_sha256": expected_source_sha256,
              "checkpoint_sha256": _sha256(output), "profile": adapter.profile,
              "tokenizer_sha256": adapter.tokenizer_sha256, "source_tokenizer_check": source_check,
              "tokenizer_directory": str(tokenizer_dir), "old_blank_id": old_layout.blank_id,
              "blank_id": new_layout.blank_id, "inactive_outputs": len(adapter.inactive_native_ids),
              "retained_state": preservation, "new_row_initialization": policy,
              "prompt_registry": registry, "asr_accuracy_evaluated": False}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
