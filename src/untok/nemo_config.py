"""Prepare native NeMo model settings and a one-time fine-tuning configuration."""
from __future__ import annotations

import hashlib
from pathlib import Path


def apply_model_config(cfg, path):
    """Apply native constructor settings without changing token-to-weight identity."""
    if path is None:
        return cfg, None
    from omegaconf import OmegaConf

    path = Path(path).resolve()
    settings = OmegaConf.load(path)
    if not OmegaConf.is_dict(settings):
        raise ValueError("Model settings must be a YAML mapping of native model fields")
    original = OmegaConf.create({'model': cfg})
    updated = OmegaConf.merge(original, {'model': settings})
    for key in ('target', '_target_', 'tokenizer', 'labels', 'decoder.vocab_size',
                'joint.num_classes', 'joint.vocabulary'):
        if OmegaConf.select(updated.model, key) != OmegaConf.select(original.model, key):
            raise ValueError(f"Model settings cannot change migrated model.{key}")
    OmegaConf.resolve(updated)
    effective = OmegaConf.to_container(updated.model, resolve=True)
    return updated.model, {
        'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'effective': {key: effective[key] for key in settings},
    }


def native_training_config(checkpoint_config, checkpoint_path, template_path, overrides_path=None):
    """Use a native recipe with the checkpoint's exact model and prompt registry.

    Training datasets and optimizer come from the recipe, not old checkpoint
    paths or optimizer state. Subsequent execution belongs to NVIDIA's script.
    """
    from omegaconf import OmegaConf

    recipe = OmegaConf.load(template_path)
    if overrides_path is not None:
        overrides = OmegaConf.load(overrides_path)
        # A supplied optimizer is complete, as in NeMo's fine-tuning helper.
        if OmegaConf.select(overrides, 'model.optim') is not None:
            recipe.model.optim = overrides.model.optim
        recipe = OmegaConf.merge(recipe, overrides)
    native = OmegaConf.create({'model': checkpoint_config})
    model = OmegaConf.create(OmegaConf.to_container(native.model, resolve=True))
    for key in ('train_ds', 'validation_ds', 'test_ds', 'optim'):
        model.pop(key, None)
        if key in recipe.model:
            model[key] = recipe.model[key]
    # The checkpoint owns augmentation. The stock fine-tuning helper assigns a
    # separate spec_augment attribute; RNNT actually uses spec_augmentation.
    model.pop('spec_augment', None)
    model.tokenizer.update_tokenizer = False
    recipe.model = model
    recipe.init_from_nemo_model = str(Path(checkpoint_path).resolve())
    return recipe
