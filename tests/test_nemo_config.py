"""Native configuration composition preserves checkpoint identity and recipe controls."""
import hashlib

import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from untok.nemo_config import apply_model_config, native_training_config


def write_yaml(path, value):
    OmegaConf.save(OmegaConf.create(value), path)
    return path


@pytest.fixture
def checkpoint():
    return OmegaConf.create({
        'target': 'untok.native_runtime.NativeNemotronRNNTModel',
        'sample_rate': 16000,
        'tokenizer': {'type': 'untok_native_unigram', 'bundle_manifest_sha256': 'fixed'},
        'labels': ['a', 'ब'],
        'model_defaults': {'prompt_dictionary': {'en': 0, 'hi': 6, 'auto': 101},
                           'num_prompts': 128},
        'encoder': {'n_layers': 24, 'd_model': 1024, 'subsampling_factor': 8},
        'preprocessor': {'sample_rate': '${model.sample_rate}'},
        'decoder': {'vocab_size': 2},
        'joint': {'num_classes': 2, 'vocabulary': ['a', 'ब']},
        'optim': {'name': 'old_optimizer', 'old_option': True},
        'train_ds': {'manifest_filepath': '/old/machine/train.jsonl'},
        'spec_augment': {'freq_masks': 1},
    })


@pytest.fixture
def template(tmp_path):
    # A portable subset of the native template, including its interpolation paths.
    return write_yaml(tmp_path / 'native.yaml', {
        'name': 'native_training',
        'model': {
            'sample_rate': 16000,
            'model_defaults': {'prompt_dictionary': {'wrong_generic_prompt': 127}},
            'encoder': {'n_layers': 42, 'd_model': 1024, 'subsampling_factor': 8},
            'train_ds': {'manifest_filepath': '???', 'is_tarred': True,
                         'prompt_dictionary': '${model.model_defaults.prompt_dictionary}',
                         'num_prompts': '${model.model_defaults.num_prompts}',
                         'sample_rate': '${model.sample_rate}',
                         'subsampling_factor': '${model.encoder.subsampling_factor}',
                         'default_prompt_mode': 'unified'},
            'validation_ds': {'manifest_filepath': None},
            'test_ds': {'manifest_filepath': None},
            'optim': {'name': 'adamw', 'lr': 0.1, 'betas': [0.9, 0.98],
                      'sched': {'name': 'NoamAnnealing', 'd_model': '${model.encoder.d_model}'}},
            'spec_augment': {'freq_masks': 2},
        },
        'trainer': {'accelerator': 'gpu', 'limit_train_batches': 1000},
        'exp_manager': {'name': '${name}'},
    })


def test_model_settings_apply_before_construction_and_record_effective_values(checkpoint, tmp_path):
    before = OmegaConf.to_container(checkpoint, resolve=False)
    path = write_yaml(tmp_path / 'model.yaml', {
        'encoder': {'att_context_size': [[56, 0], [56, 1], [56, 3], [56, 6]],
                    'att_context_probs': [0.3, 0.3, 0.2, 0.2]},
        'freeze_updates': {'enabled': True, 'modules': {'encoder': -1, 'preprocessor': -1}},
        'preprocessor': {'sample_rate': '${model.sample_rate}'},
        'spec_augment': None,
    })
    cfg, receipt = apply_model_config(checkpoint, path)
    assert cfg.encoder.n_layers == 24
    assert cfg.encoder.att_context_probs == [0.3, 0.3, 0.2, 0.2]
    assert cfg.freeze_updates.modules.encoder == -1
    assert cfg.preprocessor.sample_rate == 16000
    assert cfg.spec_augment is None
    assert cfg.tokenizer == checkpoint.tokenizer
    assert receipt['effective']['encoder'] == OmegaConf.to_container(cfg.encoder)
    assert receipt['effective']['preprocessor'] == {'sample_rate': 16000}
    assert receipt['effective']['spec_augment'] is None
    assert receipt['source'] == str(path)
    assert receipt['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert OmegaConf.to_container(checkpoint, resolve=False) == before


@pytest.mark.parametrize('override,field', [
    ({'target': 'other.Model'}, 'target'),
    ({'_target_': 'other.Model'}, '_target_'),
    ({'tokenizer': {'type': 'bpe'}}, 'tokenizer'),
    ({'labels': ['different']}, 'labels'),
    ({'decoder': {'vocab_size': 3}}, 'decoder.vocab_size'),
    ({'joint': {'num_classes': 3}}, 'joint.num_classes'),
    ({'joint': {'vocabulary': ['different']}}, 'joint.vocabulary'),
])
def test_model_settings_reject_token_or_weight_identity_changes(checkpoint, tmp_path, override, field):
    path = write_yaml(tmp_path / 'model.yaml', override)
    with pytest.raises(ValueError, match=field):
        apply_model_config(checkpoint, path)


def test_model_settings_require_a_mapping(checkpoint, tmp_path):
    path = write_yaml(tmp_path / 'model.yaml', ['encoder', 'freeze_updates'])
    with pytest.raises(ValueError, match='YAML mapping'):
        apply_model_config(checkpoint, path)


def test_no_model_settings_leaves_native_configuration_unchanged(checkpoint):
    cfg, receipt = apply_model_config(checkpoint, None)
    assert cfg is checkpoint
    assert receipt is None


def test_training_configuration_uses_exact_registry_and_native_controls(checkpoint, template, tmp_path):
    before = OmegaConf.to_container(checkpoint, resolve=False)
    overrides = write_yaml(tmp_path / 'recipe.yaml', {
        'model': {'train_ds': {'is_tarred': False, 'default_prompt_mode': 'langID'}},
        'trainer': {'limit_train_batches': 1.0, 'val_check_interval': 1.0},
        'exp_manager': {'resume_from_checkpoint': 'resume.ckpt'},
    })
    cfg = native_training_config(checkpoint, tmp_path / 'adapted.nemo', template, overrides)
    assert cfg.model.encoder.n_layers == 24
    assert cfg.model.train_ds.prompt_dictionary == {'en': 0, 'hi': 6, 'auto': 101}
    assert cfg.model.train_ds.default_prompt_mode == 'langID'
    assert cfg.model.train_ds.num_prompts == 128
    assert cfg.model.train_ds.sample_rate == 16000
    assert cfg.model.train_ds.subsampling_factor == 8
    assert cfg.model.train_ds.is_tarred is False
    assert OmegaConf.is_missing(cfg.model.train_ds, 'manifest_filepath')
    assert cfg.model.test_ds.manifest_filepath is None
    assert cfg.model.tokenizer.update_tokenizer is False
    assert 'spec_augment' not in cfg.model
    assert 'old_option' not in cfg.model.optim
    assert cfg.model.optim.sched.d_model == 1024
    assert cfg.trainer.accelerator == 'gpu'
    assert cfg.trainer.limit_train_batches == 1.0
    assert type(cfg.trainer.limit_train_batches) is float
    assert cfg.exp_manager.resume_from_checkpoint == 'resume.ckpt'
    assert cfg.init_from_nemo_model == str(tmp_path / 'adapted.nemo')
    assert OmegaConf.to_container(checkpoint, resolve=False) == before
    saved = tmp_path / 'adapted.train.yaml'
    OmegaConf.save(cfg, saved)
    loaded = OmegaConf.load(saved)
    loaded.model.model_defaults.prompt_dictionary.hi = 12
    assert loaded.model.train_ds.prompt_dictionary.hi == 12


def test_supplied_optimizer_replaces_the_complete_native_optimizer(checkpoint, template, tmp_path):
    overrides = write_yaml(tmp_path / 'recipe.yaml', {
        'model': {'optim': {'name': 'sgd', 'lr': 0.01, 'momentum': 0.9}},
    })
    cfg = native_training_config(checkpoint, tmp_path / 'adapted.nemo', template, overrides)
    assert OmegaConf.to_container(cfg.model.optim) == {'name': 'sgd', 'lr': 0.01, 'momentum': 0.9}


def test_training_overrides_are_optional(checkpoint, template, tmp_path):
    cfg = native_training_config(checkpoint, tmp_path / 'adapted.nemo', template)
    assert cfg.model.optim.name == 'adamw'
    assert cfg.model.train_ds.default_prompt_mode == 'unified'
    assert cfg.model.train_ds.prompt_dictionary == checkpoint.model_defaults.prompt_dictionary
