"""Native artifact persistence and exact-class registration without importing NeMo."""
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.bundles import load_tokenizer_bundle, package_tokenizer_bundles
from untok.native_checkpoint import migrate_native_checkpoint
from untok.native_runtime import (
    _register_native_target, _setup_native_tokenizer, install_native_output_mask,
    native_bundle_config, verify_native_output_mask,
)
from untok.unigram import build_native_tokenizer


def test_native_file_inference_uses_verified_tensor_prompt_path(tmp_path, monkeypatch):
    import untok.inference as inference
    from untok.native_runtime import transcribe_native_file

    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"audio provenance fixture")
    calls = []
    class Model:
        native_tokenizer_sha256 = "verified-native"
        eval_calls = 0
        def eval(self):
            self.eval_calls += 1
            return self
    model = Model()
    def observed(*args):
        calls.append(args)
        return (["actual-return"], {"verified_kernel_calls": 1})
    monkeypatch.setattr(inference, "_transcribe_with_verified_prompt", observed)
    assert transcribe_native_file(model, audio, target_lang="hi-IN") == (["actual-return"], {"verified_kernel_calls": 1})
    assert calls == [(model, audio, hashlib.sha256(audio.read_bytes()).hexdigest(), "hi-IN")]
    assert model.eval_calls == 2
    def fail(*args):
        raise ValueError("transcription failed")
    monkeypatch.setattr(inference, "_transcribe_with_verified_prompt", fail)
    with pytest.raises(ValueError, match="transcription failed"):
        transcribe_native_file(model, audio, target_lang="hi-IN")
    assert model.eval_calls == 4
    with pytest.raises(ValueError, match="native untok checkpoint"):
        transcribe_native_file(SimpleNamespace(), audio, target_lang="hi-IN")


def masked_toy_model():
    from test_native_checkpoint import toy_model

    model = toy_model(5)
    model.tokenizer = SimpleNamespace(inactive_native_ids=(1, 3), blank_id=5)
    return model


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype_name", ["float32", "float64", "float16", "bfloat16"])
def test_native_mask_preserves_parameters_and_excludes_logits_before_softmax(training, dtype_name):
    torch = pytest.importorskip("torch")
    model = masked_toy_model().to(dtype=getattr(torch, dtype_name)).train(training)
    head = model.joint.joint_net[-1]
    parameters = {name: value for name, value in model.named_parameters()}
    with torch.no_grad():
        # Inactive rows would win every argmax if the mask were omitted.
        head.bias[[1, 3]] = 100
    state = {name: value.clone() for name, value in model.state_dict().items()}
    inputs = torch.randn(2, 3, head.in_features, dtype=head.weight.dtype, requires_grad=True)
    unmasked = head(inputs)
    assert set(unmasked.argmax(-1).flatten().tolist()) <= {1, 3}
    report = install_native_output_mask(model)
    masked = model.joint.joint_net[-1](inputs)
    active = [0, 2, 4, 5]
    assert report["passed"] and report["active_text_rows"] == 3
    assert torch.isneginf(masked[..., [1, 3]]).all()
    assert torch.equal(masked[..., active], unmasked[..., active])
    assert set(masked.argmax(-1).flatten().tolist()) <= set(active)
    probabilities = masked.softmax(-1)
    assert torch.equal(probabilities[..., [1, 3]], torch.zeros_like(probabilities[..., [1, 3]]))
    # Removing zero terms changes reduction grouping by roundoff only.
    assert torch.allclose(probabilities[..., active], unmasked[..., active].softmax(-1),
                          atol=2 * torch.finfo(head.weight.dtype).eps, rtol=0)
    loss = -masked.log_softmax(-1)[..., 2].float().mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(inputs.grad).all()
    for value in (head.weight, head.bias):
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[[1, 3]]) == 0
    assert model.joint.joint_net[-1].training == training
    assert set(model.state_dict()) == set(state)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(value is dict(model.named_parameters())[name] for name, value in parameters.items())
    assert verify_native_output_mask(model) == report


def test_native_mask_survives_trace_and_state_reload_and_detects_tampering(tmp_path):
    torch = pytest.importorskip("torch")
    model = masked_toy_model().eval()
    install_native_output_mask(model)
    head = model.joint.joint_net[-1]
    probe = torch.ones(2, head.in_features)
    traced = torch.jit.trace(head, probe)
    path = tmp_path / "traced-head.pt"
    torch.jit.save(traced, str(path))
    reloaded = torch.jit.load(str(path))
    assert torch.equal(reloaded(probe), head(probe))
    assert torch.isneginf(reloaded(probe)[..., [1, 3]]).all()
    state = tmp_path / "weights.pt"
    torch.save(model.state_dict(), state)
    restored = masked_toy_model().eval()
    install_native_output_mask(restored)
    restored.load_state_dict(torch.load(state, weights_only=True))
    assert verify_native_output_mask(restored) == verify_native_output_mask(model)
    assert torch.equal(restored.joint.joint_net[-1](probe), head(probe))
    restored.joint.joint_net[-1].inactive_output_mask[1] = False
    with pytest.raises(ValueError, match="mask differs"):
        verify_native_output_mask(restored)
    restored.joint.joint_net[-1] = torch.nn.Linear(head.in_features, head.out_features)
    with pytest.raises(ValueError, match="mask is missing"):
        verify_native_output_mask(restored)


@pytest.mark.parametrize("inactive", [(5,), (-1,), (True,), (1.0,), (1, 1)])
def test_native_mask_rejects_blank_invalid_or_duplicate_inactive_rows(inactive):
    model = masked_toy_model()
    model.tokenizer.inactive_native_ids = inactive
    with pytest.raises(ValueError, match="unique text rows"):
        install_native_output_mask(model)


def test_native_mask_rejects_sampled_joint_bypass():
    model = masked_toy_model()
    model.joint.sampled_joint = lambda *args: None
    with pytest.raises(ValueError, match="standard RNNT joint"):
        install_native_output_mask(model)


def test_native_mask_cuda_graph_when_cuda_is_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; trace coverage runs on CPU")
    model = masked_toy_model().cuda().eval()
    install_native_output_mask(model)
    head = model.joint.joint_net[-1]
    inputs = torch.ones(2, head.in_features, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(3):
            head(inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        result = head(inputs)
    inputs.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isneginf(result[..., [1, 3]]).all()
    assert torch.equal(result, head(inputs))


@pytest.mark.parametrize("backend", ["pytorch", "warprnnt_numba"])
@pytest.mark.parametrize("fused", [False, True])
def test_real_nemo_joint_and_loss_preserve_finite_training_with_inactive_rows(backend, fused):
    """Optional integration gate, exercised with the pinned NeMo environment."""
    torch = pytest.importorskip("torch")
    rnnt = pytest.importorskip("nemo.collections.asr.modules.rnnt")
    from nemo.collections.asr.losses.rnnt import RNNTLoss

    joint = rnnt.RNNTJoint(
        jointnet={"encoder_hidden": 4, "pred_hidden": 4, "joint_hidden": 4, "activation": "relu"},
        num_classes=5, log_softmax=None, fuse_loss_wer=fused, fused_batch_size=1 if fused else None,
    )
    model = SimpleNamespace(joint=joint, tokenizer=SimpleNamespace(inactive_native_ids=(1, 3), blank_id=5))
    original = {name: value.detach().clone() for name, value in joint.state_dict().items()}
    install_native_output_mask(model)
    assert set(original) == set(joint.state_dict())
    assert all(torch.equal(value, joint.state_dict()[name]) for name, value in original.items())
    criterion = RNNTLoss(num_classes=5, loss_name=backend, reduction="sum")
    encoder = torch.randn(2, 4, 3, requires_grad=True)
    decoder = torch.randn(2, 4, 3, requires_grad=True)
    labels = torch.tensor([[0, 2], [2, 4]], dtype=torch.int64)
    input_lengths, target_lengths = torch.tensor([3, 3]), torch.tensor([2, 2])
    # This projected entry point is also used by greedy, beam and CUDA graphs.
    projected = joint.joint_after_projection(
        joint.project_encoder(encoder.transpose(1, 2)), joint.project_prednet(decoder.transpose(1, 2)),
    )
    assert torch.isneginf(projected[..., [1, 3]]).all()
    if fused:
        joint.set_fuse_loss_wer(True, loss=criterion, metric=torch.nn.Identity())
        loss, *_ = joint(encoder_outputs=encoder, decoder_outputs=decoder, encoder_lengths=input_lengths,
                         transcripts=labels, transcript_lengths=target_lengths, compute_wer=False)
    else:
        logits = joint(encoder_outputs=encoder, decoder_outputs=decoder)
        assert torch.isneginf(logits[..., [1, 3]]).all()
        loss = criterion(log_probs=logits, targets=labels,
                         input_lengths=input_lengths, target_lengths=target_lengths)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(encoder.grad).all() and torch.isfinite(decoder.grad).all()
    for parameter in (joint.joint_net[-1].weight, joint.joint_net[-1].bias):
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad[[1, 3]]) == 0
    assert verify_native_output_mask(model)["passed"]


def test_real_nemo_serialization_forwards_trainer_to_native_constructor(monkeypatch):
    """NeMo's real constructor dispatch must not silently drop the Trainer."""
    torch = pytest.importorskip("torch")
    common = pytest.importorskip("nemo.core.classes.common")
    from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
    from omegaconf import OmegaConf
    from untok.native_runtime import get_native_nemo_model_class

    seen = []

    def parent_constructor(self, cfg, trainer=None):
        # Exercise actual Serialization dispatch without allocating an ASR
        # network or constructing training datasets for a sentinel Trainer.
        torch.nn.Module.__init__(self)
        self._cfg = cfg
        self.tokenizer = SimpleNamespace(blank_id=5, inactive_native_ids=())
        seen.append((cfg, trainer))

    monkeypatch.setattr(EncDecRNNTBPEModelWithPrompt, "__init__", parent_constructor)
    native_class = get_native_nemo_model_class()
    config = OmegaConf.create({"target": "untok.native_runtime.NativeNemotronRNNTModel"})
    trainer = object()
    assert common.Serialization._inspect_signature_for_trainer(native_class)
    model = native_class.from_config_dict(config=config, trainer=trainer)
    assert type(model) is native_class
    assert len(seen) == 1 and seen[0][1] is trainer
    assert seen[0][0].target == config.target


@pytest.fixture
def native_bundle(tmp_path):
    proto = pb.ModelProto()
    proto.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    proto.trainer_spec.unk_id = 0
    proto.trainer_spec.bos_id = proto.trainer_spec.eos_id = proto.trainer_spec.pad_id = -1
    proto.normalizer_spec.name = "identity"
    proto.normalizer_spec.remove_extra_whitespaces = False
    for index, (piece, score) in enumerate([("<unk>", 0), ("▁", 0), ("a", -2), ("b", -3), ("z", -253), ("я", -4)]):
        proto.pieces.add(piece=piece, score=score, type=2 if index == 0 else 1)
    proto.trainer_spec.vocab_size = len(proto.pieces)
    base = tmp_path / "base.model"
    base.write_bytes(proto.SerializeToString())
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"base_tokenizer_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                                     "additions": [{"piece": "க", "score": -2}, {"piece": "▁க", "score": -1}]}))
    bundle = tmp_path / "bundle"
    build_native_tokenizer(base, selection, bundle)
    return bundle


class ArtifactModel:
    def __init__(self, cfg, directory):
        self.cfg, self.directory = cfg, directory
        self.directory.mkdir()
        self.registered = {}

    def register_artifact(self, key, source):
        # Emulate NeMo archive names instead of relying on original filenames.
        destination = self.directory / f"archive_{len(self.registered)}_{Path(source).name}"
        shutil.copyfile(source, destination)
        self.registered[key] = str(destination)
        self.cfg["bundle_files"][key.rsplit(".", 1)[-1]] = str(destination)
        return str(destination)


def test_native_bundle_survives_renamed_archive_paths_and_source_directory_removal(native_bundle, tmp_path, native_sentencepiece):
    original = load_tokenizer_bundle(native_bundle)
    cfg = native_bundle_config(native_bundle)
    first = ArtifactModel(cfg, tmp_path / "first-archive")
    _setup_native_tokenizer(first, cfg)
    assert len(first.registered) == 6
    assert first.tokenizer.model_bytes == original.model_bytes
    assert first.tokenizer.source_native_to_target_native == (0, 1, 2, 3, 4, 5, 8)
    shutil.rmtree(native_bundle)
    restored_cfg = copy.deepcopy(cfg)
    second = ArtifactModel(restored_cfg, tmp_path / "restored-archive")
    _setup_native_tokenizer(second, restored_cfg)
    assert second.native_bundle_manifest_sha256 == first.native_bundle_manifest_sha256
    assert second.tokenizer.id_map.to_dict() == original.id_map.to_dict()
    assert second.tokenizer.model_bytes == original.model_bytes
    assert second.tokenizer.base_model_bytes == original.base_model_bytes
    for text in ("", "  a  b ", "a\u200cb", "க", "aகb", "🙂a"):
        assert second.tokenizer.text_to_ids(text) == original.text_to_ids(text)
        assert second.tokenizer.ids_to_text(second.tokenizer.text_to_ids(text)) == original.ids_to_text(original.text_to_ids(text))
    assert second.tokenizer.ids_to_text([2, 2, 3]) == "aab"


def test_v5_archive_reconstructs_physical_vocabulary_and_required_mask(tmp_path, native_sentencepiece):
    torch = pytest.importorskip("torch")
    from test_native_checkpoint import toy_model

    bundle = Path(__file__).resolve().parents[1] / "src/untok/data/latin"
    cfg = native_bundle_config(bundle)
    first = ArtifactModel(cfg, tmp_path / "first-profile-archive")
    _setup_native_tokenizer(first, cfg)
    assert len(first.tokenizer.get_active_vocab()) == 2653
    assert len(first.tokenizer.get_vocab()) == 13087
    assert len(first.tokenizer.tokenizer.get_vocab()) == first.tokenizer.vocab_size == 13087
    assert set(first.tokenizer.tokenizer.get_vocab().values()) == set(range(13087))
    copied = copy.deepcopy(first.tokenizer)
    assert copied.tokenizer.get_vocab() == first.tokenizer.tokenizer.get_vocab()
    assert copied.get_vocab() == first.tokenizer.get_vocab()
    first.joint = toy_model(first.tokenizer.vocab_size).joint
    original_state = {name: value.clone() for name, value in first.joint.state_dict().items()}
    evidence = install_native_output_mask(first)
    assert evidence["inactive_output_rows"] == 10434
    state_path = tmp_path / "joint-state.pt"
    torch.save(first.joint.state_dict(), state_path)
    restored = ArtifactModel(copy.deepcopy(cfg), tmp_path / "restored-profile-archive")
    _setup_native_tokenizer(restored, restored.cfg)
    restored.joint = toy_model(restored.tokenizer.vocab_size).joint
    install_native_output_mask(restored)
    restored.joint.load_state_dict(torch.load(state_path, weights_only=True))
    assert verify_native_output_mask(restored) == evidence
    assert restored.tokenizer.inactive_native_ids == first.tokenizer.inactive_native_ids
    assert restored.tokenizer.tokenizer.get_vocab() == first.tokenizer.tokenizer.get_vocab()
    assert all(torch.equal(value, restored.joint.state_dict()[name]) for name, value in original_state.items())


@pytest.mark.parametrize("failure", ["wrong_type", "missing_file", "unsafe_name", "manifest_hash", "extra_file"])
def test_native_runtime_rejects_ambiguous_artifacts(native_bundle, tmp_path, failure):
    cfg = native_bundle_config(native_bundle)
    if failure == "wrong_type": cfg["type"] = "untok_hf_bpe"
    elif failure == "missing_file": cfg["bundle_files"].pop("file_0")
    elif failure == "unsafe_name": cfg["bundle_filenames"]["file_0"] = "../outside.model"
    elif failure == "manifest_hash": cfg["bundle_manifest_sha256"] = "0" * 64
    else:
        cfg["bundle_filenames"]["file_9"] = "extra.model"
        cfg["bundle_files"]["file_9"] = str(native_bundle / "tokenizer.model")
    model = ArtifactModel(cfg, tmp_path / "archive")
    with pytest.raises(ValueError):
        _setup_native_tokenizer(model, cfg)


def test_native_registration_does_not_allow_legacy_or_other_classes(monkeypatch):
    class Serialization:
        pass

    class NativeClass(Serialization):
        pass

    target = "untok.native_runtime.NativeNemotronRNNTModel"
    resolved = {target: NativeClass}
    common = ModuleType("nemo.core.classes.common")
    common.Serialization = Serialization
    common._is_target_allowed = lambda name: name == "nemo.collections.ExistingModel"
    common.ALLOWED_TARGET_PREFIXES = ["nemo.collections."]
    common.hydra = SimpleNamespace(utils=SimpleNamespace(get_class=lambda name: resolved[name]))
    monkeypatch.setitem(sys.modules, common.__name__, common)
    _register_native_target(NativeClass)
    predicate = common._is_target_allowed
    assert predicate(target) and predicate("nemo.collections.ExistingModel")
    for other in ("untok.runtime.ExtendedNemotronRNNTModel", "untok.other.Model", target + "Alias", "os.system"):
        assert not predicate(other)
    assert common.ALLOWED_TARGET_PREFIXES == ["nemo.collections."]
    resolved[target] = type("Other", (Serialization,), {})
    assert not predicate(target)
    _register_native_target(NativeClass)
    assert common._is_target_allowed is predicate
    with pytest.raises(ValueError, match="Serialization"):
        _register_native_target(object)


def test_native_migration_requires_checkpoint_pin_before_nemo_or_output(native_bundle, tmp_path):
    source = tmp_path / "source.nemo"
    source.write_bytes(b"unit-test checkpoint bytes")
    destination = tmp_path / "new.nemo"
    with pytest.raises(ValueError, match="explicit pin"):
        migrate_native_checkpoint(source, native_bundle, destination, expected_source_sha256="0" * 64)
    assert not destination.exists() and not destination.with_suffix(".migration.json").exists()
    correct = hashlib.sha256(source.read_bytes()).hexdigest()
    destination.write_bytes(b"existing output")
    with pytest.raises(ValueError, match="new output paths"):
        migrate_native_checkpoint(source, native_bundle, destination, expected_source_sha256=correct)
    assert destination.read_bytes() == b"existing output"


@pytest.mark.parametrize("corrupt_reload", [False, True])
@pytest.mark.parametrize("profile", ["full", "latin-indic", "latin"])
@pytest.mark.parametrize("configured", [False, True])
def test_native_migration_orchestration_saves_verifies_and_rejects_corrupt_reload(
    monkeypatch, native_bundle, tmp_path, corrupt_reload, profile, configured,
):
    """The archive is a unit-test stand-in; actual NeMo remains an integration gate."""
    torch = pytest.importorskip("torch")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    import sentencepiece as spm
    import untok.native_checkpoint as migration
    from test_native_checkpoint import toy_model

    if profile != "full":
        package_tokenizer_bundles(native_bundle, tmp_path / "variants", profiles=[profile], make_zips=False)
        native_bundle = tmp_path / "variants" / profile

    convert = OmegaConf.create
    plain = lambda cfg: OmegaConf.to_container(cfg, resolve=True)
    # These are migration orchestration tests, not native tokenizer API tests.
    # The native facade has its own tests against installed NVIDIA code.
    monkeypatch.setattr("untok.nemo_tokenizer.create_nemo_tokenizer", lambda adapter, path: adapter)

    def copy_modules(model, size):
        toy = toy_model(size)
        for name, module in toy.named_children():
            model.add_module(name, module)
        model.register_buffer("retained_buffer", toy.retained_buffer.clone())

    class SourceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            backend = spm.SentencePieceProcessor(model_file=str(native_bundle / "base-tokenizer.model"))
            self.tokenizer = SimpleNamespace(tokenizer=backend, vocab_size=backend.get_piece_size(),
                ids_to_tokens=lambda ids: [backend.id_to_piece(i) for i in ids],
                text_to_ids=lambda text: backend.encode(text, out_type=int))
            self.cfg = convert({"model_defaults": {"num_prompts": 64, "prompt_dictionary": {"auto": 0, "en-US": 1}},
                                "tokenizer": {"type": "bpe"}, "encoder": {"att_context_size": [56, 13]},
                                "train_ds": {"manifest": "private-source"}})
            copy_modules(self, backend.get_piece_size())

    class NativeModel(torch.nn.Module):
        def __init__(self, cfg, trainer):
            super().__init__()
            self.cfg = cfg
            if configured:
                assert cfg.encoder.att_context_size == [[56, 0], [56, 1]]
                assert cfg.freeze_updates.modules.encoder == -1
            self.registered = {}
            _setup_native_tokenizer(self, cfg.tokenizer)
            copy_modules(self, self.tokenizer.vocab_size)

        def register_artifact(self, key, path):
            self.registered[key.rsplit(".", 1)[-1]] = Path(path).read_bytes()
            return path

        def save_to(self, path):
            payload = {"cfg": plain(self.cfg), "state": self.state_dict(), "artifacts": self.registered}
            torch.save(payload, path)

        @classmethod
        def restore_from(cls, path, map_location):
            payload = torch.load(path, map_location=map_location, weights_only=True)
            cfg = convert(payload["cfg"])
            directory = tmp_path / "restored-payload"
            directory.mkdir()
            for key, data in payload["artifacts"].items():
                artifact = directory / f"renamed-{key}"
                artifact.write_bytes(data)
                cfg.tokenizer.bundle_files[key] = str(artifact)
            restored = cls(cfg, trainer=None)
            restored.load_state_dict(payload["state"])
            if corrupt_reload is True:
                with torch.no_grad(): restored.encoder.weight[0, 0] += 1
            elif corrupt_reload == "settings":
                restored.cfg.freeze_updates.modules.encoder = 0
            return restored

    source_model = SourceModel()
    for name in ("nemo", "nemo.collections", "nemo.collections.asr", "nemo.collections.asr.models",
                 "nemo.collections.asr.models.rnnt_bpe_models_prompt"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["nemo.collections.asr.models"].ASRModel = SimpleNamespace(restore_from=lambda *args, **kwargs: source_model)
    sys.modules["nemo.collections.asr.models.rnnt_bpe_models_prompt"].EncDecRNNTBPEModelWithPrompt = SourceModel
    monkeypatch.setattr(migration, "get_native_nemo_model_class", lambda: NativeModel)
    source = tmp_path / "pinned.nemo"
    source.write_bytes(b"pinned synthetic source for orchestration only")
    destination = tmp_path / "migrated.nemo"
    kwargs = {"expected_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    if configured:
        settings, template, overrides = (tmp_path / name for name in ("settings.yaml", "template.yaml", "overrides.yaml"))
        settings.write_text("encoder:\n  att_context_size: [[56, 0], [56, 1]]\nfreeze_updates:\n  enabled: true\n  modules:\n    encoder: -1\n")
        template.write_text("model:\n  train_ds:\n    manifest_filepath: null\n  optim:\n    name: adamw\n    lr: 0.001\ntrainer:\n  max_epochs: 1\n")
        overrides.write_text("model:\n  train_ds:\n    manifest_filepath: train.jsonl\n")
        kwargs.update(model_config=settings, training_template=template, training_overrides=overrides)
    if corrupt_reload == "publication":
        real_link = migration.os.link
        def occupied_sidecar(temporary, output):
            if output == destination.with_suffix(".train.yaml"):
                output.write_text("another writer's artifact")
            real_link(temporary, output)
        monkeypatch.setattr(migration.os, "link", occupied_sidecar)
    if corrupt_reload:
        error = FileExistsError if corrupt_reload == "publication" else ValueError
        match = ("Native model settings changed" if corrupt_reload == "settings" else
                 None if corrupt_reload == "publication" else "Retained learned state changed")
        with pytest.raises(error, match=match):
            migration.migrate_native_checkpoint(source, native_bundle, destination, **kwargs)
        assert not destination.exists() and not destination.with_suffix(".migration.json").exists()
        if corrupt_reload == "publication":
            assert destination.with_suffix(".train.yaml").read_text() == "another writer's artifact"
        else:
            assert not destination.with_suffix(".train.yaml").exists()
    else:
        report = migration.migrate_native_checkpoint(source, native_bundle, destination, **kwargs)
        assert destination.exists() and destination.with_suffix(".migration.json").exists()
        assert report["before_save"]["passed"] and report["after_reload"]["passed"]
        assert report["after_reload"]["all_source_values_preserved"] == (profile == "full")
        assert report["old_model_to_new_model"] == {
            "full": [0, 1, 2, 3, 4, 5, 8],
            "latin-indic": [0, 1, 2, 3, 4, None, 7],
            "latin": [0, 1, 2, 3, 4, None, 5],
        }[profile]
        assert len(report["prompt_registry"]["target_assignments"]) == 22
        assert report["new_row_initialization"]["verified_after_reload"]
        assert report["initialization"] == "text-donor"
        policy = report["new_row_initialization"]
        if profile == "latin":
            assert policy["policy"] == "no_added_rows"
        else:
            assert policy["policy"] == "retained_text_donor_mean_bound_v1"
            assert policy["max_new_mass_ratio"] == .05
        if configured:
            sidecar = destination.with_suffix(".train.yaml")
            trained = OmegaConf.load(sidecar)
            assert trained.init_from_nemo_model == str(destination.resolve())
            assert trained.model.train_ds.manifest_filepath == "train.jsonl"
            assert trained.model.tokenizer.update_tokenizer is False
            assert report["training_config"]["sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
            assert report["model_settings"]["effective"]["encoder"]["att_context_size"] == [[56, 0], [56, 1]]
        else:
            assert report["model_settings"] is None and report["training_config"] is None
        assert not report["asr_accuracy_evaluated"] and not report["training_forward_evaluated"]
        assert source_model.cfg.train_ds == {"manifest": "private-source"}


@pytest.mark.parametrize("failure", ["settings", "publication"])
def test_native_migration_rejects_changed_settings_and_rolls_back_only_own_files(
    monkeypatch, native_bundle, tmp_path, failure,
):
    test_native_migration_orchestration_saves_verifies_and_rejects_corrupt_reload(
        monkeypatch, native_bundle, tmp_path, failure, "full", True)


def test_native_migration_options_and_sidecar_collision_fail_before_nemo_import(tmp_path):
    source, output = tmp_path / "source.nemo", tmp_path / "output.nemo"
    source.write_bytes(b"pinned source")
    kwargs = {"expected_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    for initialization in ("blank", "text-donor", "random"):
        with pytest.raises(TypeError, match="unexpected keyword argument 'initialization'"):
            migrate_native_checkpoint(source, "latin-indic", output, initialization=initialization, **kwargs)
    with pytest.raises(ValueError, match="require a training template"):
        migrate_native_checkpoint(source, "latin-indic", output, training_overrides="extra.yaml", **kwargs)
    output.with_suffix(".train.yaml").write_text("existing user data")
    with pytest.raises(ValueError, match="new output paths"):
        migrate_native_checkpoint(source, "latin-indic", output, training_template="native.yaml", **kwargs)
    assert output.with_suffix(".train.yaml").read_text() == "existing user data"


def test_native_migration_named_bundle_resolves_packaged_directory(monkeypatch, tmp_path):
    from importlib import resources
    source = tmp_path / "source.nemo"
    source.write_bytes(b"pinned source")
    observed = []
    def loaded(directory):
        observed.append(directory)
        raise RuntimeError("stop before native model loading")
    monkeypatch.setattr("untok.bundles.load_tokenizer_bundle", loaded)
    with pytest.raises(RuntimeError, match="stop before"):
        migrate_native_checkpoint(source, "latin-indic", tmp_path / "output.nemo",
                                  expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    assert observed == [Path(resources.files("untok").joinpath("data", "latin-indic")).resolve()]
