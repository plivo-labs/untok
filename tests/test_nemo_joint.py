"""Torch unit checks plus optional integration with the pinned NVIDIA RNNTJoint."""
import importlib.util
from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
SOURCE = Path(__file__).parents[1] / "src/untok/nemo.py"
VOCABULARY = ["<unk>", "<unused_nemotron_1>", "word", "<unused_nemotron_3>", "last"]
JOINTNET = {"encoder_hidden": 4, "pred_hidden": 4, "joint_hidden": 4, "activation": "relu"}


@pytest.fixture
def integration(monkeypatch):
    """Use a small RNNT stand-in locally; real NeMo checks are separate below."""
    class StandInJoint(torch.nn.Module):
        def __init__(self, jointnet, num_classes, vocabulary=None, **kwargs):
            super().__init__()
            self.vocabulary, self.forwarded_options = vocabulary, kwargs
            self.joint_net = torch.nn.Sequential(
                torch.nn.ReLU(), torch.nn.Linear(jointnet["joint_hidden"], num_classes + 1)
            )

    for name in (
        "nemo", "nemo.collections", "nemo.collections.asr", "nemo.collections.asr.modules",
        "nemo.collections.asr.modules.rnnt", "nemo.core", "nemo.core.classes", "nemo.core.classes.common",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["nemo"].__file__ = "/nvidia/nemo/__init__.py"
    sys.modules["nemo.collections.asr.modules.rnnt"].RNNTJoint = StandInJoint
    common = sys.modules["nemo.core.classes.common"]
    common.ALLOWED_TARGET_PREFIXES = ["nemo.collections."]
    common._is_target_allowed = lambda target: target.startswith("nemo.collections.")
    spec = importlib.util.spec_from_file_location("untok_nemo_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, common


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("training", [True, False])
def test_mask_keeps_parameters_state_keys_blank_and_finite_gradients(integration, dtype, training):
    module, _ = integration
    source = torch.nn.Linear(4, 6).to(dtype=dtype).train(training)
    head = module._MaskedLinear(source, [1, 3])
    assert head.weight is source.weight and head.bias is source.bias
    assert set(head.state_dict()) == set(source.state_dict()) == {"weight", "bias"}
    assert head.training is training and head.inactive_output_mask.dtype == torch.bool
    inputs = torch.randn(2, 4, dtype=dtype, requires_grad=True)
    raw, masked = source(inputs), head(inputs)
    assert torch.isneginf(masked[..., [1, 3]]).all()
    assert torch.equal(masked[..., [0, 2, 4, 5]], raw[..., [0, 2, 4, 5]])
    loss = -masked.float().log_softmax(-1)[..., 2].mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(inputs.grad).all()
    assert torch.isfinite(head.weight.grad).all() and torch.isfinite(head.bias.grad).all()
    assert not head.weight.grad[[1, 3]].any() and not head.bias.grad[[1, 3]].any()


def test_joint_rebuilds_mask_from_exact_vocabulary_and_reloads_state(integration):
    module, _ = integration
    joint = module.MaskedRNNTJoint(JOINTNET, 5, vocabulary=VOCABULARY, masking_prob=0.2)
    assert joint.forwarded_options == {"masking_prob": 0.2}
    assert joint.joint_net[-1].inactive_output_mask.tolist() == [False, True, False, True, False, False]
    restored = module.MaskedRNNTJoint(JOINTNET, 5, vocabulary=VOCABULARY)
    restored.load_state_dict(joint.state_dict(), strict=True)
    probe = torch.ones(2, 4)
    assert torch.equal(joint.joint_net(probe), restored.joint_net(probe))
    ordinary = module.MaskedRNNTJoint(JOINTNET, 2, vocabulary=["<unused_nemotron_1>", "text"])
    assert type(ordinary.joint_net[-1]) is torch.nn.Linear
    unmasked = module.MaskedRNNTJoint(JOINTNET, 5, vocabulary=VOCABULARY, mask_unused_tokens=False)
    assert type(unmasked.joint_net[-1]) is torch.nn.Linear
    with pytest.raises(ValueError, match="physical text ID"):
        module.MaskedRNNTJoint(JOINTNET, 5)


def test_registration_adds_only_full_target_without_replacing_validation(integration):
    module, common = integration
    predicate = common._is_target_allowed
    module.register()
    module.register()
    assert common.ALLOWED_TARGET_PREFIXES == ["nemo.collections.", "untok.nemo.MaskedRNNTJoint"]
    assert common._is_target_allowed is predicate
    common.ALLOWED_TARGET_PREFIXES = ()
    with pytest.raises(RuntimeError, match="Unsupported NeMo"):
        module.register()


@pytest.mark.parametrize("command,suffix", [
    ("train", "asr_transducer/speech_to_text_rnnt_bpe_prompt.py"),
    ("evaluate", "asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"),
])
def test_launcher_passes_native_arguments_unchanged(integration, monkeypatch, command, suffix):
    calls = []
    arguments = [str(SOURCE), command, "--config-path=/config", "model.train_ds.batch_size=4"]
    monkeypatch.setattr(sys, "argv", arguments.copy())
    monkeypatch.setattr(runpy, "run_path", lambda *args, **kwargs: calls.append((args, kwargs)))
    exec(compile(SOURCE.read_text(), str(SOURCE), "exec"), {"__name__": "__main__"})
    assert calls == [(("/nvidia/examples/asr/" + suffix,), {"run_name": "__main__"})]
    assert sys.argv == arguments[:1] + arguments[2:]


@pytest.mark.parametrize("backend", ["pytorch", "warprnnt_numba"])
@pytest.mark.parametrize("fused", [False, True])
def test_real_nemo_joint_loss_and_registration(backend, fused):
    """Requires pinned NeMo; includes native fused and projected decoding paths."""
    pytest.importorskip("nemo.collections.asr.modules.rnnt")
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    from nemo.core.classes import common
    from untok.nemo import MaskedRNNTJoint, register

    register()
    assert common._is_target_allowed("untok.nemo.MaskedRNNTJoint")
    assert not common._is_target_allowed("untok.bundles.load_tokenizer_bundle")
    joint = MaskedRNNTJoint(JOINTNET, 5, vocabulary=VOCABULARY, log_softmax=None,
                           fuse_loss_wer=fused, fused_batch_size=1 if fused else None)
    criterion = RNNTLoss(num_classes=5, loss_name=backend, reduction="sum")
    encoder = torch.randn(2, 4, 3, requires_grad=True)
    decoder = torch.randn(2, 4, 3, requires_grad=True)
    labels = torch.tensor([[0, 2], [2, 4]], dtype=torch.int64)
    input_lengths, target_lengths = torch.tensor([3, 3]), torch.tensor([2, 2])
    projected = joint.joint_after_projection(
        joint.project_encoder(encoder.transpose(1, 2)), joint.project_prednet(decoder.transpose(1, 2)))
    assert torch.isneginf(projected[..., [1, 3]]).all()
    assert torch.isfinite(projected[..., [0, 2, 4, 5]]).all()
    if fused:
        joint.set_fuse_loss_wer(True, loss=criterion, metric=torch.nn.Identity())
        loss, *_ = joint(encoder_outputs=encoder, decoder_outputs=decoder, encoder_lengths=input_lengths,
                         transcripts=labels, transcript_lengths=target_lengths, compute_wer=False)
    else:
        loss = criterion(log_probs=joint(encoder_outputs=encoder, decoder_outputs=decoder), targets=labels,
                         input_lengths=input_lengths, target_lengths=target_lengths)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(encoder.grad).all() and torch.isfinite(decoder.grad).all()
    for parameter in (joint.joint_net[-1].weight, joint.joint_net[-1].bias):
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad[[1, 3]]) == 0
