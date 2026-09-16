"""Native probe controls with small models; no real audio or ASR claims."""
import copy
from types import SimpleNamespace

import pytest

import untok.checkpoint_validation as runner
from untok.runtime import IdMap


@pytest.fixture
def model():
    torch = pytest.importorskip("torch")

    class FakeNativeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.cfg = {
                "encoder": {"att_context_size": [56, 3]},
                "preprocessor": {"sample_rate": 16000},
                "decoder": {"vocab_size": 4}, "joint": {"num_classes": 4},
                "model_defaults": {"num_prompts": 128, "enc_hidden": 2,
                                   "prompt_dictionary": {"hi-IN": 6, "auto": 101}},
                "decoding": {"strategy": "greedy_batch", "greedy": {"max_symbols": 10}},
            }
            self._set_decoder(True)

        def _set_decoder(self, graphs):
            self.decoding = SimpleNamespace(decoding=SimpleNamespace(
                max_symbols=10, loop_labels=True, use_cuda_graph_decoder=graphs,
                decoding_computer=SimpleNamespace(
                    allow_cuda_graphs=graphs, cuda_graphs_mode="FULL_GRAPH" if graphs else None)))

        def change_decoding_strategy(self, config, verbose=False):
            self.cfg["decoding"] = copy.deepcopy(config)
            self._set_decoder(config["greedy"]["use_cuda_graph_decoder"])

    return FakeNativeModel()


def test_eager_instrumentation_preserves_weights_and_decoding(model):
    torch = pytest.importorskip("torch")
    before = {name: value.clone() for name, value in model.state_dict().items()}
    evidence = runner._configure_eager_decoding(model)
    assert "use_cuda_graph_decoder" not in evidence["serialized_config"]["greedy"]
    assert evidence["effective_config"]["greedy"]["use_cuda_graph_decoder"] is False
    assert evidence["runtime"]["use_cuda_graph_decoder"] is False
    assert evidence["runtime"]["allow_cuda_graphs"] is False
    assert evidence["runtime"]["max_symbols"] == 10
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


@pytest.mark.parametrize("failure", ["graph_flag", "computer_flag", "active_graph", "max_symbols", "weights"])
def test_eager_setup_rejects_unapplied_flags_or_unintended_changes(model, monkeypatch, failure):
    torch = pytest.importorskip("torch")
    real_change = model.change_decoding_strategy

    def broken_change(config, **kwargs):
        real_change(config, **kwargs)
        decoder = model.decoding.decoding
        if failure == "graph_flag":
            decoder.use_cuda_graph_decoder = True
        elif failure == "computer_flag":
            decoder.decoding_computer.allow_cuda_graphs = True
        elif failure == "active_graph":
            decoder.decoding_computer.cuda_graphs_mode = "FULL_GRAPH"
        elif failure == "max_symbols":
            decoder.max_symbols = 20
        else:
            with torch.no_grad():
                model.encoder.weight.add_(1)

    monkeypatch.setattr(model, "change_decoding_strategy", broken_change)
    with pytest.raises(ValueError, match="CUDA graph|algorithm or max_symbols|Model tensors changed"):
        runner._configure_eager_decoding(model)


def test_serialized_decoder_mismatch_is_rejected_before_eager_setup(model):
    changed = copy.deepcopy(model)
    changed.cfg["decoding"]["greedy"]["max_symbols"] = 20
    with pytest.raises(ValueError, match="Inference configuration changed"):
        runner._equivalent_inference_config(model, changed, "greedy_batch")
    assert model.decoding.decoding.use_cuda_graph_decoder is True


def test_head_trace_masks_before_softmax_and_always_removes_hooks():
    torch = pytest.importorskip("torch")
    head = torch.nn.Linear(2, 4)
    inputs = torch.ones(1, 2)
    with runner._head_trace(head, old_to_new=(0, 1, 3)) as trace:
        output = head(inputs)
        assert torch.isneginf(output[0, 2])
    assert trace["calls"] == 1
    assert torch.isfinite(trace["probes"][0][1]).all()
    assert not head._forward_hooks
    with pytest.raises(RuntimeError):
        with runner._head_trace(head):
            raise RuntimeError("fixture")
    assert not head._forward_hooks


def test_hypothesis_requires_ids_and_preserves_repetition():
    mapping = IdMap((0, 1, None, 2), (0, 1, 3), 2, 3, 2, "hash")
    result = runner._hypothesis([SimpleNamespace(text="aa", y_sequence=[1, 1, 2])], mapping)
    assert result["model_ids"] == [1, 1, 2]
    assert result["canonical_hf_ids"] == [1, 1, 3]
    with pytest.raises(ValueError, match="y_sequence"):
        runner._hypothesis(["aa"], mapping)
