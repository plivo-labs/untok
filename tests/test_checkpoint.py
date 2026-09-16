"""Native tensor-layout, retained-logit and public-ID contracts."""
import pytest

from untok.checkpoint import inspect_nemo_layout, compare_old_logits, mask_new_outputs_for_test
from untok.runtime import IdMap


def _model(vocabulary_size):
    torch = pytest.importorskip("torch")

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blank_idx = vocabulary_size
            self.blank_as_pad = True
            self.prediction = torch.nn.ModuleDict({
                "embed": torch.nn.Embedding(vocabulary_size + 1, 5, padding_idx=vocabulary_size),
                "rnn": torch.nn.LSTM(5, 5, batch_first=True),
            })

    class Joint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._num_extra_outputs = 0
            self.pred = torch.nn.Linear(5, 5)
            self.enc = torch.nn.Linear(7, 5)
            self.joint_net = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(5, vocabulary_size + 1))

    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(8, 7)
    model.decoder = Decoder()
    model.joint = Joint()
    model.prompt_kernel = torch.nn.Linear(128, 7)
    model.register_buffer("retained_buffer", torch.tensor([7, 11]))
    return model


def test_native_layout_recognizes_the_blank_padding_and_output_rows():
    model = _model(4)
    layout = inspect_nemo_layout(model)
    assert layout.blank_id == 4 and layout.output_size == 5
    assert layout.embedding_key == "decoder.prediction.embed.weight"
    assert layout.output_weight_key == "joint.joint_net.1.weight"
    assert layout.output_bias_key == "joint.joint_net.1.bias"


@pytest.mark.parametrize("failure", ["ctc", "extra_outputs", "padding", "output_size", "shared_parameter"])
def test_native_layout_rejects_unsupported_heads(failure):
    torch = pytest.importorskip("torch")
    model = _model(4)
    if failure == "ctc":
        model.ctc_decoder = object()
    elif failure == "extra_outputs":
        model.joint._num_extra_outputs = 1
    elif failure == "padding":
        model.decoder.prediction["embed"].padding_idx = 0
    elif failure == "output_size":
        model.joint.joint_net[-1] = torch.nn.Linear(5, 7)
    else:
        model.shared_weight = model.joint.joint_net[-1].weight
    with pytest.raises(ValueError, match="CTC|Extra output|blank index|Shared"):
        inspect_nemo_layout(model)


def test_retained_logit_comparison_and_validation_mask():
    torch = pytest.importorskip("torch")
    original = torch.tensor([[1., 2., 3., 4., 5.]])
    expanded = torch.tensor([[1., 2., 3., 4., 100., 200., 5.]])
    mapping = (0, 1, 2, 3, 6)
    assert compare_old_logits(original, expanded, mapping)["max_absolute_error"] == 0
    masked = mask_new_outputs_for_test(expanded, mapping)
    assert torch.isneginf(masked[:, 4:6]).all()
    assert torch.equal(masked[:, list(mapping)], original)
    assert expanded[0, 4] == 100
    expanded[0, 6] += 1
    with pytest.raises(ValueError, match="logits changed"):
        compare_old_logits(original, expanded, mapping)
    with pytest.raises(ValueError, match="Invalid old-output map"):
        mask_new_outputs_for_test(expanded, (0, 0))


def test_public_id_map_preserves_padding_and_blank_spaces():
    mapping = IdMap((0, 1, None, 3, 2), (0, 1, 4, 3), 2, 3, 3, "hash")
    assert mapping.to_model([4, 1, 4]) == [2, 1, 2]
    assert mapping.to_canonical([2, 1, 3]) == [4, 1]
    assert mapping.to_canonical([3], drop_blank=False) == [3]
    with pytest.raises(ValueError, match="padding"):
        mapping.to_model([2])
    with pytest.raises(ValueError, match="blank"):
        mapping.to_model([3])
