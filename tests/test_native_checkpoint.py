"""Original-to-current migration preserves every learned value and blank placement."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.bundles import load_tokenizer_bundle
from untok.checkpoint import inspect_nemo_layout
from untok.native_checkpoint import (
    retained_row_pairs, transfer_native_state_dict,
    validate_source_native_tokenizer, verify_native_state_transfer,
)
from untok.native_donors import initialize_text_donor_rows


def toy_model(vocabulary_size):
    torch = pytest.importorskip("torch")

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blank_idx = vocabulary_size
            self.blank_as_pad = True
            self.prediction = torch.nn.ModuleDict({
                "embed": torch.nn.Embedding(vocabulary_size + 1, 4, padding_idx=vocabulary_size),
                "rnn": torch.nn.LSTM(4, 4, batch_first=True),
            })

    class Joint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._num_extra_outputs = 0
            self.joint_net = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(4, vocabulary_size + 1))

    model = torch.nn.Module()
    model.decoder, model.joint = Decoder(), Joint()
    model.encoder = torch.nn.Linear(4, 4)
    model.prompt_kernel = torch.nn.Linear(4, 4)
    model.register_buffer("retained_buffer", torch.tensor([7, 11]))
    return model



@pytest.mark.parametrize("profile", ["original", "latin", "latin-indic", "full"])
def test_all_source_values_blank_and_added_rows_survive_roundtrip(profile, tmp_path):
    torch = pytest.importorskip("torch")
    adapter = load_tokenizer_bundle(Path(__file__).parents[1] / "src/untok/data" / profile)
    mapping = adapter.source_native_to_target_native
    source, target = toy_model(13087), toy_model(adapter.vocab_size)
    old, new = inspect_nemo_layout(source), inspect_nemo_layout(target)
    assert mapping == (*range(13087), adapter.blank_id)
    assert old.blank_id == 13087
    assert new.blank_id == (13087 if profile in {"original", "latin"} else 20360)
    with torch.no_grad():
        weight = source.decoder.prediction["embed"].weight
        weight.copy_(torch.arange(weight.numel()).reshape_as(weight) / weight.numel())
        weight[old.blank_id] = 0
    original = copy.deepcopy(source.state_dict())
    transferred = transfer_native_state_dict(original, target.state_dict(), old, new, mapping)
    initialized, policy = initialize_text_donor_rows(transferred, new, adapter, mapping)
    added = policy["new_model_rows"]
    assert added == list(range(13087, new.blank_id))
    target.load_state_dict(initialized)
    target.tokenizer = adapter
    report = verify_native_state_transfer(original, target.state_dict(), old, new, mapping)
    assert report["all_source_values_preserved"] and report["learned_values_omitted"] == 0
    assert report["learned_values_preserved"] == sum(value.numel() for value in original.values())
    for key in old.row_keys:
        assert torch.equal(target.state_dict()[key][list(mapping)], original[key])
        assert torch.equal(target.state_dict()[key][new.blank_id], original[key][old.blank_id])
    if added:
        assert policy["policy"] == "retained_text_donor_mean_bound_v1"
        assert policy["max_new_mass_ratio"] == .05
        assert torch.all(initialized[new.embedding_key][added] > 0)
        assert not torch.equal(initialized[new.output_weight_key][added],
                               original[old.output_weight_key][old.blank_id].expand(len(added), -1))
    # Identical retained prefixes preserve raw logits. Inactive rows are the
    # intentional output mask; added rows remain enabled and mass-bounded.
    active = [i for i in range(13087) if i not in set(adapter.inactive_native_ids)]
    prefix = torch.tensor([[old.blank_id, *active[:3]]])
    target_prefix = torch.tensor([[mapping[i] for i in prefix[0].tolist()]])
    features = torch.randn(1, prefix.shape[1], 4)
    def forward(model, ids):
        predicted, _ = model.decoder.prediction["rnn"](model.decoder.prediction["embed"](ids))
        return model.joint.joint_net(model.encoder(features) + predicted)
    left, right = forward(source, prefix), forward(target, target_prefix)
    torch.testing.assert_close(left[..., active + [old.blank_id]], right[..., active + [new.blank_id]])
    if added:
        ratio = right[..., added].double().logsumexp(-1) - right[..., active + [new.blank_id]].double().logsumexp(-1)
        assert float(ratio.detach().exp().max()) <= .05 + 1e-7
    path = tmp_path / "state.pt"
    torch.save(target.state_dict(), path)
    restored = torch.load(path, weights_only=True)
    assert verify_native_state_transfer(original, restored, old, new, mapping)["passed"]
    for key in new.row_keys:
        assert torch.equal(restored[key][added], initialized[key][added])
    assert all(torch.equal(original[key], value) for key, value in source.state_dict().items())


@pytest.mark.parametrize("mapping", [
    (0, 1, 2, 3), (0, 1, 2, 3, None), (0, 1, 2, 3, 4),
    (0, 1, 1, 3, 6), (0, True, 2, 3, 6), (0, 1.0, 2, 3, 6),
    (0, -1, 2, 3, 6), (0, 7, 2, 3, 6), (0, None, 1, 2, 6),
])
def test_renumbered_removed_or_ambiguous_source_rows_are_rejected(mapping):
    with pytest.raises(ValueError, match="retain every original"):
        retained_row_pairs(inspect_nemo_layout(toy_model(4)), inspect_nemo_layout(toy_model(6)), mapping)


@pytest.mark.parametrize("tensor", ["encoder.weight", "prompt_kernel.weight", "decoder.prediction.embed.weight"])
def test_verifier_detects_changed_source_state(tensor):
    source, target = toy_model(4), toy_model(6)
    old, new = inspect_nemo_layout(source), inspect_nemo_layout(target)
    mapping = (0, 1, 2, 3, 6)
    state = transfer_native_state_dict(source.state_dict(), target.state_dict(), old, new, mapping)
    state[tensor][0, 0] += 1
    with pytest.raises(ValueError, match="Retained learned state changed"):
        verify_native_state_transfer(source.state_dict(), state, old, new, mapping)


@pytest.mark.parametrize("kind", ["dtype", "shape", "keys"])
def test_transfer_rejects_dtypes_shapes_and_missing_keys(kind):
    source, target = toy_model(4), toy_model(6)
    old, new = inspect_nemo_layout(source), inspect_nemo_layout(target)
    state = copy.deepcopy(target.state_dict())
    if kind == "dtype": state["encoder.weight"] = state["encoder.weight"].double()
    elif kind == "shape": state["encoder.weight"] = state["encoder.weight"][:1]
    else: del state["encoder.weight"]
    with pytest.raises(ValueError):
        transfer_native_state_dict(source.state_dict(), state, old, new, (0, 1, 2, 3, 6))


def test_source_checks_every_actual_sentencepiece_id_and_wrapper_encoding():
    data = (Path(__file__).parents[1] / "src/untok/data/original/tokenizer.model").read_bytes()
    backend = spm.SentencePieceProcessor(model_proto=data)
    tokenizer = SimpleNamespace(tokenizer=backend, vocab_size=13087,
        ids_to_tokens=lambda ids: [backend.id_to_piece(i) for i in ids],
        text_to_ids=lambda text: backend.encode(text, out_type=int))
    source = SimpleNamespace(tokenizer=tokenizer)
    assert validate_source_native_tokenizer(source, data)["native_tokenizer_sha256"] == hashlib.sha256(data).hexdigest()
    changed = pb.ModelProto.FromString(data)
    changed.pieces[2].score -= .1
    with pytest.raises(ValueError, match="pinned native base"):
        validate_source_native_tokenizer(source, changed.SerializeToString())
    tokenizer.text_to_ids = lambda text: [2]
    with pytest.raises(ValueError, match="wrapper encoding"):
        validate_source_native_tokenizer(source, data)
