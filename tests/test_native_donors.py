"""Donor coefficients, retained state and the initial probability bound."""
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import math
from types import SimpleNamespace

import pytest
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.checkpoint import RNNTLayout
from untok.native_donors import ExactDonors, donor_paths, initialize_text_donor_rows


def fixture():
    torch = pytest.importorskip("torch")
    base = pb.ModelProto()
    base.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    for spelling, score, kind in [("<unk>", 0, 2), ("<tag>", 0, 3), ("▁", -1, 1),
                                  ("a", -1, 1), ("b", -2, 1), ("ab", -3, 1)]:
        base.pieces.add(piece=spelling, score=score, type=kind)
    target = pb.ModelProto()
    target.CopyFrom(base)
    for spelling in ("aba", "▁aba", "Ω"):
        target.pieces.add(piece=spelling, score=-4, type=1)
    mapping = (0, 1, 2, 3, 4, 5, 9)
    adapter = SimpleNamespace(base_model_bytes=base.SerializeToString(), model_bytes=target.SerializeToString(),
                              source_native_to_target_native=mapping, inactive_native_ids=())
    layout = RNNTLayout("embedding", "weight", "bias", 9, 10)
    state = {"embedding": torch.arange(40, dtype=torch.float32).reshape(10, 4) / 13,
             "weight": torch.arange(30, dtype=torch.float32).reshape(10, 3) / 13,
             "bias": torch.arange(10, dtype=torch.float32) / 13,
             "shared": torch.tensor([1., 2.]), "buffer": torch.tensor(7)}
    return adapter, layout, state, mapping


def test_literal_paths_score_ties_and_repeated_coefficients():
    adapter, _, _, mapping = fixture()
    base = pb.ModelProto.FromString(adapter.base_model_bytes)
    finder = ExactDonors(base, mapping)
    assert finder.segment("aba") == (-4., (3, 4, 3))
    assert finder.segment("▁aba") == (-5., (2, 3, 4, 3))
    for unsupported in (" a", "ﬁ", "<unk>", "<tag>"):
        assert finder.segment(unsupported) is None
    normal, paths, cmax = donor_paths(adapter, [6, 7, 8], set(mapping))
    assert normal == (2, 3, 4, 5)
    assert paths == {6: (3, 4, 3), 7: (2, 3, 4, 3), 8: ()}
    assert cmax == Fraction(17, 12)
    base.pieces[5].score = -2
    assert ExactDonors(base, mapping).segment("aba") == (-3., (5, 3))


def test_exact_means_preserve_every_other_row_blank_shared_state_and_rng():
    torch = pytest.importorskip("torch")
    adapter, layout, state, mapping = fixture()
    before = {key: value.clone() for key, value in state.items()}
    rng = torch.random.get_rng_state().clone()
    result, policy = initialize_text_donor_rows(state, layout, adapter, mapping)
    for key in state:
        assert torch.equal(state[key], before[key])
        if key in layout.row_keys:
            assert torch.equal(result[key][list(mapping)], before[key][list(mapping)])
        else:
            assert torch.equal(result[key], before[key])
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(result["embedding"][6], before["embedding"][[3, 4, 3]].double().mean(0).float())
    assert torch.equal(result["weight"][8], before["weight"][[2, 3, 4, 5]].double().mean(0).float())
    assert torch.equal(result["bias"][8], before["bias"][[2, 3, 4, 5]].double().mean(0).float() + policy["bias_shift"])
    assert policy["bias_shift"] == math.log(.05 / (17 / 12))
    assert policy["max_new_mass_ratio"] == .05
    h = torch.linspace(-2, 2, 90).reshape(30, 3).double()
    old_z = (h @ result["weight"][list(mapping)].double().T + result["bias"][list(mapping)].double()).logsumexp(-1)
    new_z = (h @ result["weight"][[6, 7, 8]].double().T + result["bias"][[6, 7, 8]].double()).logsumexp(-1)
    assert bool(((new_z - old_z).exp() <= .05 + 1e-7).all())


def test_inactive_source_rows_are_protected_but_excluded_from_donors():
    torch = pytest.importorskip("torch")
    adapter, layout, state, mapping = fixture()
    target = pb.ModelProto.FromString(adapter.model_bytes)
    target.pieces[3].piece, target.pieces[3].type = "<unused_3>", pb.ModelProto.SentencePiece.UNUSED
    adapter.model_bytes, adapter.inactive_native_ids = target.SerializeToString(), (3,)
    normal, paths, cmax = donor_paths(adapter, [6, 7, 8], set(mapping))
    assert normal == (2, 4, 5) and all(not path for path in paths.values())
    result, policy = initialize_text_donor_rows(state, layout, adapter, mapping)
    for key in layout.row_keys:
        assert torch.equal(result[key][3], state[key][3])
    assert torch.equal(result["weight"][6], state["weight"][[2, 4, 5]].double().mean(0).float())
    assert cmax == 1 and policy["fallback_count"] == 3


@pytest.mark.parametrize("ratio", [0, -1, 1, float("nan"), float("inf")])
def test_invalid_mass_budget(ratio):
    adapter, layout, state, mapping = fixture()
    with pytest.raises(ValueError, match="mass ratio"):
        initialize_text_donor_rows(state, layout, adapter, mapping, max_new_mass_ratio=ratio)


def test_added_rows_require_output_bias_to_enforce_mass_bound():
    torch = pytest.importorskip("torch")
    adapter, layout, state, mapping = fixture()
    before = {key: value.clone() for key, value in state.items()}
    with pytest.raises(ValueError, match="requires a joint output bias"):
        initialize_text_donor_rows(state, replace(layout, output_bias_key=None), adapter, mapping)
    assert all(torch.equal(state[key], value) for key, value in before.items())


@pytest.mark.parametrize("failure", ["dtype", "shape", "nonfinite"])
def test_invalid_vocabulary_state_does_not_mutate_inputs(failure):
    torch = pytest.importorskip("torch")
    adapter, layout, state, mapping = fixture()
    before = state["embedding"].clone()
    if failure == "dtype":
        state["weight"] = state["weight"].double()
    elif failure == "shape":
        state["weight"] = state["weight"][:2]
    else:
        state["weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite CPU FP32"):
        initialize_text_donor_rows(state, layout, adapter, mapping)
    assert torch.equal(state["embedding"], before)


def test_previously_trained_additions_are_retained_and_no_additions_skip_donors(monkeypatch):
    torch = pytest.importorskip("torch")
    adapter, layout, state, mapping = fixture()
    result, policy = initialize_text_donor_rows(state, layout, adapter, (*mapping[:-1], 6, 9))
    assert policy["new_model_rows"] == [7, 8]
    for key in state:
        assert torch.equal(result[key][6], state[key][6]) if key in layout.row_keys else torch.equal(result[key], state[key])
    monkeypatch.setattr("untok.native_donors.donor_paths", lambda *args: pytest.fail("No additions need no donors"))
    result, policy = initialize_text_donor_rows(state, layout, adapter, tuple(range(10)))
    assert policy == {"policy": "no_added_rows", "new_row_count": 0, "new_model_rows": []}
    assert all(torch.equal(result[key], state[key]) for key in state)


def test_real_selected_bundle_matches_the_qualified_donor_recipe():
    from untok.bundles import load_tokenizer

    adapter = load_tokenizer("latin-indic")
    retained = set(adapter.source_native_to_target_native)
    added = sorted(set(range(adapter.blank_id)) - retained)
    normal, paths, cmax = donor_paths(adapter, added, retained)
    blob = json.dumps({str(i): list(paths[i]) for i in added}, sort_keys=True, separators=(",", ":")).encode()
    assert len(added) == 7273 and len(normal) == 3098
    assert sum(bool(path) for path in paths.values()) == 1462
    assert hashlib.sha256(blob).hexdigest() == "b89052bac1bb1c0451f392c4e6fd9cf61b96d5271229f3cbc143d08a0ff1dd90"
    assert cmax == Fraction(6895405, 43372)
    assert math.log(.05 / float(cmax)) == pytest.approx(-8.064528728662534, abs=1e-14, rel=0)
