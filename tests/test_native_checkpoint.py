"""Native full/subset row transfer, retained logits and fail-closed migration inputs."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.checkpoint import inspect_nemo_layout
from untok.native_checkpoint import (
    _prompt_registry, compare_retained_logits,
    retained_row_pairs, select_source_native_inventory, transfer_native_state_dict, validate_source_native_tokenizer,
    verify_native_state_transfer,
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


def toy_adapter(mapping, vocabulary_size):
    base, target = pb.ModelProto(), pb.ModelProto()
    for spelling in "abcd":
        base.pieces.add(piece=spelling, score=-1, type=pb.ModelProto.SentencePiece.NORMAL)
    for index in range(vocabulary_size):
        spelling = base.pieces[mapping.index(index)].piece if index in mapping else "acd"
        target.pieces.add(piece=spelling, score=-1, type=pb.ModelProto.SentencePiece.NORMAL)
    return SimpleNamespace(base_model_bytes=base.SerializeToString(), model_bytes=target.SerializeToString(),
                           source_native_to_target_native=mapping, inactive_native_ids=())


@pytest.mark.parametrize("size,mapping,removed", [
    (6, (0, 1, 2, 3, 6), 0),
    (5, (0, None, 1, 2, 5), 1),
    (2, (0, None, 1, None, 2), 2),
])
def test_full_and_subset_transfer_preserve_precisely_retained_rows(size, mapping, removed):
    torch = pytest.importorskip("torch")
    torch.manual_seed(112)
    source, target = toy_model(4), toy_model(size)
    old_layout, new_layout = inspect_nemo_layout(source), inspect_nemo_layout(target)
    original = copy.deepcopy(source.state_dict())
    target.load_state_dict(transfer_native_state_dict(original, target.state_dict(), old_layout, new_layout, mapping))
    report = verify_native_state_transfer(original, target.state_dict(), old_layout, new_layout, mapping)
    assert report["passed"] and report["removed_source_text_rows"] == removed
    assert report["all_source_values_preserved"] == (removed == 0)
    assert all(torch.equal(original[k], v) for k, v in source.state_dict().items())
    kept_old, kept_new = retained_row_pairs(old_layout, new_layout, mapping)
    for key in old_layout.row_keys:
        assert torch.equal(target.state_dict()[key][list(kept_new)], original[key][list(kept_old)])
        assert torch.equal(target.state_dict()[key][new_layout.blank_id], original[key][old_layout.blank_id])
    omitted = sum(original[k][[i for i, value in enumerate(mapping) if value is None]].numel() for k in old_layout.row_keys)
    assert report["learned_values_omitted"] == omitted
    assert report["learned_values_preserved"] + omitted == sum(v.numel() for v in original.values())


@pytest.mark.parametrize("mapping,size", [((0, 1, 2, 3, 6), 6), ((0, None, 1, 2, 5), 5)])
def test_retained_prediction_prefixes_logits_and_new_rows_are_trainable(mapping, size):
    torch = pytest.importorskip("torch")
    torch.manual_seed(33)
    source, target = toy_model(4), toy_model(size)
    old_layout, new_layout = inspect_nemo_layout(source), inspect_nemo_layout(target)
    transferred = transfer_native_state_dict(source.state_dict(), target.state_dict(), old_layout, new_layout, mapping)
    initial, policy = initialize_text_donor_rows(transferred, new_layout, toy_adapter(mapping, size), mapping)
    target.load_state_dict(initial)
    old_prefix = torch.tensor([[4, 0, 2, 2, 3]])
    new_prefix = torch.tensor([[mapping[i] for i in old_prefix[0].tolist()]])
    audio = torch.randn(1, 5, 4)

    def forward(model, ids):
        predicted, _ = model.decoder.prediction["rnn"](model.decoder.prediction["embed"](ids))
        return model.joint.joint_net(model.encoder(audio) + predicted)

    left, right = forward(source, old_prefix), forward(target, new_prefix)
    assert compare_retained_logits(left, right, old_layout, new_layout, mapping)["passed"]
    old_ids, new_ids = retained_row_pairs(old_layout, new_layout, mapping)
    added = policy["new_model_rows"]
    ratio = right[..., added].double().logsumexp(-1) - right[..., list(new_ids)].double().logsumexp(-1)
    assert float(ratio.detach().exp().max()) <= .05 + 1e-7
    expected = source.state_dict()[old_layout.embedding_key][[0, 2, 3]].double().mean(0).float()
    assert torch.equal(target.state_dict()[new_layout.embedding_key][added[0]], expected)
    new_labels = torch.tensor([[added[0], added[1], added[0], added[1], added[0]]])
    loss = torch.nn.functional.cross_entropy(forward(target, new_labels).flatten(0, 1), new_labels.flatten())
    loss.backward()
    assert torch.isfinite(loss)
    for param in (target.decoder.prediction["embed"].weight, target.joint.joint_net[-1].weight):
        assert torch.isfinite(param.grad[added]).all() and param.grad[added].abs().sum() > 0


def test_removed_output_can_change_predictions_despite_exact_retained_logits():
    torch = pytest.importorskip("torch")
    source, target = toy_model(4), toy_model(2)
    old, new = inspect_nemo_layout(source), inspect_nemo_layout(target)
    mapping = (0, None, 1, None, 2)
    with torch.no_grad():
        source.joint.joint_net[-1].weight.zero_()
        source.joint.joint_net[-1].bias.zero_()
        source.joint.joint_net[-1].bias[1] = 100
    target.load_state_dict(transfer_native_state_dict(source.state_dict(), target.state_dict(), old, new, mapping))
    features = torch.ones(1, 4)
    before, after = source.joint.joint_net[-1](features), target.joint.joint_net[-1](features)
    assert before.argmax(-1).item() == 1 and mapping[1] is None
    assert compare_retained_logits(before, after, old, new, mapping)["passed"]
    assert after.argmax(-1).item() == 0


def test_logit_comparison_excludes_only_explicit_inactive_target_rows():
    torch = pytest.importorskip("torch")
    old, new = inspect_nemo_layout(toy_model(4)), inspect_nemo_layout(toy_model(4))
    left = torch.tensor([[1., 2., 3., 4., 5.]])
    right = left.clone()
    right[:, [1, 3]] = -torch.inf
    mapping = (0, 1, 2, 3, 4)
    with pytest.raises(ValueError, match="non-finite"):
        compare_retained_logits(left, right, old, new, mapping)
    report = compare_retained_logits(left, right, old, new, mapping, inactive_target_ids=(1, 3))
    assert report["passed"] and report["inactive_target_outputs_excluded"] == [1, 3]
    right[:, 2] += 1
    with pytest.raises(ValueError, match="logits changed"):
        compare_retained_logits(left, right, old, new, mapping, inactive_target_ids=(1, 3))
    for invalid in ((4,), (1, 1), (True,), (-1,)):
        with pytest.raises(ValueError, match="Inactive logit IDs"):
            compare_retained_logits(left, right, old, new, mapping, inactive_target_ids=invalid)


@pytest.mark.parametrize("mapping", [
    (0, 1, 2, 3), (0, 1, 2, 3, None), (0, 1, 2, 3, 4),
    (0, 1, 1, 3, 6), (0, True, 2, 3, 6), (0, 1.0, 2, 3, 6),
    (0, -1, 2, 3, 6), (0, 7, 2, 3, 6), (None, None, None, None, 6),
])
def test_invalid_or_ambiguous_subset_mapping_is_rejected(mapping):
    old, new = inspect_nemo_layout(toy_model(4)), inspect_nemo_layout(toy_model(6))
    with pytest.raises(ValueError):
        retained_row_pairs(old, new, mapping)


@pytest.mark.parametrize("tensor", ["encoder.weight", "prompt_kernel.weight", "decoder.prediction.embed.weight"])
def test_subset_verifier_detects_tampering_with_retained_state(tensor):
    source, target = toy_model(4), toy_model(5)
    old, new = inspect_nemo_layout(source), inspect_nemo_layout(target)
    mapping = (0, None, 1, 2, 5)
    state = transfer_native_state_dict(source.state_dict(), target.state_dict(), old, new, mapping)
    state[tensor][0, 0] += 1
    with pytest.raises(ValueError, match=tensor.replace(".", r"\.")):
        verify_native_state_transfer(source.state_dict(), state, old, new, mapping)


def test_subset_transfer_rejects_dtypes_shapes_and_missing_keys():
    source, target = toy_model(4), toy_model(5)
    old, new = inspect_nemo_layout(source), inspect_nemo_layout(target)
    mapping = (0, None, 1, 2, 5)
    for kind in ("dtype", "shape", "keys"):
        state = copy.deepcopy(target.state_dict())
        if kind == "dtype": state["encoder.weight"] = state["encoder.weight"].double()
        elif kind == "shape": state["encoder.weight"] = state["encoder.weight"][:1]
        else: del state["encoder.weight"]
        with pytest.raises(ValueError):
            transfer_native_state_dict(source.state_dict(), state, old, new, mapping)


def test_source_checks_actual_sentencepiece_bytes_and_wrapper_encoding():
    model = pb.ModelProto()
    model.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    model.trainer_spec.unk_id = 0
    model.trainer_spec.bos_id = model.trainer_spec.eos_id = model.trainer_spec.pad_id = -1
    model.normalizer_spec.name = "identity"
    for piece, kind in [("<unk>", 2), ("▁", 1), ("a", 1)]:
        model.pieces.add(piece=piece, score=-1, type=kind)
    model.trainer_spec.vocab_size = len(model.pieces)
    data = model.SerializeToString()
    backend = spm.SentencePieceProcessor(model_proto=data)
    tokenizer = SimpleNamespace(tokenizer=backend, vocab_size=3,
        ids_to_tokens=lambda ids: [backend.id_to_piece(i) for i in ids],
        text_to_ids=lambda text: backend.encode(text, out_type=int))
    source = SimpleNamespace(tokenizer=tokenizer)
    assert validate_source_native_tokenizer(source, data)["native_tokenizer_sha256"] == hashlib.sha256(data).hexdigest()
    model.pieces[2].score = -2
    with pytest.raises(ValueError, match="pinned native base"):
        validate_source_native_tokenizer(source, model.SerializeToString())
    tokenizer.text_to_ids = lambda text: [2]
    with pytest.raises(ValueError, match="wrapper encoding"):
        validate_source_native_tokenizer(source, data)


def test_prompt_registry_keeps_old_slots_and_requires_all_22_targets():
    defaults = SimpleNamespace(num_prompts=64, prompt_dictionary={"auto": 0, "en-US": 1, "hi-IN": 2})
    registry = _prompt_registry(defaults)
    assert len(registry["target_assignments"]) == 22
    assert registry["prompt_dictionary"]["hi-IN"] == 2
    assert registry["prompt_dictionary"]["en-US"] == 1
    assert _prompt_registry(defaults, registry) == registry
    bad = copy.deepcopy(registry)
    bad["prompt_dictionary"]["en-US"] = 3
    with pytest.raises(ValueError, match="upstream prompt assignment"):
        _prompt_registry(defaults, bad)
    wrong = SimpleNamespace(num_prompts=64, prompt_dictionary={"auto": 0, "en-US": 1, "hi-IN": 3})
    with pytest.raises(ValueError, match="different pinned processor"):
        _prompt_registry(wrong, registry)


def test_profile_prompt_registries_filter_names_without_reusing_removed_slots():
    defaults = SimpleNamespace(num_prompts=64, prompt_dictionary={
        "auto": 0, "en-US": 1, "en": 1, "hi-IN": 2, "hi": 2,
        "ar-AR": 3, "ja-JP": 4, "or-KE": 5, "mt-MT": 6, "sl-SI": 7,
    })
    original = _prompt_registry(defaults, profile="original")
    assert original["prompt_dictionary"] == defaults.prompt_dictionary
    assert not original["target_assignments"]
    latin = _prompt_registry(defaults, profile="latin")
    assert latin["prompt_dictionary"] == {"auto": 0, "en-US": 1, "en": 1, "mt-MT": 6, "sl-SI": 7}
    assert not latin["allocated_this_build"]
    assert latin["reserved_source_prompt_slots"] == list(range(8))
    assert not set(range(8)) & set(latin["unused_prompt_slots"])
    indic = _prompt_registry(defaults, profile="latin-indic")
    assert len(indic["target_assignments"]) == 22
    assert not {"ar-AR", "ja-JP", "or-KE"} & set(indic["prompt_dictionary"])
    assert indic["prompt_dictionary"]["hi"] == indic["prompt_dictionary"]["hi-IN"] == 2
    assert not set(indic["allocated_this_build"].values()) & set(defaults.prompt_dictionary.values())
    for profile, registry in (("original", original), ("latin", latin), ("latin-indic", indic)):
        assert _prompt_registry(defaults, registry, profile=profile)["prompt_dictionary"] == registry["prompt_dictionary"]
        changed = copy.deepcopy(registry)
        changed["prompt_dictionary"]["en-US"] = 9
        with pytest.raises(ValueError, match="slots|upstream prompt assignment"):
            _prompt_registry(defaults, changed, profile=profile)
        if profile != "original":
            changed = copy.deepcopy(registry)
            changed["prompt_dictionary"]["ja-JP"] = 4
            with pytest.raises(ValueError, match="profile|target identities"):
                _prompt_registry(defaults, changed, profile=profile)


def test_indic_profile_accepts_validated_full_registry_and_rejects_missing_target():
    defaults = SimpleNamespace(num_prompts=64, prompt_dictionary={"auto": 0, "en-US": 1, "ar-AR": 2})
    full = _prompt_registry(defaults)
    indic = _prompt_registry(defaults, full, profile="latin-indic")
    assert "ar-AR" not in indic["prompt_dictionary"]
    assert indic["prompt_dictionary"]["ml-IN"] == full["prompt_dictionary"]["ml-IN"]
    altered = copy.deepcopy(indic)
    altered["prompt_dictionary"].pop("ml-IN")
    with pytest.raises(ValueError, match="identity assignments"):
        _prompt_registry(defaults, altered, profile="latin-indic")


@pytest.fixture(scope="module")
def clean_checkpoint_bundles(tmp_path_factory):
    """Use the real profile recipe while keeping acoustic weights wholly synthetic."""
    pytest.importorskip("torch")
    from untok.bundles import PROFILES, load_tokenizer_bundle
    from untok.clean import build_clean_bundles

    data = Path(__file__).resolve().parents[1] / "src" / "untok" / "data"
    output = tmp_path_factory.mktemp("clean-checkpoint-bundles")
    build_clean_bundles(data / "source", output)
    return {profile: load_tokenizer_bundle(output / profile) for profile in PROFILES}


@pytest.mark.parametrize("profile", ["original", "full", "latin-indic", "latin"])
@pytest.mark.parametrize("source_inventory", ["original_native_base", "original_untok_full_v1"])
def test_real_clean_mapping_transfers_synthetic_checkpoint_and_relocates_blank(
    clean_checkpoint_bundles, tmp_path, profile, source_inventory,
):
    torch = pytest.importorskip("torch")
    adapter = clean_checkpoint_bundles[profile]
    is_base = source_inventory == "original_native_base"
    source_bytes = adapter.base_model_bytes if is_base else adapter.full_model_bytes
    source_processor = spm.SentencePieceProcessor(model_proto=source_bytes)
    mapping = adapter.source_native_to_target_native if is_base else adapter.full_native_to_subset_native
    torch.manual_seed(823)
    source, target = toy_model(source_processor.get_piece_size()), toy_model(adapter.vocab_size)
    source.tokenizer = SimpleNamespace(
        backend=source_processor, vocab_size=source_processor.get_piece_size(),
        ids_to_tokens=lambda ids: [source_processor.id_to_piece(index) for index in ids],
        text_to_ids=lambda text: source_processor.encode(text, out_type=int),
    )
    selected_inventory, selected_mapping, source_check = select_source_native_inventory(source, adapter)
    assert selected_inventory == source_inventory and selected_mapping == mapping
    assert source_check["native_tokenizer_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    old, new = inspect_nemo_layout(source), inspect_nemo_layout(target)
    retained = [index for index, dest in enumerate(mapping) if dest is not None]
    removed = [index for index, dest in enumerate(mapping) if dest is None]
    assert (old.blank_id == new.blank_id) == (is_base and profile in {"original", "latin"})
    if is_base:
        assert not removed
        assert mapping[:-1] == tuple(range(old.blank_id))
    else:
        assert removed
    assert mapping[old.blank_id] == new.blank_id

    # Mark removed rows with conspicuous values so an initializer accidentally
    # averaging them cannot pass. Every retained text row remains distinguishable.
    with torch.no_grad():
        embedding = source.decoder.prediction["embed"].weight
        embedding.copy_(torch.arange(embedding.numel()).reshape_as(embedding) / embedding.numel())
        embedding[removed] = 10_000
        embedding[old.blank_id] = 0
        source.joint.joint_net[-1].weight[removed] = 20_000
        source.joint.joint_net[-1].bias[removed] = 30_000
    original = copy.deepcopy(source.state_dict())
    transferred = transfer_native_state_dict(original, target.state_dict(), old, new, mapping)
    initial, policy = initialize_text_donor_rows(transferred, new, adapter, mapping)
    additions = policy["new_model_rows"]
    assert set(additions) == set(range(new.output_size)) - {mapping[index] for index in retained}
    # ID 13087 was the original blank but denotes '#' in the v1 extension:
    # migration from the base must initialize that text row, not copy blank to it.
    if profile in {"full", "latin-indic"}:
        assert (adapter.token_to_id("#") in additions) == is_base
        assert adapter.token_to_id("#") != new.blank_id
    if profile in {"original", "latin"} or not is_base:
        assert not additions and policy["policy"] == "no_added_rows"
    if additions:
        assert policy["policy"] == "retained_text_donor_mean_bound_v1"
        assert policy["max_new_mass_ratio"] == .05
        assert torch.all(initial[new.embedding_key][additions] > 0)
        for key in new.row_keys:
            assert torch.isfinite(initial[key][additions]).all()
            assert initial[key][additions].abs().max() < 1000
        assert not torch.equal(initial[new.output_weight_key][additions],
                               original[old.output_weight_key][old.blank_id].expand(len(additions), -1))
    expected_new = {key: initial[key][additions].clone() for key in new.row_keys}

    target.load_state_dict(initial)
    from untok.native_runtime import install_native_output_mask

    target.tokenizer = adapter
    mask = install_native_output_mask(target)
    assert mask["inactive_output_rows"] == len(adapter.inactive_native_ids)
    migrated = target.state_dict()
    report = verify_native_state_transfer(original, migrated, old, new, mapping)
    assert report["removed_source_text_rows"] == len(removed)
    assert report["learned_values_omitted"] == sum(original[key][removed].numel() for key in old.row_keys)
    assert report["all_source_values_preserved"] == (not removed)
    for key in old.row_keys:
        assert torch.equal(migrated[key][[mapping[index] for index in retained]], original[key][retained])
        assert torch.equal(migrated[key][new.blank_id], original[key][old.blank_id])
        assert torch.equal(migrated[key][additions], expected_new[key])
    for key in set(original) - set(old.row_keys):
        assert torch.equal(migrated[key], original[key])
    assert all(torch.equal(original[key], value) for key, value in source.state_dict().items())

    # Dormant source Parameters survive, but their outputs are deliberately
    # masked and cannot be compared as active logits or used as text labels.
    active_retained = [index for index in retained[:-1] if mapping[index] not in adapter.inactive_native_ids]
    old_prefix = torch.tensor([[old.blank_id, *active_retained[:3], active_retained[2]]])
    new_prefix = torch.tensor([[mapping[index] for index in old_prefix[0].tolist()]])
    features = torch.randn(1, old_prefix.shape[1], 4)

    def forward(model, prefix):
        predicted, _ = model.decoder.prediction["rnn"](model.decoder.prediction["embed"](prefix))
        return model.joint.joint_net(model.encoder(features) + predicted)

    logits = forward(target, new_prefix)
    assert compare_retained_logits(forward(source, old_prefix), logits, old, new, mapping,
                                   inactive_target_ids=adapter.inactive_native_ids)["passed"]
    if additions:
        active_targets = [mapping[index] for index in active_retained] + [new.blank_id]
        ratio = logits[..., additions].double().logsumexp(-1) - logits[..., active_targets].double().logsumexp(-1)
        assert float(ratio.detach().exp().max()) <= .05 + 1e-7

    # A tensor state round trip checks these rows survive serialization. This is
    # deliberately not a NeMo .nemo restoration or an acoustic-quality test.
    saved = tmp_path / "synthetic-clean-state.pt"
    torch.save(migrated, saved)
    restored = torch.load(saved, weights_only=True)
    assert verify_native_state_transfer(original, restored, old, new, mapping)["passed"]
    assert all(torch.equal(restored[key][additions], expected_new[key]) for key in new.row_keys)


def test_clean_source_inventory_selection_rejects_wrong_bytes_and_encoding(clean_checkpoint_bundles):
    adapter = clean_checkpoint_bundles["full"]

    def source_for(raw):
        processor = spm.SentencePieceProcessor(model_proto=raw)
        return SimpleNamespace(tokenizer=SimpleNamespace(
            backend=processor, vocab_size=processor.get_piece_size(),
            ids_to_tokens=lambda ids: [processor.id_to_piece(index) for index in ids],
            text_to_ids=lambda text: processor.encode(text, out_type=int),
        ))

    altered = pb.ModelProto()
    altered.ParseFromString(adapter.full_model_bytes)
    altered.pieces[2].score -= 0.25
    for raw in (altered.SerializeToString(), adapter.model_bytes):
        with pytest.raises(ValueError, match="not a pinned"):
            select_source_native_inventory(source_for(raw), adapter)
    source = source_for(adapter.full_model_bytes)
    source.tokenizer.text_to_ids = lambda text: [2]
    with pytest.raises(ValueError, match="wrapper encoding"):
        select_source_native_inventory(source, adapter)
    source = source_for(adapter.base_model_bytes)
    source.tokenizer.vocab_size += 1
    with pytest.raises(ValueError, match="vocabulary size"):
        select_source_native_inventory(source, adapter)
