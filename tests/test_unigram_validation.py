import hashlib
from importlib import resources
import json
from pathlib import Path

import pytest
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.cli import main
from untok.unigram import build_native_tokenizer
from untok.unigram_validation import validate_native_tokenizer


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def candidate(tmp_path):
    model = pb.ModelProto()
    model.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    model.trainer_spec.unk_id = 0
    model.trainer_spec.bos_id = model.trainer_spec.eos_id = model.trainer_spec.pad_id = -1
    model.normalizer_spec.name = "identity"
    model.normalizer_spec.remove_extra_whitespaces = False
    for piece, score, kind in [("<unk>", 0, 2), ("▁", -1, 1), ("a", -2, 1), ("b", -2, 1), ("z", -8, 1)]:
        model.pieces.add(piece=piece, score=score, type=kind)
    model.trainer_spec.vocab_size = len(model.pieces)
    base = tmp_path / "base.model"
    base.write_bytes(model.SerializeToString())
    selection = tmp_path / "selection.json"
    write(selection, {"base_tokenizer_sha256": sha(base), "additions": [{"piece": "ab", "score": -1}]})
    bundle = tmp_path / "bundle"
    info = build_native_tokenizer(base, selection, bundle)
    policy = tmp_path / "policy.json"
    write(policy, {"schema_version": 1, "base_tokenizer_sha256": sha(base),
                   "normalizer_sha256": info["normalizer_sha256"],
                   "profiles": {"x": {"script": "Latn", "characters": ["a", "b"]}},
                   "protected_piece_groups": {"new": {"pieces": ["ab"]}},
                   "normalizer_probes": ["", "a", "  a  b  ", "\ta\nb", "ﬁ", "क्\u200cष"],
                   "exact_encoding_probes": ["a"]})
    data = tmp_path / "data" / "manifest.json"
    for phase in ("train", "dev", "reserve"):
        path = data.parent / phase / "x.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('\n'.join(json.dumps({"text": text, "source": "toy", "record_id": str(i)}, ensure_ascii=False)
                                  for i, text in enumerate(["ab", "  a  b  ", "xy"])) + '\n')
    write(data, {"native_base_sha256": sha(base), "normalizer_sha256": info["normalizer_sha256"],
                 "files": {f"{phase}/x.jsonl": sha(data.parent / phase / "x.jsonl") for phase in ("train", "dev", "reserve")}})
    return bundle, policy, data


def receipt(candidate):
    bundle, policy, data = candidate
    path = data.parent / "receipt.json"
    write(path, {"tokenizer_sha256": sha(bundle / "tokenizer.model"),
                 "data_manifest_sha256": sha(data), "bundle_manifest_sha256": sha(bundle / "manifest.json"),
                 "selection_sha256": sha(bundle / "selection.json"), "policy_sha256": sha(policy)})
    return path


def change(path, mutate):
    obj = json.loads(path.read_text()); mutate(obj); write(path, obj)


def test_portable_validation_and_known_segmentation_changes(candidate, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = validate_native_tokenizer(*candidate, max_examples=1)
    assert result["passed"] and result["status"] == "passed"
    assert result["structural_passed"] and result["corpus_status"] == "passed"
    assert result["selection_quality_status"] == result["training_usage_status"] == "passed"
    assert result["corpora"]["x"]["base_representable_changed"] == 1
    assert result["corpora"]["x"]["unknown_records"] == 1
    assert result["corpora"]["x"]["unknown_codepoint_occurrences"] == 2
    assert result["corpora"]["x"]["roundtrip_failures"] == 0
    assert result["protected_piece_groups"]["new"]["passed"]
    assert result["no_added_match_probes"] > 0
    assert result["old_random_sequence_decode_trials"] == 10000
    assert not result["checkpoint_validated"] and not result["asr_validated"]


def test_missing_or_empty_corpora_are_incomplete(candidate):
    bundle, policy, data = candidate
    assert validate_native_tokenizer(bundle, policy)["status"] == "incomplete"
    change(data, lambda obj: obj["files"].pop("dev/x.jsonl"))
    result = validate_native_tokenizer(*candidate)
    assert result["status"] == "incomplete" and not result["passed"]
    assert result["missing_profiles"] == ["x"]
    empty = data.parent / "dev/x.jsonl"
    empty.write_text("")
    change(data, lambda obj: obj["files"].update({"dev/x.jsonl": sha(empty)}))
    result = validate_native_tokenizer(*candidate)
    assert result["status"] == "incomplete" and result["empty_profiles"] == ["x"]


@pytest.mark.parametrize("target,field", [(1, "base_tokenizer_sha256"), (1, "normalizer_sha256"),
                                          (2, "native_base_sha256"), (2, "normalizer_sha256")])
def test_wrong_base_and_normalizer_bindings(candidate, target, field):
    change(candidate[target], lambda obj: obj.update({field: "bad"}))
    with pytest.raises(ValueError, match="hash"):
        validate_native_tokenizer(*candidate)


def test_phase_hash_and_escape_checks(candidate):
    _, _, data = candidate
    change(data, lambda obj: obj["files"].update({"dev/x.jsonl": "bad"}))
    with pytest.raises(ValueError, match="integrity"):
        validate_native_tokenizer(*candidate)
    change(data, lambda obj: obj["files"].update({"../outside": "bad"}))
    with pytest.raises(ValueError, match="contained"):
        validate_native_tokenizer(*candidate)


def test_dev_does_not_read_or_hash_reserve(candidate, monkeypatch):
    reserve = candidate[2].parent / "reserve/x.jsonl"
    original = Path.open
    def guard(self, *args, **kwargs):
        assert self != reserve, "Dev validation opened sealed reserve"
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    assert validate_native_tokenizer(*candidate)["passed"]


def test_reserve_requires_receipt_before_any_corpus_read(candidate, monkeypatch):
    original = Path.open
    def guard(self, *args, **kwargs):
        assert self.suffix != ".jsonl", "Read corpus before rejecting receipt"
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    with pytest.raises(ValueError, match="requires a selection receipt"):
        validate_native_tokenizer(*candidate, phase="reserve")
    rec = receipt(candidate)
    change(rec, lambda obj: obj.update({"policy_sha256": "bad"}))
    with pytest.raises(ValueError, match="policy_sha256"):
        validate_native_tokenizer(*candidate, phase="reserve", selection_receipt_path=rec)


@pytest.mark.parametrize("field", ["tokenizer_sha256", "data_manifest_sha256", "bundle_manifest_sha256", "selection_sha256", "policy_sha256"])
def test_every_reserve_binding_required(candidate, field):
    rec = receipt(candidate)
    change(rec, lambda obj: obj.pop(field))
    with pytest.raises(ValueError, match=field):
        validate_native_tokenizer(*candidate, phase="reserve", selection_receipt_path=rec)


def test_reserve_omits_examples_even_if_requested(candidate):
    result = validate_native_tokenizer(*candidate, phase="reserve", selection_receipt_path=receipt(candidate), max_examples=5)
    assert result["passed"]
    assert not result["corpora"]["x"]["base_representable_change_examples"]
    assert not result["corpora"]["x"]["roundtrip_failure_examples"]


def test_missing_protected_piece_or_coverage_is_failure(candidate):
    policy = candidate[1]
    change(policy, lambda obj: obj["protected_piece_groups"]["new"].update({"pieces": ["missing"]}))
    result = validate_native_tokenizer(*candidate)
    assert result["status"] == "failed" and not result["structural_passed"]
    assert result["protected_piece_groups"]["new"]["missing"] == ["missing"]
    change(policy, lambda obj: obj["profiles"]["x"]["characters"].append("क"))
    assert validate_native_tokenizer(*candidate)["alphabet_coverage"]["x"]["missing"] == ["क"]


def test_roundtrip_expected_text_failure_not_hidden(candidate):
    data = candidate[2]
    text = data.parent / "dev/x.jsonl"
    text.write_text(json.dumps({"text": "ab", "expected_text": "wrong"}) + '\n')
    change(data, lambda obj: obj["files"].update({"dev/x.jsonl": sha(text)}))
    result = validate_native_tokenizer(*candidate)
    assert result["status"] == "failed"
    assert result["corpora"]["x"]["roundtrip_failures"] == 1
    assert result["corpora"]["x"]["roundtrip_failure_examples"] == []


def test_malformed_record_fails_clearly(candidate):
    data = candidate[2]
    text = data.parent / "dev/x.jsonl"
    text.write_text('{"text": 42}\n')
    change(data, lambda obj: obj["files"].update({"dev/x.jsonl": sha(text)}))
    with pytest.raises(ValueError, match="text string"):
        validate_native_tokenizer(*candidate)


def test_cli_writes_report_and_incomplete_is_nonzero(candidate, tmp_path, capsys):
    bundle, policy, data = candidate
    out = tmp_path / "report.json"
    args = ["validate-unigram", "--bundle", str(bundle), "--policy", str(policy), "--output", str(out)]
    assert main(args) == 2
    assert json.loads(out.read_text())["status"] == "incomplete"
    assert main(args + ["--corpora", str(data)]) == 0
    assert json.loads(out.read_text())["passed"]


@pytest.mark.parametrize("add_prefix", [False, True])
@pytest.mark.parametrize("as_suffix", [False, True])
@pytest.mark.parametrize("escape", [False, True])
def test_expected_text_matches_backend_whitespace_contract(add_prefix, as_suffix, escape):
    import sentencepiece as spm
    from untok.unigram_validation import _expected
    model = pb.ModelProto()
    model.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    model.trainer_spec.unk_id = 0
    model.trainer_spec.bos_id = model.trainer_spec.eos_id = model.trainer_spec.pad_id = -1
    model.trainer_spec.treat_whitespace_as_suffix = as_suffix
    model.normalizer_spec.name = "identity"
    model.normalizer_spec.remove_extra_whitespaces = False
    model.normalizer_spec.add_dummy_prefix = add_prefix
    model.normalizer_spec.escape_whitespaces = escape
    for piece, kind in [("<unk>", 2), ("▁", 1), (" ", 1), ("a", 1)]:
        model.pieces.add(piece=piece, score=-1, type=kind)
    model.trainer_spec.vocab_size = len(model.pieces)
    proc = spm.SentencePieceProcessor(model_proto=model.SerializeToString())
    for text in ("", " ", "a", " a", "a ", "  a  ", "▁a"):
        assert _expected(proc, model, text) == proc.decode(proc.encode(text))


def test_hash_valid_wrong_language_file_rejected(candidate):
    data = candidate[2]
    text = data.parent / "dev/x.jsonl"
    text.write_text(json.dumps({"text": "ab", "language": "wrong"}) + '\n')
    change(data, lambda obj: obj["files"].update({"dev/x.jsonl": sha(text)}))
    with pytest.raises(ValueError, match="language disagrees with profile x"):
        validate_native_tokenizer(*candidate)


def rebuild(candidate, tmp_path, mutate):
    bundle, policy, data = candidate
    selection = tmp_path / "changed-selection.json"
    obj = json.loads((bundle / "selection.json").read_text())
    mutate(obj)
    write(selection, obj)
    output = tmp_path / "changed-bundle"
    build_native_tokenizer(bundle / "base-tokenizer.model", selection, output)
    return output, policy, data


def test_unused_optional_piece_rejects_fabricated_training_count(candidate, tmp_path):
    revised = rebuild(candidate, tmp_path, lambda obj: obj["additions"].append(
        {"piece": "az", "score": -1, "training_occurrences": 999}))
    result = validate_native_tokenizer(*revised)
    assert result["structural_passed"] and result["corpus_status"] == "passed"
    assert result["selection_quality_status"] == result["training_usage_status"] == "failed"
    assert result["training_usage"]["optional_unused"] == ["az"]
    assert result["training_usage"]["declared_count_mismatches"] == ["az"]


def test_all_required_pieces_need_fresh_witnesses_even_outside_protected_groups(candidate, tmp_path):
    revised = rebuild(candidate, tmp_path, lambda obj: obj["additions"].append(
        {"piece": "aa", "score": -8, "required_reasons": [{"kind": "alphabet"}],
         "required_encoding_witness": {"input": "aa", "native_id": 6}}))
    result = validate_native_tokenizer(*revised)
    assert result["protected_piece_groups"]["new"]["passed"]
    assert result["selection_quality"]["required_unwitnessed"] == ["aa"]
    assert result["selection_quality"]["strictly_dominated_additions"][0]["best_split"] == ["a", "a"]
    assert result["selection_quality_status"] == "failed"


def test_required_coverage_piece_may_be_unused_in_training(candidate, tmp_path):
    revised = rebuild(candidate, tmp_path, lambda obj: obj["additions"].append(
        {"piece": "az", "score": -1, "required_reasons": [{"kind": "alphabet"}]}))
    result = validate_native_tokenizer(*revised)
    assert result["passed"]
    assert result["selection_quality"]["required_piece_witnesses"]["az"] is not None
    assert result["training_usage"]["required_unused"] == ["az"]


def test_missing_training_keeps_structure_and_corpus_status_separate(candidate):
    change(candidate[2], lambda obj: obj["files"].pop("train/x.jsonl"))
    result = validate_native_tokenizer(*candidate)
    assert result["structural_passed"] and result["corpus_status"] == "passed"
    assert result["training_usage_status"] == result["selection_quality_status"] == "incomplete"
    assert result["selection_quality_gates"]["optional_training_usage"] is None
    assert result["status"] == "incomplete" and not result["passed"]


def test_training_manifest_cannot_replace_selection_pinned_manifest(candidate, tmp_path, monkeypatch):
    revised = rebuild(candidate, tmp_path, lambda obj: obj.update({"data_manifest_sha256": "wrong"}))
    original = Path.open
    def guard(self, *args, **kwargs):
        assert self.parent.name != "train", "Read unrelated training corpus"
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    result = validate_native_tokenizer(*revised)
    assert result["corpus_status"] == "passed"
    assert result["selection_quality_status"] == "incomplete"
    assert not result["training_usage"]["selection_manifest_matches"]


def test_training_file_hash_is_verified_before_encoding(candidate):
    change(candidate[2], lambda obj: obj["files"].update({"train/x.jsonl": "wrong"}))
    with pytest.raises(ValueError, match="Training corpus integrity"):
        validate_native_tokenizer(*candidate)


@pytest.mark.parametrize("limits", [
    {"max_unknown_records": 0}, {"max_unknown_tokens": 0},
    {"max_unknown_record_rate": 0}, {"max_unknown_token_rate": 0},
    {"max_mean_tokens_per_normalized_character": .01}, {"max_tokens_p95": 1}, {"max_tokens_p99": 1},
])
def test_explicit_unknown_and_efficiency_thresholds_are_release_gates(candidate, limits):
    change(candidate[1], lambda obj: obj.update({"corpus_thresholds": {"dev": {"default": limits}}}))
    result = validate_native_tokenizer(*candidate)
    assert result["corpus_status"] == "failed" and not result["passed"]
    assert not result["corpus_gates"]["configured_release_thresholds"]
    assert result["selection_quality_status"] == "passed"


def test_per_profile_thresholds_override_defaults_at_inclusive_boundary(candidate):
    change(candidate[1], lambda obj: obj.update({"corpus_thresholds": {"dev": {
        "default": {"max_unknown_records": 0}, "profiles": {"x": {"max_unknown_records": 1}}}}}))
    assert validate_native_tokenizer(*candidate)["passed"]


@pytest.mark.parametrize("limit", [-1, True, float("inf"), "0"])
def test_invalid_thresholds_rejected(candidate, limit):
    change(candidate[1], lambda obj: obj.update({"corpus_thresholds": {"dev": {
        "default": {"max_unknown_records": limit}}}}))
    with pytest.raises(ValueError, match="corpus threshold"):
        validate_native_tokenizer(*candidate)


def test_boundary_marker_is_not_a_normalization_alias(candidate, tmp_path):
    revised = rebuild(candidate, tmp_path, lambda obj: obj["additions"].append(
        {"piece": "▁ab", "score": -1, "required_reasons": [{"kind": "boundary"}]}))
    result = validate_native_tokenizer(*revised)
    assert not result["selection_quality"]["normalization_aliases"]
    assert not result["selection_quality"]["normalization_failures"]


def test_equal_score_split_is_not_strict_dominance(candidate, tmp_path):
    revised = rebuild(candidate, tmp_path, lambda obj: obj["additions"].append(
        {"piece": "aa", "score": -4, "required_reasons": [{"kind": "coverage"}]}))
    result = validate_native_tokenizer(*revised)
    assert not result["selection_quality"]["strictly_dominated_additions"]


@pytest.mark.parametrize("piece,normalized,alias", [("क़", "क़", False), ("ｆｉ", "fi", True)])
def test_normalization_dead_addition_is_rejected(tmp_path, piece, normalized, alias):
    """The exact क़ release-audit reproduction must fail without needing a corpus."""
    root = Path(__file__).resolve().parents[1]
    original = resources.files("untok").joinpath("data", "source")
    selection = json.loads((original / "selection.json").read_text())
    selection["additions"].append({"piece": piece, "score": -10, "training_occurrences": 123})
    selected = tmp_path / "selection.json"
    write(selected, selection)
    bundle = tmp_path / "bundle"
    build_native_tokenizer(original / "base-tokenizer.model", selected, bundle)
    policy = json.loads((root / "configs/native-unigram-validation.json").read_text())
    if alias:
        policy["approved_new_latin_pieces"].append(piece)
    policy_path = tmp_path / "policy.json"
    write(policy_path, policy)
    result = validate_native_tokenizer(bundle, policy_path)
    assert result["structural_passed"]
    assert result["selection_quality_status"] == result["status"] == "failed"
    assert result["selection_quality"]["normalization_failures"] == [
        {"native_id": 20550, "piece": piece, "normalized": normalized}]
    assert bool(result["selection_quality"]["normalization_aliases"]) == alias


def test_shipped_additions_pass_all_static_quality_checks_without_training_claims():
    root = Path(__file__).resolve().parents[1]
    result = validate_native_tokenizer(resources.files("untok").joinpath("data", "source"), root / "configs/native-unigram-validation.json")
    assert result["selection_quality"]["static_passed"]
    assert result["selection_quality"]["new_piece_count"] == 7463
    assert result["selection_quality"]["required_piece_count"] == 1183
    assert result["training_usage_status"] == "incomplete"
    assert result["selection_quality_status"] == "incomplete"
