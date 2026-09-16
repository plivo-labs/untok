import hashlib
from importlib import resources
import json
from pathlib import Path

import pytest

from untok.clean import ALGORITHM, build_clean_bundles
from untok.clean_validation import validate_clean_tokenizer
from untok.unigram_validation import validate_native_tokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="module")
def clean_bundle(tmp_path_factory):
    output = tmp_path_factory.mktemp("clean-validation") / "bundles"
    source = resources.files("untok").joinpath("data", "source")
    build_clean_bundles(source, output)
    return output / "full"


@pytest.fixture
def candidate(clean_bundle, tmp_path):
    from sentencepiece import sentencepiece_model_pb2 as pb

    manifest = json.loads((clean_bundle / "manifest.json").read_text())
    policy = tmp_path / "policy.json"
    write(policy, {"schema_version": 1, "base_tokenizer_sha256": manifest["base_tokenizer_sha256"],
                   "tokenizer_sha256": manifest["tokenizer_sha256"], "normalizer_sha256": manifest["normalizer_sha256"],
                   "profiles": {"ml": {"script": "Malayalam", "characters": ["അ", "വ", "ർ"]}},
                   "normalizer_probes": ["അവർ", "അവര്‍", "क्\u200cष"],
                   "corpus_thresholds": {"dev": {"default": {"max_unknown_records": 0}}}})
    model = pb.ModelProto()
    model.ParseFromString((clean_bundle / "base-tokenizer.model").read_bytes())
    source_normalizer = hashlib.sha256(model.normalizer_spec.SerializeToString()).hexdigest()
    data = tmp_path / "data" / "manifest.json"
    files = {}
    for phase in ("dev", "reserve"):
        path = data.parent / phase / "ml.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps({"text": text, "language": "ml"}, ensure_ascii=False) + "\n"
                                for text in ["അവർ", "അവര്‍", "क्\u200cष"]), encoding="utf-8")
        files[f"{phase}/ml.jsonl"] = sha(path)
    write(data, {"native_base_sha256": manifest["base_tokenizer_sha256"],
                 "normalizer_sha256": source_normalizer, "files": files})
    return clean_bundle, policy, data


def receipt(candidate):
    bundle, policy, data = candidate
    source = resources.files("untok").joinpath("data", "source", "selection.json")
    result = data.parent / "receipt.json"
    write(result, {"tokenizer_sha256": sha(bundle / "tokenizer.model"),
                   "data_manifest_sha256": sha(data), "bundle_manifest_sha256": sha(bundle / "manifest.json"),
                   "selection_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "policy_sha256": sha(policy)})
    return result


def test_dispatch_preserves_original_native_identity_and_scopes_quality_to_additions(candidate):
    result = validate_native_tokenizer(*candidate)
    assert result["algorithm"] == ALGORITHM and result["tokenizer_version"] == 3
    assert result["structural_passed"] and result["inventory_quality"]["passed"]
    assert result["corpus_status"] == "passed"
    assert result["corpora"]["ml"]["roundtrip_failures"] == 0
    assert result["corpora"]["ml"]["normalizer_failures"] == 0
    assert result["corpora"]["ml"]["source_normalizer_changes"] == 0
    assert result["old_ID_compatibility"] is True
    assert result["original_text_id_range"] == [0, 13086]
    assert result["public_pad_id"] == 13087
    assert result["public_blank_id"] == 13088
    assert result["gates"]["all_original_piece_messages_preserved"]
    assert result["gates"]["all_original_model_metadata_preserved"]
    assert result["gates"]["original_normalizer_unchanged"]
    assert not result["inherited_inventory_quality"]["passed"]
    assert len(result["inherited_inventory_quality"]["normalization_failures"]) == 4
    assert len(result["original_base_inventory_quality"]["strictly_dominated_pieces"]) == 249
    assert result["inventory_quality"]["checked_piece_count"] == result["native_vocabulary_size"] - 13087
    assert result["requires_retokenized_training_labels"] is True
    assert result["selection_quality_status"] == result["training_usage_status"] == "incomplete"
    assert result["status"] == "incomplete" and not result["passed"]
    assert not result["selection_quality"]["required_unwitnessed"]
    assert not result["selection_quality"]["uncovered_required_text"]
    assert "\u200c" not in result["selection_quality"]["required_piece_witnesses"]


@pytest.mark.parametrize("field", ["tokenizer_sha256", "normalizer_sha256", "base_tokenizer_sha256"])
def test_clean_policy_must_pin_current_artifact(candidate, field):
    policy = json.loads(candidate[1].read_text())
    policy[field] = "wrong"
    write(candidate[1], policy)
    with pytest.raises(ValueError, match=field):
        validate_clean_tokenizer(*candidate)


def test_frozen_manifest_binds_unchanged_original_normalizer(candidate):
    bundle, _, data = candidate
    manifest = json.loads(data.read_text())
    manifest["normalizer_sha256"] = "changed-normalizer"
    write(data, manifest)
    with pytest.raises(ValueError, match="normalizer hash disagrees with source"):
        validate_clean_tokenizer(*candidate)


def test_clean_reserve_requires_current_receipt_before_any_corpus_access(candidate, monkeypatch):
    original = Path.open
    def guard(self, *args, **kwargs):
        assert self.suffix != ".jsonl", "Opened corpus before rejecting receipt"
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    with pytest.raises(ValueError, match="requires a selection receipt"):
        validate_clean_tokenizer(*candidate, phase="reserve")
    rec = receipt(candidate)
    value = json.loads(rec.read_text())
    value["tokenizer_sha256"] = json.loads(resources.files("untok").joinpath("data", "source", "manifest.json").read_text())["tokenizer_sha256"]
    write(rec, value)
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        validate_clean_tokenizer(*candidate, phase="reserve", selection_receipt_path=rec)


def test_clean_reserve_omits_text_examples(candidate):
    result = validate_clean_tokenizer(*candidate, phase="reserve", selection_receipt_path=receipt(candidate), max_examples=10)
    assert result["corpus_status"] == "passed"
    assert not result["corpora"]["ml"]["roundtrip_failure_examples"]
    assert not result["corpora"]["ml"]["base_representable_change_examples"]


def test_dev_does_not_open_reserve_and_training_cannot_use_an_unrelated_manifest(candidate, monkeypatch):
    original = Path.open
    def guard(self, *args, **kwargs):
        assert self.parent.name not in {"reserve", "train"}, "Opened sealed or unrelated corpus"
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    result = validate_clean_tokenizer(*candidate)
    assert not result["training_usage"]["selection_manifest_matches"]
    assert result["corpus_status"] == "passed"


def test_clean_unknown_and_efficiency_thresholds_fail_release(candidate):
    bundle, policy_path, data = candidate
    policy = json.loads(policy_path.read_text())
    policy["corpus_thresholds"]["dev"]["default"]["max_tokens_p95"] = 0
    write(policy_path, policy)
    result = validate_clean_tokenizer(bundle, policy_path, data)
    assert result["corpus_status"] == result["status"] == "failed"
    assert not result["corpus_gates"]["configured_release_thresholds"]


def test_clean_corpus_expected_text_is_still_an_explicit_contract(candidate):
    data = candidate[2]
    path = data.parent / "dev/ml.jsonl"
    path.write_text(json.dumps({"text": "അവര്‍", "expected_text": "wrong"}) + "\n", encoding="utf-8")
    manifest = json.loads(data.read_text())
    manifest["files"]["dev/ml.jsonl"] = sha(path)
    write(data, manifest)
    result = validate_clean_tokenizer(*candidate)
    assert result["corpora"]["ml"]["roundtrip_failures"] == 1
    assert result["corpus_status"] == "failed"


@pytest.mark.parametrize("mutation", ["piece", "score", "type", "normalizer", "metadata"])
def test_validator_independently_rejects_original_native_mutations(candidate, monkeypatch, mutation):
    from sentencepiece import sentencepiece_model_pb2 as pb
    import untok.clean_validation as module

    adapter = module.CleanTokenizerAdapter(candidate[0])
    model = pb.ModelProto()
    model.ParseFromString(adapter.model_bytes)
    if mutation == "piece":
        model.pieces[100].piece += "altered"
    elif mutation == "score":
        model.pieces[100].score -= 1
    elif mutation == "type":
        model.pieces[100].type = pb.ModelProto.SentencePiece.UNUSED
    elif mutation == "normalizer":
        model.normalizer_spec.add_dummy_prefix = not model.normalizer_spec.add_dummy_prefix
    else:
        model.trainer_spec.model_prefix += "altered"
    adapter.model_bytes = model.SerializeToString()
    # Simulate a loader regression: validation must independently verify the
    # complete native prefix, rather than trust an adapter's advertised status.
    monkeypatch.setattr(module, "CleanTokenizerAdapter", lambda _: adapter)
    with pytest.raises(ValueError, match="Native piece message|Native metadata"):
        module.validate_clean_tokenizer(*candidate)
