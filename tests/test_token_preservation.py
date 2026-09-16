"""Independent published-baseline checks for the simplified current builder."""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil

import pytest
from sentencepiece import sentencepiece_model_pb2 as pb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src/untok/data"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load_script("verify_tokens").verify
build = load_script("build_tokenizers").build


def test_every_shipped_asset_and_token_is_preserved():
    report = verify(DATA)
    assert report["assets_unchanged"] == 39
    assert report["profiles"]["latin-indic"]["added_entries"] == 7273


def test_current_builder_reproduces_all_published_bytes(tmp_path):
    report = build(tmp_path)
    assert report["files_written"] == 39
    assert verify(tmp_path, baseline_dir=DATA)["passed"]
    for original in DATA.glob("*/*"):
        if original.is_file():
            assert (tmp_path / original.relative_to(DATA)).read_bytes() == original.read_bytes()
    for profile in report["profiles"]:
        assert (tmp_path / profile / "licenses/OpenMDW-1.1.txt").is_file()
        assert (tmp_path / profile / "THIRD_PARTY.md").is_file()


@pytest.mark.parametrize("mutation", ["remove", "reorder", "score", "type", "normalizer"])
def test_verifier_rejects_changes_even_if_local_manifest_hashes_are_updated(tmp_path, mutation):
    shutil.copytree(DATA, tmp_path, dirs_exist_ok=True)
    target = tmp_path / "full/tokenizer.model"
    model = pb.ModelProto.FromString(target.read_bytes())
    if mutation == "remove":
        del model.pieces[-1]
    elif mutation == "reorder":
        first, second = model.pieces[-2].SerializeToString(), model.pieces[-1].SerializeToString()
        model.pieces[-2].ParseFromString(second)
        model.pieces[-1].ParseFromString(first)
    elif mutation == "score":
        model.pieces[-1].score += 0.125
    elif mutation == "type":
        model.pieces[-1].type = pb.ModelProto.SentencePiece.UNUSED
    else:
        model.normalizer_spec.add_dummy_prefix = not model.normalizer_spec.add_dummy_prefix
    target.write_bytes(model.SerializeToString())
    manifest_path = tmp_path / "full/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["tokenizer.model"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(tmp_path, baseline_dir=DATA)


def test_builder_refuses_in_place_overwrite():
    with pytest.raises(ValueError, match="Output must be empty"):
        build(DATA)


@pytest.mark.parametrize("field", ["inactive_native_ids", "target_blank_id"])
def test_verifier_rejects_changed_inactive_mask_or_blank(tmp_path, field):
    shutil.copytree(DATA, tmp_path, dirs_exist_ok=True)
    target = tmp_path / "latin-indic/native-row-map.json"
    mapping = json.loads(target.read_text())
    if field == "inactive_native_ids":
        mapping[field].pop()
    else:
        mapping[field] -= 1
    target.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="inactive IDs changed|native blank changed"):
        verify(tmp_path)
