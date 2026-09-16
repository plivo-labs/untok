"""Current native IDs and artifact integrity, with no historical build path."""
import json
from pathlib import Path
import pickle
import shutil

import pytest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.bundles import PROFILES, load_tokenizer

DATA = Path(__file__).resolve().parents[1] / "src/untok/data"
COUNTS = {"original": (13087, 13087), "latin": (13087, 2653),
          "latin-indic": (20360, 10372), "full": (20360, 20360)}
TEXTS = ("", "which", "a", "Hello world.", "हिन्दी मराठी", "অসমীয়া বাংলা", "ગુજરાતી", "ਪੰਜਾਬੀ",
         "ଓଡ଼ିଆ", "தமிழ்", "తెలుగు", "ಕನ್ನಡ", "അവന്‍", "اردو", "ᱥᱟᱱᱛᱟᱲᱤ", "ꯃꯤꯇꯩ",
         "Ａ ﬁ", "a\u200cb", "  two  spaces ", "🙂 unknown")


@pytest.mark.parametrize("profile", PROFILES)
def test_native_model_id_contract_and_sentencepiece_behavior(profile):
    adapter = load_tokenizer(profile)
    model = pb.ModelProto.FromString(adapter.model_bytes)
    base = pb.ModelProto.FromString(adapter.base_model_bytes)
    size, active = COUNTS[profile]
    assert (adapter.vocab_size, adapter.blank_id, adapter.pad_id) == (size, size, size)
    assert adapter.active_vocab_size == active
    assert adapter.source_native_to_target_native == (*range(13087), size)
    assert len(adapter.inactive_native_ids) == size - active
    assert set(adapter.get_acoustic_vocab().values()) == set(range(size))
    assert set(adapter.get_vocab().values()) == set(adapter.active_native_ids)
    for index, original in enumerate(base.pieces):
        piece = model.pieces[index]
        if index in adapter._inactive:
            assert piece.type == pb.ModelProto.SentencePiece.UNUSED
        else:
            assert piece.SerializeToString() == original.SerializeToString(), index
    assert model.normalizer_spec.SerializeToString() == base.normalizer_spec.SerializeToString()
    expected = spm.SentencePieceProcessor(model_file=str(DATA / profile / "tokenizer.model"))
    restored = pickle.loads(pickle.dumps(adapter))
    for text in TEXTS:
        ids = expected.encode(text)
        assert adapter.text_to_ids(text) == restored.text_to_ids(text) == ids
        assert adapter.ids_to_text([*ids, size]) == expected.decode(ids)
        assert not adapter._inactive.intersection(ids)


@pytest.mark.parametrize("profile", PROFILES)
def test_explicit_directory_uses_the_same_current_profile(profile):
    assert load_tokenizer(DATA / profile).model_bytes == load_tokenizer(profile).model_bytes


@pytest.mark.parametrize("name", ["tokenizer.model", "native-row-map.json", "cleanup.json"])
def test_modified_artifact_is_rejected(tmp_path, name):
    folder = tmp_path / "changed"
    shutil.copytree(DATA / "latin-indic", folder)
    path = folder / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_tokenizer(folder)


def test_recomputed_manifest_cannot_authorize_a_token_change(tmp_path):
    folder = tmp_path / "changed"
    shutil.copytree(DATA / "full", folder)
    model = pb.ModelProto.FromString((folder / "tokenizer.model").read_bytes())
    model.pieces[13087].score += 1
    changed = model.SerializeToString()
    (folder / "tokenizer.model").write_bytes(changed)
    manifest = json.loads((folder / "manifest.json").read_text())
    import hashlib
    manifest["files"]["tokenizer.model"] = hashlib.sha256(changed).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="published current profile"):
        load_tokenizer(folder)


@pytest.mark.parametrize("value", [True, 1.5, -1, 20361])
def test_invalid_text_ids_are_rejected(value):
    with pytest.raises(ValueError):
        load_tokenizer("latin-indic").ids_to_text([value])


def test_inactive_ids_are_not_text_and_blank_is_not_a_piece():
    tokenizer = load_tokenizer("latin-indic")
    with pytest.raises(ValueError, match="Inactive"):
        tokenizer.ids_to_text([tokenizer.inactive_native_ids[0]])
    assert tokenizer.id_to_token(tokenizer.blank_id) == "<blank>"
    assert "text_to_public_ids" not in dir(tokenizer)
