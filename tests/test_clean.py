"""Native ID preservation and artifact integrity of append-only Unigram bundles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.bundles import PROFILES, load_tokenizer_bundle
from untok.clean import ALGORITHM, CleanTokenizerAdapter, build_clean_bundles
from untok.unigram import validate_native_prefix


DATA = Path(__file__).resolve().parents[1] / "src" / "untok" / "data"


def _model(path):
    model = pb.ModelProto()
    model.ParseFromString(Path(path).read_bytes())
    return model


def _files(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


@pytest.fixture(scope="module")
def clean_bundles(tmp_path_factory):
    source = DATA / "source"
    before = _files(source)
    output = tmp_path_factory.mktemp("clean-bundles")
    build_clean_bundles(source, output)
    assert _files(source) == before
    adapters = {profile: load_tokenizer_bundle(output / profile) for profile in PROFILES}
    return output, adapters


@pytest.mark.parametrize("profile", PROFILES)
def test_every_original_id_piece_score_type_and_metadata_is_unchanged(clean_bundles, profile):
    output, adapters = clean_bundles
    base_bytes = (DATA / "source" / "base-tokenizer.model").read_bytes()
    base = _model(DATA / "source" / "base-tokenizer.model")
    target = _model(output / profile / "tokenizer.model")
    report = validate_native_prefix(base_bytes, adapters[profile].model_bytes)
    assert report["native_entries_preserved"] == 13087
    assert len(base.pieces) == 13087
    for original_id, piece in enumerate(base.pieces):
        assert target.pieces[original_id].SerializeToString() == piece.SerializeToString()
        assert adapters[profile].token_to_id(piece.piece) == original_id
    assert adapters[profile].token_to_id("▁в") == 45
    assert adapters[profile].token_to_id("o") == 46
    assert target.normalizer_spec.SerializeToString() == base.normalizer_spec.SerializeToString()
    for model in (base, target):
        model.ClearField("pieces")
        model.trainer_spec.ClearField("vocab_size")
    assert target.SerializeToString() == base.SerializeToString()


@pytest.mark.parametrize("profile", ["full", "latin-indic"])
def test_malayalam_spelling_and_standalone_runtime_preserve_native_normalization(clean_bundles, profile):
    output, adapters = clean_bundles
    adapter = adapters[profile]
    standalone = spm.SentencePieceProcessor(model_file=str(output / profile / "tokenizer.model"))
    original = spm.SentencePieceProcessor(model_file=str(DATA / "source" / "base-tokenizer.model"))
    pairs = [
        ("ണ\u0d4d\u200d", "ൺ"),
        ("ന\u0d4d\u200d", "ൻ"),
        ("ര\u0d4d\u200d", "ർ"),
        ("ല\u0d4d\u200d", "ൽ"),
        ("ള\u0d4d\u200d", "ൾ"),
        ("അവന്\u200d", "അവൻ"),
        ("അവര്\u200d", "അവർ"),
        ("അവള്\u200d", "അവൾ"),
        ("വീട്ടില്\u200d", "വീട്ടിൽ"),
    ]
    for legacy, atomic in pairs:
        assert standalone.encode(legacy) != standalone.encode(atomic)
        for text in (legacy, atomic):
            ids = standalone.encode(text, out_type=int)
            assert adapter.unk_id not in ids
            assert ids == adapter.text_to_ids(text)
            assert standalone.decode(ids) == adapter.ids_to_text(ids) == text
            assert standalone.normalize(text) == original.normalize(text)


@pytest.mark.parametrize("profile", ["full", "latin-indic"])
def test_joiner_normalization_is_identical_to_the_native_model(clean_bundles, profile):
    _, adapters = clean_bundles
    adapter = adapters[profile]
    original = spm.SentencePieceProcessor(model_file=str(DATA / "source" / "base-tokenizer.model"))
    preserved = [
        "ന്\u200dറ", "അവന്\u200dറ", "എന്\u200dറെ",
        "क्\u200dष", "a\u200db",
    ]
    for text in preserved:
        ids = adapter.text_to_ids(text)
        assert adapter.unk_id not in ids
        assert adapter.ids_to_text(ids) == text
        assert adapter.backend.normalize(text) == original.normalize(text)
    mixed = "അവന്\u200dറ അവര്\u200d"
    assert adapter.ids_to_text(adapter.text_to_ids(mixed)) == mixed
    for text in ["क्\u200cष", "കു\u200c", "ا\u200cب", "a\u200cb"]:
        assert adapter.backend.normalize(text) == original.normalize(text)
        assert adapter.text_to_ids(text) == adapter.text_to_ids(text.replace("\u200c", " "))
        assert adapter.ids_to_text(adapter.text_to_ids(text)) == text.replace("\u200c", " ")


@pytest.mark.parametrize("profile", PROFILES)
def test_native_normalization_and_whitespace_have_no_exceptions(clean_bundles, profile):
    _, adapters = clean_bundles
    current = adapters[profile].backend
    original = spm.SentencePieceProcessor(model_file=str(DATA / "source" / "tokenizer.model"))
    for text in [
        "  hello  world  ", "a\tb\nc", "Ａ，ﬁ？⁇", "e\u0301", "क़", "가",
        "a\u200db", "<bg-BG> hello <hi-IN>", "½ Ⅳ ²", "a\u00a0b",
        "a\u200cb", "ന്\u200dറ", "അവര്\u200d", "അവർ",
    ]:
        assert current.normalize(text) == original.normalize(text)
    for text in ["  hello  world  ", "\u200d", "Ð Þ Ā Ċ Ə"]:
        ids = adapters[profile].text_to_ids(text)
        assert adapters[profile].unk_id not in ids
        assert adapters[profile].ids_to_text(ids) == text


@pytest.mark.parametrize("profile", PROFILES)
def test_only_appended_pieces_must_be_stable_and_match_their_best_native_path(clean_bundles, profile):
    output, _ = clean_bundles
    model = _model(output / profile / "tokenizer.model")
    normalizer = spm.SentencePieceNormalizer(
        model_proto=model.SerializeToString(), add_dummy_prefix=False,
        escape_whitespaces=True, remove_extra_whitespaces=False,
    )
    # Use the real SentencePiece Viterbi engine independently of the builder's
    # split-dominance implementation. Removing only its synthetic prefix makes
    # the surface under test exactly the candidate piece, including boundaries.
    model.normalizer_spec.add_dummy_prefix = False
    processor = spm.SentencePieceProcessor(model_proto=model.SerializeToString())
    strings = [piece.piece for piece in model.pieces]
    assert len(strings) == len(set(strings))
    base_size = len(_model(DATA / "source" / "base-tokenizer.model").pieces)
    native_scores = [piece.score for piece in model.pieces[:base_size]
                     if piece.type == pb.ModelProto.SentencePiece.NORMAL]
    for piece in model.pieces[base_size:]:
        assert min(native_scores) <= piece.score <= max(native_scores)
        if piece.type != pb.ModelProto.SentencePiece.NORMAL:
            continue
        surface = piece.piece.replace("▁", " ")
        assert normalizer.normalize(surface) == piece.piece
        ids = processor.encode(surface, out_type=int)
        assert processor.unk_id() not in ids, piece.piece
        best_score = sum(processor.get_score(index) for index in ids)
        assert best_score <= piece.score + 1e-6, (piece.piece, best_score, piece.score)


@pytest.mark.parametrize("profile", PROFILES)
def test_retained_rows_new_rows_and_blank_namespaces_are_unambiguous(clean_bundles, profile):
    output, adapters = clean_bundles
    adapter = adapters[profile]
    assert isinstance(adapter, CleanTokenizerAdapter)
    original = _model(DATA / "source" / "tokenizer.model")
    base = _model(DATA / "source" / "base-tokenizer.model")
    target = _model(output / profile / "tokenizer.model")
    forward = adapter.full_native_to_subset_native
    reverse = adapter.subset_native_to_full_native
    assert len(forward) == len(original.pieces) + 1
    assert len(reverse) == len(target.pieces) + 1
    assert adapter.blank_id == adapter.vocab_size == len(target.pieces)
    assert forward[-1] == adapter.source_native_to_target_native[-1] == adapter.blank_id
    assert reverse[-1] == len(original.pieces)
    assert adapter.source_native_to_target_native[:-1] == forward[:len(base.pieces)]
    assert adapter.source_native_to_target_native[:-1] == tuple(range(len(base.pieces)))
    assert adapter.id_map.hf_pad_id == 13087
    assert adapter.id_map.hf_blank_id == 13088
    for new_id, old_id in enumerate(reverse[:-1]):
        if old_id is None:
            assert target.pieces[new_id].score == -32.0
            continue
        assert forward[old_id] == new_id
        assert target.pieces[new_id].SerializeToString() == original.pieces[old_id].SerializeToString()
    assert "\u200c" not in adapter.vocab
    assert reverse[adapter.token_to_id("Ð")] is None
    for piece in ["？", "⁇", "Ａ", "，", "▁anh", "▁в"]:
        old_id = next(i for i, row in enumerate(original.pieces) if row.piece == piece)
        assert forward[old_id] == old_id
        assert adapter.token_to_id(piece) == old_id
    with pytest.raises(ValueError, match="padding"):
        adapter.public_ids_to_text([adapter.id_map.hf_pad_id])
    with pytest.raises(ValueError, match="blank"):
        adapter.id_map.to_model([adapter.id_map.hf_blank_id])
    assert adapter.id_map.to_model([adapter.id_map.hf_blank_id], allow_blank=True) == [adapter.blank_id]
    assert adapter.public_ids_to_text([adapter.id_map.hf_blank_id]) == ""
    assert adapter.public_ids_to_text(adapter.text_to_public_ids("a\u200cb")) == "a b"


@pytest.mark.parametrize("name", ["tokenizer.model", "native-row-map.json", "cleanup.json", "manifest.json"])
def test_rehashed_artifact_tampering_cannot_pass_the_recipe_check(clean_bundles, tmp_path, name):
    output, _ = clean_bundles
    altered = tmp_path / "altered"
    shutil.copytree(output / "latin", altered)
    path = altered / name
    if name == "tokenizer.model":
        model = _model(path)
        model.pieces[2].score -= 0.5
        path.write_bytes(model.SerializeToString())
    else:
        data = json.loads(path.read_text())
        if name == "native-row-map.json":
            data["source_native_to_target_native"][0] = 1
        elif name == "cleanup.json":
            data["global_score_refit"] = True
        else:
            data["requires_retokenized_training_labels"] = not data["requires_retokenized_training_labels"]
        path.write_text(json.dumps(data))
    if name != "manifest.json":
        manifest_path = altered / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if name == "tokenizer.model":
            manifest["tokenizer_sha256"] = manifest["files"][name]
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned cleanup recipe"):
        load_tokenizer_bundle(altered)


def test_unrehashed_artifact_tampering_fails_before_recipe_reconstruction(clean_bundles, tmp_path):
    output, _ = clean_bundles
    altered = tmp_path / "altered"
    shutil.copytree(output / "latin", altered)
    path = altered / "native-row-map.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_tokenizer_bundle(altered)


def test_shipped_profiles_match_the_reproducible_recipe(clean_bundles):
    output, _ = clean_bundles
    for profile in PROFILES:
        manifest = json.loads((DATA / profile / "manifest.json").read_text())
        assert manifest["algorithm"] == ALGORITHM
        assert _files(output / profile) == _files(DATA / profile)


def test_zip_resources_keep_native_ids_and_normalization_after_extraction(tmp_path, monkeypatch):
    import zipfile
    import untok.bundles as bundles

    archive = tmp_path / "resources.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for directory in (DATA / "full", DATA / "extension-v3"):
            for path in directory.iterdir():
                stream.write(path, "data/" + path.relative_to(DATA).as_posix())
    with zipfile.ZipFile(archive) as stream:
        monkeypatch.setattr(bundles.resources, "files", lambda package: zipfile.Path(stream))
        adapter = bundles.load_tokenizer_bundle("full")
    archive.unlink()
    assert adapter.token_to_id("▁в") == 45
    assert adapter.token_to_id("o") == 46
    assert adapter.text_to_ids("അവര്‍") != adapter.text_to_ids("അവർ")
    assert adapter.ids_to_text(adapter.text_to_ids("a\u200cb")) == "a b"


def test_packaging_a_clean_subset_reconstructs_the_other_profiles(clean_bundles, tmp_path):
    from untok.bundles import package_tokenizer_bundles

    original, _ = clean_bundles
    output = tmp_path / "packaged"
    receipt = package_tokenizer_bundles(original / "latin", output, make_zips=False)
    assert receipt["source_tokenizer_sha256"] == hashlib.sha256((original / "latin" / "tokenizer.model").read_bytes()).hexdigest()
    assert receipt["full_tokenizer_sha256"] == hashlib.sha256((DATA / "source" / "tokenizer.model").read_bytes()).hexdigest()
    for profile in PROFILES:
        assert _files(output / profile) == _files(original / profile)
