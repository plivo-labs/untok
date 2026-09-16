"""Stable native IDs, inactive vocabulary slots and profile integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.bundles import PROFILES, load_tokenizer_bundle
from untok.clean import ALGORITHM, LEGACY_ALGORITHM, CleanTokenizerAdapter, build_clean_bundles
from untok.profile_policy import piece_allowed
from untok.unigram import build_native_tokenizer, native_id_map, validate_native_prefix


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


@pytest.mark.parametrize("profile", ["original", "full"])
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


def test_original_is_an_exact_copy_and_latin_never_adds_pieces(clean_bundles):
    output, adapters = clean_bundles
    original = (DATA / "source/base-tokenizer.model").read_bytes()
    assert (output / "original/tokenizer.model").read_bytes() == original
    base = pb.ModelProto.FromString(original)
    originals = {piece.piece for piece in base.pieces}
    assert set(adapters["latin"].vocab) < originals
    assert adapters["original"].blank_id == 13087
    assert adapters["original"].source_native_to_target_native == tuple(range(13088))
    for profile in PROFILES:
        assert "ð" not in adapters[profile].vocab  # removed rare-Latin source extension
        assert "Ð" not in adapters[profile].vocab  # removed later case-closure extension
    for profile in ("latin", "latin-indic"):
        assert "<bg-BG>" not in adapters[profile].vocab
        assert "<ru-RU>" not in adapters[profile].vocab
        assert "<ar-AR>" not in adapters[profile].vocab
        assert "<en-US>" in adapters[profile].vocab
        assert "▁в" not in adapters[profile].vocab
    assert "<hi-IN>" not in adapters["latin"].vocab
    assert "<hi-IN>" in adapters["latin-indic"].vocab


@pytest.mark.parametrize("profile,physical_size,active_size,original_active", [
    ("original", 13087, 13087, 13087),
    ("latin", 13087, 2653, 2653),
    ("latin-indic", 20360, 10372, 3099),
    ("full", 20360, 20360, 13087),
])
def test_every_active_original_piece_keeps_its_exact_native_slot(
    clean_bundles, profile, physical_size, active_size, original_active,
):
    output, adapters = clean_bundles
    adapter = adapters[profile]
    base = _model(DATA / "source/base-tokenizer.model")
    target = _model(output / profile / "tokenizer.model")
    active = set(adapter.active_native_ids)
    inactive = set(adapter.inactive_native_ids)
    vocabulary = adapter.get_vocab()
    expected_original = {index for index, row in enumerate(base.pieces) if piece_allowed(row, profile)}
    assert len(target.pieces) == adapter.vocab_size == physical_size
    assert adapter.active_vocab_size == len(vocabulary) == active_size
    assert len(expected_original) == original_active
    assert active.isdisjoint(inactive)
    assert active | inactive == set(range(physical_size))
    assert active & set(range(13087)) == expected_original
    assert inactive == set(range(13087)) - expected_original
    assert set(vocabulary.values()) == active
    assert adapter.vocab == vocabulary
    assert adapter.get_acoustic_vocab() == {row.piece: index for index, row in enumerate(target.pieces)}
    assert adapter.source_native_to_target_native[:-1] == tuple(range(13087))
    assert adapter.full_native_to_subset_native[:13087] == tuple(range(13087))
    assert adapter.subset_native_to_full_native[:13087] == tuple(range(13087))

    original_strings = {row.piece for row in base.pieces}
    placeholders = {target.pieces[index].piece for index in inactive}
    assert len(placeholders) == len(inactive)
    assert placeholders.isdisjoint(original_strings)
    assert placeholders.isdisjoint(vocabulary)
    for original_id, row in enumerate(base.pieces):
        actual = target.pieces[original_id]
        if original_id in active:
            assert actual.SerializeToString() == row.SerializeToString(), (profile, original_id, row.piece)
            assert adapter.token_to_id(row.piece) == original_id
            assert adapter.id_map.to_canonical([original_id]) == [original_id]
            assert adapter.id_map.to_model([original_id]) == [original_id]
        else:
            assert actual.type == pb.ModelProto.SentencePiece.UNUSED
            assert row.piece not in vocabulary
            with pytest.raises(ValueError):
                adapter.token_to_id(row.piece)
            with pytest.raises(ValueError):
                adapter.token_to_id(actual.piece)
    assert adapter.token_to_id("▁") == 2
    assert adapter.token_to_id("a") == 38
    assert adapter.token_to_id("o") == 46
    assert adapter.token_to_id("Ỳ") == 13086
    assert adapter.text_to_ids("a") == adapter.text_to_public_ids("a") == [2, 38]
    assert target.normalizer_spec.SerializeToString() == base.normalizer_spec.SerializeToString()


def test_indic_additions_are_appended_after_the_entire_original_id_space(clean_bundles):
    output, adapters = clean_bundles
    base = _model(DATA / "source/base-tokenizer.model")
    indic = _model(output / "latin-indic/tokenizer.model")
    full = _model(output / "full/tokenizer.model")
    assert len(indic.pieces[13087:]) == len(full.pieces[13087:]) == 7273
    assert [row.SerializeToString() for row in indic.pieces[13087:]] == [
        row.SerializeToString() for row in full.pieces[13087:]
    ]
    original_strings = {row.piece for row in base.pieces}
    for index, row in enumerate(indic.pieces[13087:], start=13087):
        assert row.piece not in original_strings
        assert index in adapters["latin-indic"].active_native_ids
        assert adapters["latin-indic"].token_to_id(row.piece) == adapters["full"].token_to_id(row.piece) == index
        assert adapters["latin-indic"].id_map.to_canonical([index]) == [index + 2]
    assert set(adapters["latin"].get_vocab()) < original_strings
    for token, index in adapters["latin"].get_vocab().items():
        assert adapters["latin-indic"].token_to_id(token) == index
    for token, original_id in {"<hi-IN>": 3247, "क": 3251, "▁है": 3260}.items():
        assert base.pieces[original_id].piece == token
        assert adapters["latin-indic"].token_to_id(token) == original_id
        assert adapters["full"].token_to_id(token) == original_id
    # The full profile's model bytes already satisfied this contract in v4.
    assert hashlib.sha256(adapters["full"].model_bytes).hexdigest() == "815ee2313f17db264eb681f4f5a1a322fbb05c1667d9808fb3fb357ae749b857"


@pytest.mark.parametrize("profile", ["latin", "latin-indic"])
def test_inactive_slots_cannot_be_encoded_or_exposed_as_text_labels(clean_bundles, profile):
    output, adapters = clean_bundles
    adapter = adapters[profile]
    base = _model(DATA / "source/base-tokenizer.model")
    target = _model(output / profile / "tokenizer.model")
    inactive = set(adapter.inactive_native_ids)
    standalone = spm.SentencePieceProcessor(model_file=str(output / profile / "tokenizer.model"))
    # Exercise every excluded spelling and every placeholder, including the
    # source USER_DEFINED <bg-BG> tag which otherwise bypasses normal matching.
    for index in sorted(inactive):
        for text in (base.pieces[index].piece.replace("▁", " "), target.pieces[index].piece):
            ids = adapter.text_to_ids(text)
            assert ids == standalone.encode(text, out_type=int)
            assert inactive.isdisjoint(ids), (profile, index, text, ids)
            assert target.pieces[index].piece not in adapter.text_to_tokens(text)
        for decode in (adapter.ids_to_tokens, adapter.ids_to_text, adapter.public_ids_to_text):
            with pytest.raises(ValueError, match="[Ii]nactive"):
                decode([index])
        with pytest.raises(ValueError, match="[Ii]nactive"):
            adapter.id_to_token(index)
    assert "<bg-BG>" not in target.trainer_spec.user_defined_symbols
    assert 1 in inactive


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
    for text in ["  hello  world  ", "a b"]:
        ids = adapters[profile].text_to_ids(text)
        assert adapters[profile].unk_id not in ids
        assert adapters[profile].ids_to_text(ids) == text


@pytest.mark.parametrize("profile", PROFILES)
def test_only_appended_pieces_must_be_stable_and_match_their_best_native_path(clean_bundles, profile):
    output, adapters = clean_bundles
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
    native_scores = [piece.score for piece in _model(DATA / "source/base-tokenizer.model").pieces
                     if piece.type == pb.ModelProto.SentencePiece.NORMAL]
    for piece, source_id in zip(model.pieces, adapters[profile].subset_native_to_full_native[:-1]):
        if source_id < base_size:
            continue
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
    assert None not in reverse[:-1]
    for new_id, old_id in enumerate(reverse[:-1]):
        assert forward[old_id] == new_id
        if new_id in adapter.inactive_native_ids:
            assert new_id == old_id
            assert target.pieces[new_id].type == pb.ModelProto.SentencePiece.UNUSED
        else:
            assert target.pieces[new_id].SerializeToString() == original.pieces[old_id].SerializeToString()
    assert "\u200c" not in adapter.vocab
    assert "Ð" not in adapter.vocab
    for piece in ["？", "⁇", "Ａ", "，", "▁anh", "▁в"]:
        old_id = next(i for i, row in enumerate(original.pieces) if row.piece == piece)
        assert forward[old_id] == old_id
        if piece in adapter.vocab:
            assert adapter.token_to_id(piece) == old_id
        else:
            assert old_id in adapter.inactive_native_ids
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
        assert {name: raw for name, raw in _files(output / profile).items()
                if name != "THIRD_PARTY.md"} == _files(DATA / profile)


def test_zip_resources_keep_native_ids_and_normalization_after_extraction(tmp_path, monkeypatch):
    import zipfile
    import untok.bundles as bundles

    archive = tmp_path / "resources.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for directory in (DATA / "full", DATA / "extension-v3"):
            for path in directory.iterdir():
                stream.write(path, "data/" + path.relative_to(DATA).as_posix())
        stream.write(DATA / "source/selection.json", "data/source/selection.json")
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


def test_exact_native_identity_keeps_historical_trainer_vocabulary_declaration(tmp_path):
    from untok.bundles import package_tokenizer_bundles

    base = pb.ModelProto()
    base.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    base.trainer_spec.unk_id = 0
    base.trainer_spec.bos_id = base.trainer_spec.eos_id = base.trainer_spec.pad_id = -1
    base.trainer_spec.vocab_size = 256  # Historical declaration is not the actual inventory size.
    base.normalizer_spec.name = "identity"
    for text, score, kind in [("<unk>", 0, 2), ("▁", 0, 1), ("a", -3, 1)]:
        base.pieces.add(piece=text, score=score, type=kind)
    raw = base.SerializeToString()
    base_path = tmp_path / "base.model"
    base_path.write_bytes(raw)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"base_tokenizer_sha256": hashlib.sha256(raw).hexdigest(),
                                    "additions": [{"piece": "b", "score": -2}]}))
    source = tmp_path / "source"
    build_native_tokenizer(base_path, selection, source)
    # A byte-identical source can retain its historical declared size. This
    # low-level identity contract does not make an arbitrary base Nemotron.
    assert validate_native_prefix(raw, raw)["native_entries_preserved"] == 3
    mapping = native_id_map(raw, raw)
    assert mapping.model_blank_id == 3
    assert base_path.read_bytes() == raw
    destination = tmp_path / "packaged"
    with pytest.raises(ValueError, match="pinned original"):
        package_tokenizer_bundles(source, destination, profiles=("original",), make_zips=False)
    assert not destination.exists()


def test_rehashed_original_bundle_cannot_replace_the_pinned_base(clean_bundles, tmp_path):
    output, _ = clean_bundles
    altered = tmp_path / "forged-original"
    shutil.copytree(output / "original", altered)
    manifest = json.loads((altered / "manifest.json").read_text())
    for name in ("base-tokenizer.model", "full-tokenizer.model", "tokenizer.model"):
        path = altered / name
        model = _model(path)
        model.pieces[2].score -= 0.01
        path.write_bytes(model.SerializeToString())
        manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["base_tokenizer_sha256"] = manifest["files"]["base-tokenizer.model"]
    manifest["full_tokenizer_sha256"] = manifest["files"]["full-tokenizer.model"]
    manifest["tokenizer_sha256"] = manifest["files"]["tokenizer.model"]
    (altered / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned original"):
        load_tokenizer_bundle(altered)


def test_frozen_v3_bundles_remain_loadable_after_profile_contract_changes(tmp_path):
    from untok.clean import _preserved_v3_artifacts

    files, manifest, _, _ = _preserved_v3_artifacts(
        (DATA / "source/base-tokenizer.model").read_bytes(),
        (DATA / "source/tokenizer.model").read_bytes(), "latin")
    assert manifest["algorithm"] == LEGACY_ALGORITHM
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    adapter = load_tokenizer_bundle(tmp_path)
    assert adapter.vocab_size == 13573
    assert adapter.token_to_id("▁в") == 45
    assert "Ð" in adapter.vocab
    assert hashlib.sha256(adapter.model_bytes).hexdigest() == "e250b6b2f47ed337f13c3a957637feada09dbcf6e6d1bf628119647eadb70f12"


def test_frozen_v4_compact_bundles_remain_loadable_after_stable_id_changes(tmp_path):
    from untok.clean import V4_ALGORITHM, _profiles_v4_artifacts

    files, manifest, _, _ = _profiles_v4_artifacts(
        (DATA / "source/base-tokenizer.model").read_bytes(),
        (DATA / "source/tokenizer.model").read_bytes(), "latin")
    assert manifest["algorithm"] == V4_ALGORITHM
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    adapter = load_tokenizer_bundle(tmp_path)
    assert adapter.vocab_size == adapter.active_vocab_size == 2653
    assert not adapter.inactive_native_ids
    assert adapter.token_to_id("▁") == 1
    assert adapter.token_to_id("a") == 6
    assert adapter.source_native_to_target_native[38] == 6
    assert hashlib.sha256(adapter.model_bytes).hexdigest() == "035d463b9906a291b3428d56f1622dc758bb05b388b4d66905e52b68d93ea714"
