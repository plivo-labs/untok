"""Dense script bundles must preserve pieces and behavior, not acoustic accuracy."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import zipfile

import pytest
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.bundles import (
    NativeSubsetTokenizerAdapter, character_allowed, deterministic_bundle_zip,
    load_tokenizer_bundle, package_tokenizer_bundles,
)
from untok.unigram import NativeTokenizerAdapter, build_native_tokenizer


@pytest.fixture
def full(tmp_path):
    model = pb.ModelProto()
    model.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    model.trainer_spec.unk_id = 0
    model.trainer_spec.bos_id = model.trainer_spec.eos_id = model.trainer_spec.pad_id = -1
    model.normalizer_spec.name = "identity"
    model.normalizer_spec.remove_extra_whitespaces = False
    # Deliberately interleave scripts to exercise dense ID gathering.
    inventory = [
        ("<unk>", 0, pb.ModelProto.SentencePiece.UNKNOWN),
        ("▁", 0, 1), ("a", -2, 1), ("я", -3, 1), ("b", -3, 1),
        ("ab", -2, 1), ("z", -253, 1), ("क", -3, 1),
        ("aя", -1, 1), ("aक", -1, 1), ("<bg-BG>", 0, pb.ModelProto.SentencePiece.USER_DEFINED),
        ("<ru-RU>", 0, 1), ("【", -4, 1), ("】", -4, 1), ("ー", -4, 1),
        ("\u0301", -4, 1), ("\u0342", -4, 1), ("!", -4, 1),
    ]
    for text, score, kind in inventory:
        model.pieces.add(piece=text, score=score, type=kind)
    model.trainer_spec.vocab_size = len(model.pieces)
    base = tmp_path / "base.model"
    base.write_bytes(model.SerializeToString())
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "base_tokenizer_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "additions": [{"piece": p, "score": -4} for p in ["ক", "ગ", "ਕ", "କ", "க", "క", "ಕ", "ക", "ڪ", "ꯀ", "ᱚ", "▁க"]],
    }))
    directory = tmp_path / "full"
    build_native_tokenizer(base, selection, directory)
    return directory


def package(full, tmp_path):
    output = tmp_path / "packaged"
    package_tokenizer_bundles(full, output, profiles=("latin", "latin-indic", "full"))
    return output


def test_script_policy_shared_punctuation_and_mixed_scripts(full, tmp_path):
    output = package(full, tmp_path)
    latin = load_tokenizer_bundle(output / "latin")
    indic = load_tokenizer_bundle(output / "latin-indic")
    assert isinstance(latin, NativeSubsetTokenizerAdapter)
    for text in ["a", "ab", "z", "【", "】", "!", "\u0301", "<bg-BG>", "<ru-RU>"]:
        assert text in latin.vocab
    for text in ["क", "aक", "я", "aя", "ー", "\u0342"]:
        assert text not in latin.vocab
    for text in ["क", "aक", "ক", "ગ", "ਕ", "କ", "க", "క", "ಕ", "ക", "ڪ", "ꯀ", "ᱚ"]:
        assert text in indic.vocab
    for text in ["я", "aя", "ー", "\u0342"]:
        assert text not in indic.vocab
    assert character_allowed("\u200d", "latin-indic")
    assert character_allowed("\u0951", "latin-indic")
    assert character_allowed("\u0951", "latin")  # Unicode explicitly includes Latin transliteration
    assert not character_allowed("\u1cd1", "latin")
    assert not character_allowed("\u0378", "latin")  # unassigned
    with pytest.raises(ValueError):
        character_allowed("ab", "latin")


def test_dense_row_maps_blank_and_public_padding(full, tmp_path):
    output = package(full, tmp_path)
    original = load_tokenizer_bundle(full)
    reduced = load_tokenizer_bundle(output / "latin-indic")
    mapping = reduced.row_map
    assert len(mapping.source_native_to_target_native) == 19
    assert mapping.source_native_to_target_native[3] is None
    assert mapping.source_native_to_target_native[-1] == reduced.blank_id
    assert mapping.full_native_to_subset_native[-1] == reduced.blank_id
    assert mapping.subset_native_to_full_native[-1] == original.blank_id
    assert mapping.source_native_to_target_native[4] == reduced.token_to_id("b") != 4
    assert mapping.from_source([4]) == [reduced.token_to_id("b")]
    with pytest.raises(ValueError, match="removed"):
        mapping.from_source([3])
    with pytest.raises(ValueError, match="blank"):
        mapping.from_source([18])
    assert mapping.from_source([18], allow_blank=True) == [reduced.blank_id]
    with pytest.raises(ValueError, match="integers"):
        mapping.from_source([4.1])
    for target, source in enumerate(mapping.subset_native_to_full_native):
        assert mapping.full_native_to_subset_native[source] == target
    with pytest.raises(ValueError, match="padding"):
        reduced.public_ids_to_text([reduced.id_map.hf_pad_id])
    assert reduced.public_ids_to_text(reduced.text_to_public_ids("ab")) == "ab"
    assert reduced.public_ids_to_text([reduced.id_map.hf_blank_id]) == ""
    # The old base blank occupies a genuine added text row in the full model.
    assert original.source_native_to_target_native[-1] == original.blank_id
    assert original.full_native_to_subset_native[18] == 18


def test_full_files_and_archives_are_reproducible(full, tmp_path):
    def files(directory):
        return {p.relative_to(directory).as_posix(): p.read_bytes()
                for p in directory.rglob("*") if p.is_file()}

    before = files(full)
    first, second = tmp_path / "one", tmp_path / "two"
    result = package_tokenizer_bundles(full, first, profiles=("latin", "latin-indic", "full"))
    package_tokenizer_bundles(full, second, profiles=("full", "latin-indic", "latin"))
    assert before == files(full)
    assert before == files(first / "full")
    for name in ("latin.zip", "latin-indic.zip", "full.zip", "bundles.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    for profile, item in result["bundles"].items():
        archive = first / item["zip"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == item["zip_sha256"]
        with zipfile.ZipFile(archive) as z:
            assert z.namelist() == sorted(z.namelist())
            assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in z.infolist())
            assert z.read("tokenizer.model") == (first / profile / "tokenizer.model").read_bytes()
    with pytest.raises(ValueError, match="empty"):
        package_tokenizer_bundles(full, first)
    with pytest.raises(ValueError, match="overlap"):
        package_tokenizer_bundles(full, full / "reduced")


def test_prefix_validation_stays_strict_and_filtered_text_parity(full, tmp_path):
    output = package(full, tmp_path)
    original = load_tokenizer_bundle(full)
    for profile, alphabet in [("latin", ["a", "b", "z", "!", "🙂"]),
                              ("latin-indic", ["a", "क", "க", "ಕ", "🙂"])]:
        reduced = load_tokenizer_bundle(output / profile)
        with pytest.raises(ValueError, match="native Unigram bundle"):
            NativeTokenizerAdapter(output / profile)
        for length in range(1, 4):
            for chars in itertools.product(alphabet, repeat=length):
                text = "".join(chars)
                full_ids = original.text_to_ids(text)
                assert reduced.text_to_ids(text) == reduced.row_map.from_full(full_ids)
                assert reduced.ids_to_text(reduced.text_to_ids(text)) == original.ids_to_text(full_ids)
        assert reduced.ids_to_text([reduced.token_to_id("a")] * 2 + [reduced.blank_id]) == "aa"
        left, right = pb.ModelProto(), pb.ModelProto()
        left.ParseFromString(original.model_bytes)
        right.ParseFromString(reduced.model_bytes)
        assert left.normalizer_spec.SerializeToString() == right.normalizer_spec.SerializeToString()
        for target, source in enumerate(reduced.subset_native_to_full_native[:-1]):
            assert right.pieces[target].SerializeToString() == left.pieces[source].SerializeToString()


def test_minimum_score_removal_fails_without_adding_dummy_piece(full, tmp_path):
    base = tmp_path / "minimum-base.model"
    proto = pb.ModelProto()
    proto.ParseFromString((full / "base-tokenizer.model").read_bytes())
    proto.pieces[3].score = -254  # only Cyrillic has the original minimum
    base.write_bytes(proto.SerializeToString())
    selection = json.loads((full / "selection.json").read_text())
    selection["base_tokenizer_sha256"] = hashlib.sha256(base.read_bytes()).hexdigest()
    path = tmp_path / "minimum-selection.json"
    path.write_text(json.dumps(selection))
    bundle = tmp_path / "minimum-full"
    build_native_tokenizer(base, path, bundle)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="score extrema"):
        package_tokenizer_bundles(bundle, output, profiles=("latin", "latin-indic", "full"))
    assert not output.exists()


@pytest.mark.parametrize("artifact", ["native-row-map.json", "tokenizer.model", "script-policy.json", "manifest.json"])
def test_self_rehashed_tampering_is_rejected(full, tmp_path, artifact):
    output = package(full, tmp_path) / "latin"
    path = output / artifact
    manifest = json.loads((output / "manifest.json").read_text())
    if artifact == "tokenizer.model":
        proto = pb.ModelProto()
        proto.ParseFromString(path.read_bytes())
        proto.pieces[2].score -= 1
        path.write_bytes(proto.SerializeToString())
    else:
        data = json.loads(path.read_text())
        if artifact == "native-row-map.json":
            data["source_native_to_target_native"][2] = None
        elif artifact == "script-policy.json":
            data["allowed_scripts"].append("Cyrillic")
        else:
            data["native_vocabulary_size"] += 1
        path.write_text(json.dumps(data))
    if artifact != "manifest.json":
        manifest["files"][artifact] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disagrees"):
        load_tokenizer_bundle(output)


def test_zip_rejects_symlinks_before_creation(tmp_path):
    directory = tmp_path / "files"
    directory.mkdir()
    (directory / "leak").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        deterministic_bundle_zip(directory, tmp_path / "bad.zip")
    assert not (tmp_path / "bad.zip").exists()


def test_actual_candidate_all_22_alphabets_and_protected_groups(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    full = repo / "artifacts/nemotron-indic-unigram-v1"
    if not full.exists():
        pytest.skip("Locally generated candidate is not part of the source distribution")
    policy = json.loads((repo / "configs/native-unigram-validation.json").read_text())
    assert len(policy["profiles"]) == 22
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in full.iterdir() if p.is_file()}
    output = package(full, tmp_path)
    original = load_tokenizer_bundle(full)
    latin = load_tokenizer_bundle(output / "latin")
    indic = load_tokenizer_bundle(output / "latin-indic")
    assert (latin.vocab_size, indic.vocab_size, original.vocab_size) == (2653, 10372, 20550)
    assert sum(i is None for i in indic.full_native_to_subset_native[13087:-1]) == 190
    for entry in policy["profiles"].values():
        for character in entry["characters"]:
            ids = indic.text_to_ids(character)
            assert indic.unk_id not in ids
            assert ids == [indic.full_native_to_subset_native[i] for i in original.text_to_ids(character)]
            assert indic.ids_to_text(ids) == original.ids_to_text(original.text_to_ids(character))
    for name, adapter in [("hindi103", indic)]:
        for piece in policy["protected_piece_groups"][name]["pieces"]:
            assert piece in adapter.vocab
            original_id = original.token_to_id(piece)
            reduced_id = adapter.token_to_id(piece)
            assert adapter.full_native_to_subset_native[original_id] == reduced_id
            assert adapter.backend.get_score(reduced_id) == original.backend.get_score(original_id)
    for adapter in (latin, indic):
        assert not set(policy["protected_piece_groups"]["latin190"]["pieces"]) & set(adapter.vocab)
    for adapter in (latin, indic):
        for text in policy["normalizer_probes"]:
            normalized = original.backend.normalize(text)
            assert adapter.backend.normalize(text) == normalized
            if all(character_allowed(c, adapter.profile) for c in normalized):
                ids = original.text_to_ids(text)
                remapped = [adapter.full_native_to_subset_native[i] for i in ids]
                if None not in remapped:
                    assert adapter.text_to_ids(text) == remapped
                    assert adapter.ids_to_text(adapter.text_to_ids(text)) == original.ids_to_text(ids)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in full.iterdir() if p.is_file()}


def test_control_ids_remap_but_other_trainer_metadata_stays_exact(full, tmp_path):
    proto = pb.ModelProto()
    proto.ParseFromString((full / "base-tokenizer.model").read_bytes())
    proto.trainer_spec.bos_id = len(proto.pieces)
    proto.pieces.add(piece="<s>", score=0, type=pb.ModelProto.SentencePiece.CONTROL)
    proto.trainer_spec.eos_id = len(proto.pieces)
    proto.pieces.add(piece="</s>", score=0, type=pb.ModelProto.SentencePiece.CONTROL)
    proto.trainer_spec.vocab_size = len(proto.pieces)
    proto.trainer_spec.ClearField("unk_id")  # default0 must stay implicit
    base = tmp_path / "control-base.model"
    base.write_bytes(proto.SerializeToString())
    selection = json.loads((full / "selection.json").read_text())
    selection["base_tokenizer_sha256"] = hashlib.sha256(base.read_bytes()).hexdigest()
    path = tmp_path / "control-selection.json"
    path.write_text(json.dumps(selection))
    source = tmp_path / "control-full"
    build_native_tokenizer(base, path, source)
    output = tmp_path / "control-subset"
    package_tokenizer_bundles(source, output, profiles=["latin"], make_zips=False)
    adapter = load_tokenizer_bundle(output / "latin")
    assert adapter.bos_id == adapter.token_to_id("<s>")
    assert adapter.eos_id == adapter.token_to_id("</s>")
    assert adapter.bos_id != proto.trainer_spec.bos_id
    assert adapter.backend.encode("ab", add_bos=True, add_eos=True) == [adapter.bos_id, *adapter.text_to_ids("ab"), adapter.eos_id]
    target = pb.ModelProto()
    target.ParseFromString(adapter.model_bytes)
    assert not target.trainer_spec.HasField("unk_id")
    source_model = pb.ModelProto()
    source_model.ParseFromString(adapter.full_model_bytes)
    for model in (target, source_model):
        model.ClearField("pieces")
        for field in ("vocab_size", "bos_id", "eos_id"):
            model.trainer_spec.ClearField(field)
    assert target.SerializeToString() == source_model.SerializeToString()


@pytest.fixture
def package_resources(full, tmp_path, monkeypatch):
    import untok.bundles as bundles

    package_root = tmp_path / "package"
    package_tokenizer_bundles(full, package_root / "data", profiles=("latin", "latin-indic", "full"), make_zips=False)
    monkeypatch.setattr(bundles.resources, "files", lambda package: package_root)
    return package_root


@pytest.mark.parametrize("profile", ["latin", "latin-indic", "full"])
def test_profile_names_load_the_same_verified_bundle(package_resources, profile):
    from untok.bundles import load_tokenizer

    named = load_tokenizer(profile)
    explicit = load_tokenizer(package_resources / "data" / profile)
    assert named.model_bytes == explicit.model_bytes
    assert named.id_map == explicit.id_map
    assert named.source_native_to_target_native == explicit.source_native_to_target_native
    assert named.text_to_ids("ab க") == explicit.text_to_ids("ab க")


def test_same_named_local_directory_requires_explicit_path(package_resources, full, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # A damaged local directory cannot shadow the trusted installed name.
    (full / "tokenizer.model").write_bytes(b"damaged local model")
    assert load_tokenizer_bundle("full").vocab_size > 0
    for explicit in (Path("full"), "./full"):
        with pytest.raises(ValueError, match="hash mismatch"):
            load_tokenizer_bundle(explicit)


@pytest.mark.parametrize("profile", ["latin", "latin-indic", "full"])
def test_packaged_profile_cannot_bypass_artifact_validation(package_resources, profile):
    path = package_resources / "data" / profile / "tokenizer.model"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_tokenizer_bundle(profile)


def test_missing_packaged_data_is_reported(package_resources):
    import shutil

    shutil.rmtree(package_resources / "data" / "latin")
    with pytest.raises(ValueError, match="Packaged tokenizer 'latin' is missing"):
        load_tokenizer_bundle("latin")


@pytest.mark.parametrize("profile", ["latin", "latin-indic", "full"])
def test_zipped_resources_survive_archive_and_temporary_directory_removal(
    package_resources, profile, tmp_path, monkeypatch,
):
    import shutil
    import untok.bundles as bundles

    expected = load_tokenizer_bundle(package_resources / "data" / profile)
    archive = tmp_path / "resources.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for path in (package_resources / "data" / profile).iterdir():
            stream.write(path, path.relative_to(package_resources).as_posix())
    shutil.rmtree(package_resources)
    materialized = []
    original_loader = bundles._load_tokenizer_directory

    def load_and_remember(directory):
        materialized.append(Path(directory))
        return original_loader(directory)

    monkeypatch.setattr(bundles, "_load_tokenizer_directory", load_and_remember)
    with zipfile.ZipFile(archive) as stream:
        monkeypatch.setattr(bundles.resources, "files", lambda package: zipfile.Path(stream))
        actual = load_tokenizer_bundle(profile)
    archive.unlink()
    assert len(materialized) == 1 and not materialized[0].exists()
    assert actual.model_bytes == expected.model_bytes
    assert actual.id_map == expected.id_map
    for text in ("ab", "a  b", "க", "a🙂b"):
        ids = actual.text_to_ids(text)
        assert ids == expected.text_to_ids(text)
        assert actual.ids_to_text(ids) == expected.ids_to_text(ids)


def test_zip_resource_manifest_rejects_path_traversal(package_resources, tmp_path, monkeypatch):
    import untok.bundles as bundles

    manifest = json.loads((package_resources / "data" / "latin" / "manifest.json").read_text())
    manifest["files"]["../outside.model"] = "0" * 64
    archive = tmp_path / "unsafe-resources.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("data/latin/manifest.json", json.dumps(manifest))
    with zipfile.ZipFile(archive) as stream:
        monkeypatch.setattr(bundles.resources, "files", lambda package: zipfile.Path(stream))
        with pytest.raises(ValueError, match="artifact filenames"):
            load_tokenizer_bundle("latin")


@pytest.mark.parametrize("profile,size", [("original", 13087), ("latin", 2653), ("latin-indic", 10372), ("full", 20360)])
def test_installed_profiles_include_the_frozen_candidate(profile, size):
    from untok.bundles import load_tokenizer

    expected_hashes = {
        "original": "ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291",
        "latin": "035d463b9906a291b3428d56f1622dc758bb05b388b4d66905e52b68d93ea714",
        "latin-indic": "e8035679586667af932d47d467fb9ff8696c8819fbcf49168633a6df20d67dfc",
        "full": "815ee2313f17db264eb681f4f5a1a322fbb05c1667d9808fb3fb357ae749b857",
    }
    tokenizer = load_tokenizer(profile)
    assert tokenizer.vocab_size == size
    assert hashlib.sha256(tokenizer.model_bytes).hexdigest() == expected_hashes[profile]
