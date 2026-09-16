"""Portable native-study contracts and annotation lineage regressions."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

REPO = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location("native_study_" + name, REPO / "scripts/native_study" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def proc():
    return spm.SentencePieceProcessor(model_file=str(REPO / "src/untok/data/full/base-tokenizer.model"))


def policy():
    return json.loads((REPO / "configs/native-study-annotations.json").read_text())


def test_meta_already_cleaned_intake_retains_partial_transcript_status(proc):
    freeze = module("freeze")
    row = dict(language="brx", source="facebook/omnilingual-asr-corpus", record_id="synthetic",
               text="अ  आ  इ", raw_text="अ<noise> आ<hesitation> इ", speaker_id="one", prompt_id="one")
    first = freeze.prepare_row(row, "speech", "pending", "meta.jsonl", 1, proc, policy())
    assert first["text"] == row["text"]
    assert first["raw_text"] == row["raw_text"]
    assert first["tokenizer_fit_only_where_annotations_removed"] is True
    assert first["marker_removed_count"] == 2
    assert first["source_annotation_replacements"] == {"<noise>": 1, "<hesitation>": 1}
    second = freeze.prepare_row(first, "speech", "pending", "meta.jsonl", 1, proc, policy())
    assert second["text"] == first["text"]
    assert second["marker_removed_count"] == 2


def test_successive_cleanup_keeps_prior_counts_and_flag(proc):
    freeze = module("freeze")
    row = dict(language="hi", source="custom", text="अ<new>आ", raw_text="अ<old><new>आ",
               source_annotation_replacements={"<old>": 1}, marker_removed_count=1,
               tokenizer_fit_only_where_annotations_removed=True)
    result = freeze.prepare_row(row, "speech", "pending", "input.jsonl", 1, proc,
                                {"sources": {"custom": {"replace_with_space": ["<new>"]}}})
    assert result["source_annotation_replacements"] == {"<old>": 1, "<new>": 1}
    assert result["marker_removed_count"] == 2
    assert result["raw_text"] == row["raw_text"]
    assert result["tokenizer_fit_only_where_annotations_removed"]


def test_previous_partial_flag_without_current_markers_is_preserved(proc):
    result = module("freeze").prepare_row(dict(language="hi", source="custom", text="अ आ",
        tokenizer_fit_only_where_annotations_removed=True), "speech", "pending", "rows.jsonl", 1, proc, {})
    assert result["tokenizer_fit_only_where_annotations_removed"]


def test_input_row_identity_is_independent_of_workspace_root(tmp_path, proc):
    freeze = module("freeze")
    identities = []
    for name in ("left", "right"):
        root = tmp_path / name
        root.mkdir()
        path = root / "rows.jsonl"
        path.write_text(json.dumps(dict(language="hi", source="custom", text="नमस्ते", record_id="one")) + "\n")
        rows, _, _ = freeze.load_inputs([(path, "speech", "pending")], proc, {}, input_root=root)
        identities.append((rows[0]["_uid"], rows[0]["input_file"]))
    assert identities[0] == identities[1]
    assert identities[0][1] == "rows.jsonl"


def test_contract_rejects_tampering_and_escape(tmp_path):
    runner = module("run")
    artifact = tmp_path / "input.jsonl"
    artifact.write_text("synthetic\n")
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"files": {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()}}))
    assert runner.verify_contract(tmp_path, contract) == {"passed": True, "files": 1}
    artifact.write_text("changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.verify_contract(tmp_path, contract)
    with pytest.raises(ValueError, match="relative"):
        runner.contained(tmp_path, "../outside")
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        runner.contained(tmp_path, "link/escape")


def test_native_donor_lock_enforced_and_control_pieces_not_imported(tmp_path):
    fit = module("fit")
    model = pb.ModelProto()
    model.pieces.add(piece="अ", score=-1, type=1)
    model.pieces.add(piece="<control>", score=0, type=3)
    files = []
    for name in ("hi.model", "IndicBART.model", "IndicBARTSS.model"):
        path = tmp_path / name
        path.write_bytes(model.SerializeToString())
        files.append(dict(path="donors/" + name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"files": files}))
    fit.DONORS = tmp_path
    fit.SOURCE_LOCK = lock
    entries, pins = fit.donor_strings(["hi"])
    assert set(entries) == {"अ"}
    assert len(pins) == 3
    (tmp_path / "hi.model").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source lock"):
        fit.donor_strings(["hi"])


def test_verified_checkpoint_extraction_never_extracts_archive_paths(tmp_path):
    runner = module("run")
    checkpoint = tmp_path / "source.nemo"
    payload = b"synthetic-tokenizer-model"
    with tarfile.open(checkpoint, "w") as archive:
        info = tarfile.TarInfo("../../tokenizer.model")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"native_base": {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                               "tokenizer_sha256": hashlib.sha256(payload).hexdigest()}}))
    output = tmp_path / "safe" / "base.model"
    runner.extract_base(checkpoint, output, lock)
    assert output.read_bytes() == payload
    assert not (tmp_path.parent / "tokenizer.model").exists()
    with pytest.raises(FileExistsError):
        runner.extract_base(checkpoint, output, lock)


def test_published_contracts_have_no_historical_workspace_paths_or_corpus_text():
    for path in (REPO / "configs").glob("native-study*.json"):
        text = path.read_text()
        assert "/Users/" not in text and "/tmp/taskresearch" not in text
        data = json.loads(text)
        assert data
    historical = json.loads((REPO / "configs/native-study-historical.json").read_text())
    assert len(historical["files"]) == 67
    assert sum(n for p, n in historical["row_counts"].items() if p.startswith("train/")) == 280471
    assert historical["historical_cleanup_metadata_version"] == 1


def test_written_document_roles_are_stable():
    written = module("intake_written")
    assert written.role_for("same-document") == written.role_for("same-document")
    assert {written.role_for(str(i)) for i in range(100)} == {"pending", "dev"}


def test_freeze_is_immutable_and_preserves_role_separation_across_roots(tmp_path):
    freeze = module("freeze")
    manifests = []
    for folder in ("left", "right"):
        root = tmp_path / folder
        root.mkdir()
        path, reserve = root / "input.jsonl", root / "reserve.jsonl"
        records = [dict(id=str(i), text="हिन्दी पाठ " + str(i), language="hi", source=freeze.IV,
                        speaker_group=str(i)) for i in range(20)]
        records.append(dict(id="exclude", text="बेंचमार्क", language="hi", source="Flores-200"))
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        reserve.write_text(json.dumps(dict(id="reserved", text="Completely separate synthetic reserve text",
                                           language="hi", source=freeze.IV, speaker_group="reserved")) + "\n")
        specs = [(path, "speech", "pending"), (reserve, "speech", "reserve")]
        base = REPO / "src/untok/data/full/base-tokenizer.model"
        out = root / "frozen"
        result = freeze.freeze(specs, out, base=base, langs=["hi"], input_root=root)
        assert result["row_counts"]["train/hi.jsonl"] > 0
        assert result["row_counts"]["dev/hi.jsonl"] > 0
        assert result["row_counts"]["reserve/hi.jsonl"] == 1
        assert result["early_exclusions"]["FLORES_all_roles_excluded"] == 1
        assert result["historical_freeze_replay"] is False
        assert result["cleanup_metadata_version"] == 2
        assert module("run").verify_contract(out, out / "manifest.json")["passed"]
        with pytest.raises(FileExistsError):
            freeze.freeze(specs, out, base=base, langs=["hi"])
        manifests.append(result["files"])
    assert manifests[0] == manifests[1]


def test_cached_independent_intake_requires_matching_receipt(tmp_path):
    intake = module("intake_independent")
    intake.CACHE = tmp_path
    intake.ROOT = tmp_path
    intake.CACHED_ONLY = True
    url = "https://example.test/public-source"
    path = tmp_path / (intake.sha(url.encode()) + ".bin")
    payload = b'{"synthetic":true}'
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="integrity receipt"):
        intake.fetch(url)
    intake.CACHE_EXPECTED = {url: intake.sha(payload)}
    assert intake.fetch(url) == payload
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity receipt"):
        intake.fetch(url)


def test_sdist_contains_and_loads_the_required_native_fit_inputs(tmp_path):
    pytest.importorskip("setuptools")
    # Build in a clean copy, so an existing egg-info/SOURCES.txt cannot conceal
    # missing MANIFEST rules or add artifacts absent from a fresh checkout.
    staging = tmp_path / "source"
    staging.mkdir()
    for name in ("pyproject.toml", "MANIFEST.in", "README.md", "THIRD_PARTY.md"):
        shutil.copy2(REPO / name, staging / name)
    for name in ("src", "configs", "scripts", "docs", "licenses"):
        shutil.copytree(REPO / name, staging / name,
                        ignore=shutil.ignore_patterns("*.egg-info", "__pycache__", "*.pyc"))
    subprocess.run([sys.executable, "-c",
                    "from setuptools.build_meta import build_sdist; build_sdist('dist')"],
                   cwd=staging, check=True, capture_output=True, text=True)
    archive = next((staging / "dist").glob("*.tar.gz"))
    extracted = tmp_path / "extracted"
    with tarfile.open(archive) as stream:
        stream.extractall(extracted, filter="data")
    root = next(extracted.iterdir())
    cfg = json.loads((root / "configs/build.json").read_text())
    for filename, digest in (("approved-hindi-103.tsv", cfg["hindi_sha256"]),
                             ("approved-rare-latin-190.tsv", cfg["latin_sha256"])):
        assert hashlib.sha256((root / "configs" / filename).read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location("sdist_native_fit", root / "scripts/native_study/fit.py")
    fit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit)
    protected, _, _ = fit.group_alphabets(cfg, None)
    hindi = {line.split("\t")[0] for line in (root / "configs/approved-hindi-103.tsv").read_text().splitlines()}
    assert len(hindi) == 103
    assert hindi <= protected["deva"]
