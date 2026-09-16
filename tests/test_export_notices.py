"""Standalone exports keep origin notices without changing model artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from untok.bundles import deterministic_bundle_zip, load_tokenizer_bundle, package_tokenizer_bundles
from untok.clean import build_clean_bundles
from untok.export_notices import export_notice_files, write_export_notices
from untok.unigram import build_native_tokenizer


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src/untok/data"


def artifact_files(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    return {name: (directory / name).read_bytes() for name in ["manifest.json", *manifest["files"]]}


def assert_notices(directory):
    for name, content in export_notice_files().items():
        assert (directory / name).read_bytes() == content


def test_packaged_notices_equal_the_published_origin_and_license_texts():
    expected = {"THIRD_PARTY.md": (REPO / "THIRD_PARTY.md").read_bytes()}
    expected.update({"licenses/" + path.name: path.read_bytes() for path in (REPO / "licenses").glob("*.txt")})
    assert export_notice_files() == expected
    assert "licenses/OpenMDW-1.1.txt" in expected


def test_native_build_exports_notices_outside_its_model_integrity_manifest(tmp_path):
    source = DATA / "source"
    output = tmp_path / "native"
    manifest = build_native_tokenizer(source / "base-tokenizer.model", source / "selection.json", output)
    assert_notices(output)
    assert not set(manifest["files"]) & set(export_notice_files())
    assert (output / "tokenizer.model").read_bytes() == (source / "tokenizer.model").read_bytes()
    assert artifact_files(output) == artifact_files(source)
    assert load_tokenizer_bundle(output).source_native_to_target_native[:13087] == tuple(range(13087))


@pytest.mark.parametrize("profile", ["original", "full", "latin", "latin-indic"])
def test_direct_zip_adds_notices_to_old_bundles_without_modifying_source(tmp_path, profile):
    source = DATA / profile
    before = artifact_files(source)
    archive = tmp_path / "export.zip"
    deterministic_bundle_zip(source, archive)
    with zipfile.ZipFile(archive) as stream:
        assert stream.namelist() == sorted(stream.namelist())
        assert len(stream.namelist()) == len(set(stream.namelist()))
        for name, content in {**before, **export_notice_files()}.items():
            assert stream.read(name) == content
        stream.extractall(tmp_path / "extracted")
    assert artifact_files(source) == before
    assert load_tokenizer_bundle(tmp_path / "extracted").model_bytes == before["tokenizer.model"]


def test_clean_and_package_preserve_additional_source_attribution(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(DATA / "full", source)
    additional = {"NOTICE.txt": b"Additional origin notice\n", "licenses/Additional.txt": b"Additional source terms\n",
                  "LICENSE.md": b"Additional license terms\n", "COPYRIGHT": b"Additional copyright holder\n"}
    for name, content in additional.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    before = artifact_files(source)
    clean = tmp_path / "clean"
    build_clean_bundles(source, clean)
    packaged = tmp_path / "packaged"
    receipt = package_tokenizer_bundles(source, packaged)
    for profile in ("original", "full", "latin", "latin-indic"):
        for directory in (clean / profile, packaged / profile):
            assert_notices(directory)
            assert artifact_files(directory) == artifact_files(DATA / profile)
            for name, content in additional.items():
                assert (directory / name).read_bytes() == content
        assert set(export_notice_files()) <= set(receipt["bundles"][profile]["files"])
        with zipfile.ZipFile(packaged / (profile + ".zip")) as stream:
            for name, content in {**export_notice_files(), **additional}.items():
                assert stream.read(name) == content
    assert artifact_files(source) == before


def test_conflicting_notices_are_never_overwritten_or_zipped(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    notice = source / "THIRD_PARTY.md"
    notice.write_bytes(b"Existing attribution that must not be lost\n")
    with pytest.raises(ValueError, match="conflicts with packaged attribution"):
        write_export_notices(source)
    archive = tmp_path / "conflict.zip"
    with pytest.raises(ValueError, match="conflicts with packaged attribution"):
        deterministic_bundle_zip(source, archive)
    assert notice.read_bytes() == b"Existing attribution that must not be lost\n"
    assert not archive.exists()


def test_notice_writer_rejects_licenses_symlink(tmp_path):
    source, outside = tmp_path / "source", tmp_path / "outside"
    source.mkdir()
    outside.mkdir()
    (source / "licenses").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        write_export_notices(source)
    assert not list(outside.iterdir())


@pytest.mark.parametrize("name", ["third_party.md", "LICENSES/openmdw-1.1.txt"])
def test_case_insensitive_notice_collision_cannot_overwrite_attribution(tmp_path, name):
    source = tmp_path / "export"
    source.mkdir()
    with pytest.raises(ValueError, match="conflicts with packaged attribution"):
        write_export_notices(source, {name: b"Original origin notice that must be preserved\n"})
    assert not list(source.iterdir())


def test_wheel_resources_support_native_clean_package_and_direct_zip_exports(tmp_path):
    pytest.importorskip("setuptools")
    staging = tmp_path / "wheel-source"
    staging.mkdir()
    for name in ("pyproject.toml", "MANIFEST.in", "README.md", "THIRD_PARTY.md", "LICENSE", "NOTICE"):
        if (REPO / name).is_file():
            shutil.copy2(REPO / name, staging / name)
    for name in ("src", "licenses"):
        shutil.copytree(REPO / name, staging / name,
                        ignore=shutil.ignore_patterns("*.egg-info", "__pycache__", "*.pyc"))
    subprocess.run([sys.executable, "-c",
                    "from setuptools.build_meta import build_wheel; build_wheel('dist')"],
                   cwd=staging, check=True, capture_output=True, text=True)
    wheel = next((staging / "dist").glob("*.whl"))
    # Import straight from the built wheel in an isolated interpreter. This
    # also proves notices do not rely on a repository root or filesystem-only
    # resource APIs; the runtime resources are ZipPath objects here.
    script = r'''
import json, sys, zipfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import untok
assert untok.__file__.startswith(sys.argv[1])
from importlib import resources
from untok.bundles import package_tokenizer_bundles, deterministic_bundle_zip
from untok.clean import build_clean_bundles
from untok.export_notices import export_notice_files
from untok.unigram import build_native_tokenizer
root = Path(sys.argv[2]); root.mkdir()
data = resources.files('untok').joinpath('data')
source = root / 'source'; source.mkdir()
for entry in data.joinpath('source').iterdir():
    if entry.is_file(): (source / entry.name).write_bytes(entry.read_bytes())
native = root / 'native'
build_native_tokenizer(source / 'base-tokenizer.model', source / 'selection.json', native)
build_clean_bundles(native, root / 'clean')
package_tokenizer_bundles(root / 'clean/full', root / 'package', profiles=('full',))
deterministic_bundle_zip(source, root / 'direct.zip')
for directory in (native, root / 'clean/original', root / 'clean/latin', root / 'clean/latin-indic', root / 'clean/full', root / 'package/full'):
    for name, content in export_notice_files().items():
        assert (directory / name).read_bytes() == content
for profile in ('original', 'latin', 'latin-indic', 'full'):
    expected = data.joinpath(profile)
    manifest = json.loads(expected.joinpath('manifest.json').read_text())
    for name in ('manifest.json', *manifest['files']):
        assert (root / 'clean' / profile / name).read_bytes() == expected.joinpath(name).read_bytes()
for archive in (root / 'direct.zip', root / 'package/full.zip'):
    with zipfile.ZipFile(archive) as stream:
        for name, content in export_notice_files().items(): assert stream.read(name) == content
'''
    subprocess.run([sys.executable, "-I", "-c", script, str(wheel), str(tmp_path / "exports")],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
