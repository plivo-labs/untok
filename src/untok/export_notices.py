"""Legal sidecars for standalone exports, independent of model integrity files."""
from __future__ import annotations

from importlib import resources
from pathlib import Path


def export_notice_files() -> dict[str, bytes]:
    """Read distribution-owned notices from filesystem or zipped package data."""
    root = resources.files("untok").joinpath("notices")
    notices = {"THIRD_PARTY.md": root.joinpath("THIRD_PARTY.md").read_bytes()}
    licenses = root.joinpath("licenses")
    for entry in sorted(licenses.iterdir(), key=lambda item: item.name):
        if entry.is_file() and entry.name.endswith(".txt"):
            notices[f"licenses/{entry.name}"] = entry.read_bytes()
    if "licenses/OpenMDW-1.1.txt" not in notices:
        raise ValueError("Packaged tokenizer export notices are missing OpenMDW 1.1")
    return notices


def with_export_notices(files: dict[str, bytes]) -> dict[str, bytes]:
    """Add notices without silently replacing a caller's attribution text."""
    result = dict(files)
    folded = {}
    for name in result:
        key = name.casefold()
        if key in folded:
            raise ValueError(f"Bundle export conflicts with attribution path casing: {name}")
        folded[key] = name
    for name, content in export_notice_files().items():
        existing = folded.get(name.casefold())
        if existing is not None and result[existing] != content:
            raise ValueError(f"Bundle export conflicts with packaged attribution: {name}")
        if existing is not None and existing != name:
            del result[existing]
        result[name] = content
        folded[name.casefold()] = name
    return result


def source_notice_files(directory: str | Path) -> dict[str, bytes]:
    """Keep additional notices supplied with a source bundle during packaging."""
    directory = Path(directory)
    paths = []
    for entry in directory.iterdir():
        name = entry.name.casefold()
        if any(name == prefix or name.startswith((prefix + ".", prefix + "-"))
               for prefix in ("third_party", "license", "licence", "notice", "copying", "copyright", "authors")):
            paths.append(entry)
        if name in {"licenses", "licences"}:
            paths.append(entry)
            if not entry.is_symlink():
                paths.extend(entry.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Source bundle notices must not be symlinks")
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in paths if path.is_file()}


def write_export_notices(directory: str | Path, additional: dict[str, bytes] | None = None) -> None:
    """Write legal sidecars without changing tokenizer models or their manifests."""
    directory = Path(directory)
    notices = with_export_notices(additional or {})
    for name, content in notices.items():
        target = directory / name
        if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent.is_relative_to(directory)):
            raise ValueError("Tokenizer export notices cannot follow symlinks")
        if target.exists() and (not target.is_file() or target.read_bytes() != content):
            raise ValueError(f"Bundle export conflicts with packaged attribution: {name}")
    for name, content in notices.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
