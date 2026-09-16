"""Portable entry point for native source acquisition and experiment orchestration."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from untok.sources import fetch_sources, sha256

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
LANGUAGES = "hi mr brx doi kok mai ne sa sd as bn gu kn ml or pa ta te ks ur mni sat".split()
JOBS = "deva bengali gu kn ml or pa ta te arabic mni sat".split()


def contained(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Path must be relative and contained")
    path = root.resolve() / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Symlink escapes input root")
    return path


def verify_contract(root, manifest):
    data = json.loads(manifest.read_text())
    files = data.get("files") or {i["path"]: i["sha256"] for i in data["inputs"]}
    for name, expected in files.items():
        if sha256(contained(root, name)) != expected:
            raise ValueError("Input hash mismatch: " + name)
    return {"passed": True, "files": len(files)}


def intake_contract(root):
    inputs = []
    for lang in LANGUAGES:
        for role, folder, kind in [("pending", "primary-intake/train", "speech"),
                                    ("reserve", "primary-intake/reserve", "speech")]:
            name = f"{folder}/{lang}.jsonl"
            inputs.append(dict(path=name, sha256=sha256(contained(root, name)), kind=kind, role=role))
        for role, suffix in [("pending", "train"), ("dev", "dev")]:
            name = f"written/{lang}-{suffix}.jsonl"
            inputs.append(dict(path=name, sha256=sha256(contained(root, name)), kind="written", role=role))
    for filename in ("meta-train.jsonl", "urdu-nonbenchmark.jsonl", "spring-r1-staged.jsonl"):
        name = "independent-intake/" + filename
        inputs.append(dict(path=name, sha256=sha256(contained(root, name)), kind="speech", role="pending"))
    return dict(schema_version=2, historical_byte_replay=False, inputs=inputs)


def extract_base(checkpoint, output, lock):
    source = json.loads(lock.read_text())["native_base"]
    if sha256(checkpoint) != source["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 differs from pinned NVIDIA release")
    if output.exists():
        raise FileExistsError(output)
    # Read matching members; never extract arbitrary archive paths to disk.
    with tarfile.open(checkpoint) as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith("tokenizer.model") and member.size < 20_000_000:
                with archive.extractfile(member) as stream:
                    data = stream.read()
                if hashlib.sha256(data).hexdigest() == source["tokenizer_sha256"]:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(data)
                    return {"tokenizer_sha256": source["tokenizer_sha256"]}
    raise ValueError("Pinned native tokenizer not present in checkpoint")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--cache", type=Path, required=True)
    fetch.add_argument("--lock", type=Path, default=REPO / "configs/native-study-sources.lock.json")
    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    extract = sub.add_parser("extract-base")
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--lock", type=Path, default=REPO / "configs/native-study-sources.lock.json")
    fit = sub.add_parser("fit-all")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--accept-data-sha256", required=True)
    fit.add_argument("--base", type=Path, required=True)
    fit.add_argument("--donors", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--jobs", nargs="+", choices=JOBS)
    fit.add_argument("--recipe", type=Path, default=REPO / "configs/native-study-recipe.json")
    args = p.parse_args()
    if args.command == "fetch":
        result = fetch_sources(args.lock, args.cache)
    elif args.command == "verify":
        result = verify_contract(args.root, args.manifest)
    elif args.command == "extract-base":
        result = extract_base(args.checkpoint, args.output, args.lock)
    elif args.command == "manifest":
        if args.output.exists():
            raise FileExistsError(args.output)
        result = intake_contract(args.root)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        result = {"files": len(result["inputs"]), "manifest_sha256": sha256(args.output)}
    else:
        if sha256(args.data / "manifest.json") != args.accept_data_sha256:
            raise ValueError("Explicit accepted corpus hash differs")
        recipe = json.loads(args.recipe.read_text())
        jobs = args.jobs or recipe['jobs']
        if not jobs or not set(jobs) <= set(JOBS):
            raise ValueError("Unsupported or empty fitting job list")
        options = []
        for field in ('fractions', 'seeds', 'families', 'mass_multipliers'):
            options += ['--' + field.replace('_', '-'), *map(str, recipe[field])]
        options += ['--em-passes', str(recipe['em_passes'])]
        for job in jobs:
            subprocess.run([sys.executable, str(SCRIPTS / "fit.py"), "--job", job,
                            "--data-dir", str(args.data), "--accept-data-sha256", args.accept_data_sha256,
                            "--base", str(args.base), "--donors", str(args.donors),
                            "--output-root", str(args.output), *options], check=True)
        result = {"completed_jobs": jobs, "recipe_sha256": sha256(args.recipe)}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
