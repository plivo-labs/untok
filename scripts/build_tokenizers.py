"""Rebuild the four approved models without fitting or selecting new tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sentencepiece import sentencepiece_model_pb2 as pb

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/current-tokenizers.json"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def checked(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return raw


def build(output: Path, *, assets: Path = ROOT / "src/untok/data",
          base: Path | None = None, config_path: Path = CONFIG) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; shipped tokenizers are never overwritten")
    config = json.loads(config_path.read_text())
    selection = json.loads(checked(config_path.parent / config["selection_file"],
                                   config["selection_sha256"]))
    if len(selection["additions"]) != config["additions"]:
        raise ValueError("Approved addition count changed")
    # These frozen metadata files describe the already approved selection.
    # Checking their published hashes avoids retaining historical builders.
    files = {name: checked(assets / name, expected)
             for name, expected in config["shipped_files"].items()}
    base_raw = checked(base or assets / config["base_model"], config["base_model_sha256"])
    original = pb.ModelProto.FromString(base_raw)
    if len(original.pieces) != config["base_text_entries"]:
        raise ValueError("Original Nemotron vocabulary size changed")
    reports = {}
    for profile, spec in config["profiles"].items():
        model = pb.ModelProto()
        model.CopyFrom(original)
        for index in spec["inactive_ids"]:
            model.pieces[index].CopyFrom(pb.ModelProto.SentencePiece(
                piece=f"<unused_nemotron_{index}>", score=0.0,
                type=pb.ModelProto.SentencePiece.UNUSED))
        for piece, score, kind, _source_id in selection["additions"][:spec["additions"]]:
            model.pieces.add(piece=piece, score=score, type=kind)
        if profile != "original":
            model.trainer_spec.vocab_size = spec["text_entries"]
        if spec["inactive_ids"]:
            active = {p.piece for p in model.pieces if p.type != pb.ModelProto.SentencePiece.UNUSED}
            for field in ("user_defined_symbols", "control_symbols"):
                retained = [p for p in getattr(model.trainer_spec, field) if p in active]
                model.trainer_spec.ClearField(field)
                getattr(model.trainer_spec, field).extend(retained)
        raw = base_raw if profile == "original" else model.SerializeToString()
        if digest(raw) != spec["tokenizer_sha256"]:
            raise ValueError(f"Rebuilt {profile} differs from the published tokenizer")
        files[f"{profile}/tokenizer.model"] = raw
        reports[profile] = {"text_entries": len(model.pieces),
                             "active_entries": spec["active_entries"],
                             "blank_id": spec["blank_id"], "sha256": digest(raw)}
    for name, raw in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    notices = ROOT / "src/untok/notices"
    for profile in config["profiles"]:
        for notice in notices.rglob("*"):
            if notice.is_file():
                target = output / profile / notice.relative_to(notices)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(notice.read_bytes())
    return {"baseline_git_commit": config["baseline_git_commit"],
            "files_written": len(files), "profiles": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / "src/untok/data",
                        help="Existing approved metadata and source assets")
    parser.add_argument("--base", type=Path, help="Exact original Nemotron tokenizer.model")
    args = parser.parse_args()
    print(json.dumps(build(args.output, assets=args.assets, base=args.base), indent=2))


if __name__ == "__main__":
    main()
