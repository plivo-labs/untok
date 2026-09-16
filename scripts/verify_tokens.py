"""Check every published asset, token ID, score, type, mask and native blank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sentencepiece import sentencepiece_model_pb2 as pb

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/current-tokenizers.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_model(path: Path) -> pb.ModelProto:
    return pb.ModelProto.FromString(path.read_bytes())


def metadata(model: pb.ModelProto) -> bytes:
    result = pb.ModelProto()
    result.CopyFrom(model)
    result.ClearField("pieces")
    return result.SerializeToString()


def compare_models(actual: pb.ModelProto, expected: pb.ModelProto, label: str) -> None:
    require(len(actual.pieces) == len(expected.pieces), f"{label}: token count changed")
    for index, (piece, reference) in enumerate(zip(actual.pieces, expected.pieces)):
        require(piece.SerializeToString() == reference.SerializeToString(),
                f"{label}: token ID {index} spelling, score or type changed")
    require(metadata(actual) == metadata(expected),
            f"{label}: normalizer or model metadata changed")


def verify(data: Path, *, baseline_dir: Path | None = None,
           config_path: Path = CONFIG) -> dict:
    config = json.loads(config_path.read_text())
    expected_hashes = config["shipped_files"]
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    original = read_model(data / config["base_model"])
    require(len(original.pieces) == config["base_text_entries"], "Original token count changed")
    require(digest((data / config["base_model"]).read_bytes()) == config["base_model_sha256"],
            "Original Nemotron model changed")
    selection_raw = (config_path.parent / config["selection_file"]).read_bytes()
    require(digest(selection_raw) == config["selection_sha256"], "Approved selection changed")
    selection = json.loads(selection_raw)["additions"]
    require(len(selection) == config["additions"], "Approved addition count changed")
    results = {}
    for profile, spec in config["profiles"].items():
        directory = data / profile
        model = read_model(directory / "tokenizer.model")
        if baseline_dir is not None:
            compare_models(model, read_model(baseline_dir / profile / "tokenizer.model"), profile)
        manifest = json.loads((directory / "manifest.json").read_text())
        mapping = json.loads((directory / "native-row-map.json").read_text())
        inactive = spec["inactive_ids"]
        inactive_set = set(inactive)
        require(len(model.pieces) == spec["text_entries"], f"{profile}: token count changed")
        require(mapping["inactive_native_ids"] == inactive, f"{profile}: inactive IDs changed")
        require([i for i, p in enumerate(model.pieces) if p.type == pb.ModelProto.SentencePiece.UNUSED]
                == inactive, f"{profile}: inactive piece types changed")
        for index, expected in enumerate(original.pieces):
            piece = model.pieces[index]
            if index in inactive_set:
                require(piece.piece == f"<unused_nemotron_{index}>" and piece.score == 0.0,
                        f"{profile}: reserved ID {index} changed")
            else:
                require(piece.SerializeToString() == expected.SerializeToString(),
                        f"{profile}: original token ID {index} changed")
        additions = model.pieces[config["base_text_entries"]:]
        require(len(additions) == spec["additions"], f"{profile}: additions removed or inserted")
        for index, piece in enumerate(additions):
            require([piece.piece, piece.score, piece.type] == selection[index][:3],
                    f"{profile}: approved addition {index} changed")
        expected_metadata = pb.ModelProto()
        expected_metadata.CopyFrom(original)
        if profile != "original":
            expected_metadata.trainer_spec.vocab_size = spec["text_entries"]
        if inactive:
            active = {p.piece for p in model.pieces if p.type != pb.ModelProto.SentencePiece.UNUSED}
            for field in ("user_defined_symbols", "control_symbols"):
                retained = [p for p in getattr(expected_metadata.trainer_spec, field) if p in active]
                expected_metadata.trainer_spec.ClearField(field)
                getattr(expected_metadata.trainer_spec, field).extend(retained)
        require(metadata(model) == metadata(expected_metadata),
                f"{profile}: normalizer or model metadata changed")
        require(mapping["source_native_to_target_native"]
                == list(range(config["base_text_entries"])) + [spec["blank_id"]],
                f"{profile}: original IDs or blank relocation changed")
        require(mapping["source_blank_id"] == config["base_text_entries"]
                and mapping["target_blank_id"] == spec["blank_id"] == len(model.pieces),
                f"{profile}: native blank changed")
        for field, expected in (("native_vocabulary_size", spec["text_entries"]),
                                ("native_blank_id", spec["blank_id"]),
                                ("active_vocabulary_size", spec["active_entries"]),
                                ("inactive_native_slot_count", len(inactive))):
            require(manifest[field] == expected, f"{profile}: manifest {field} changed")
        require(len(model.pieces) - len(inactive) == spec["active_entries"],
                f"{profile}: active token count changed")
        results[profile] = {"text_entries": len(model.pieces),
                            "active_entries": spec["active_entries"],
                            "inactive_entries": len(inactive),
                            "added_entries": len(additions), "blank_id": spec["blank_id"]}
    # These are hashes captured from the immutable pre-cleanup commit, not
    # checksums generated from the candidate we are checking.
    for name, expected in expected_hashes.items():
        require(digest((data / name).read_bytes()) == expected, f"Published SHA256 mismatch: {name}")
        if baseline_dir is not None:
            require(digest((baseline_dir / name).read_bytes()) == expected,
                    f"Baseline is not the pinned published version: {name}")
    return {"passed": True, "baseline_git_commit": config["baseline_git_commit"],
            "assets_unchanged": len(expected_hashes), "profiles": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "src/untok/data")
    parser.add_argument("--baseline-dir", type=Path,
                        help="Independent pre-cleanup data directory, such as a git archive")
    parser.add_argument("--output", type=Path, help="Optional JSON verification receipt")
    args = parser.parse_args()
    report = json.dumps(verify(args.data, baseline_dir=args.baseline_dir), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
