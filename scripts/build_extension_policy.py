"""Reproduce Unicode coverage without changing any original tokenizer metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

BASE_HASH = "ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291"
FULL_HASH = "f987a99ce9448ca72bb2da11f36744254f9f9b12f5596fcb742ddedf950886a8"
UNICODE_HASH = "2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c"


def build(base: Path, full_path: Path, unicode_data: Path, output: Path):
    if output.exists():
        raise FileExistsError("Use a new policy output path")
    for path, expected in ((base, BASE_HASH), (full_path, FULL_HASH), (unicode_data, UNICODE_HASH)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source pin mismatch: {path.name}")
    model = pb.ModelProto()
    model.ParseFromString(full_path.read_bytes())
    normalizer = spm.SentencePieceNormalizer(model_proto=base.read_bytes(), add_dummy_prefix=False,
                                            remove_extra_whitespaces=False, escape_whitespaces=True)
    ucd = {int(row[0], 16): row for row in (line.split(";") for line in unicode_data.read_text().splitlines())}
    repertoire = {p.piece for p in model.pieces if p.type == pb.ModelProto.SentencePiece.NORMAL
                  and len(p.piece) == 1 and "LATIN" in ucd.get(ord(p.piece), ["", ""])[1]}
    seeds = len(repertoire)
    while True:
        expanded = set(repertoire)
        for char in repertoire:
            row = ucd.get(ord(char))
            if row:
                expanded.update(chr(int(row[i], 16)) for i in (12, 13, 14) if row[i])
                if row[5] and not row[5].startswith("<"):
                    expanded.update(chr(int(cp, 16)) for cp in row[5].split())
        if expanded == repertoire:
            break
        repertoire = expanded
    inventory = {p.piece for p in model.pieces}
    coverage = sorted(c for c in repertoire if c not in inventory and normalizer.normalize(c) == c)
    policy = {
        "schema_version": 1, "name": "untok-original-prefix-preserved-v3",
        "source_base_sha256": BASE_HASH, "source_full_sha256": FULL_HASH,
        "unicode_data_sha256": UNICODE_HASH,
        "coverage_policy": "Unicode 17 simple case and canonical decomposition closure of admitted single Latin pieces; normalization-stable missing scalars only",
        "coverage_seed_characters": seeds, "coverage_additions": coverage,
        "normalizer": "Exact original Nemotron normalizer; no spelling or joiner overrides",
        "original_piece_policy": "Original IDs, complete piece messages and model metadata remain unchanged",
        "profile_scope": "Filter additions only; original scripts are always retained",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--full-model", type=Path, required=True)
    parser.add_argument("--unicode-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.base, args.full_model, args.unicode_data, args.output)
    print(json.dumps({"coverage_additions": len(result["coverage_additions"]), "output": str(args.output)}))
