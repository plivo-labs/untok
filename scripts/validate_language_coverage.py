#!/usr/bin/env python3
"""Freeze pinned FLEURS transcript metadata, then report per-language coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from untok.language_validation import prepare_fleurs_corpus, validate_language_coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("configs/language-coverage-v5.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download pinned TSV metadata only; no audio")
    fetch.add_argument("--cache", type=Path, default=Path(".cache/language-coverage/fleurs"))
    fetch.add_argument("--output", type=Path, default=Path(".cache/language-coverage/prepared-v5"))
    fetch.add_argument("--offline", action="store_true", help="Use already hash-verified TSV files")
    validate = sub.add_parser("validate", help="Evaluate all four installed bundles")
    validate.add_argument("--corpora", type=Path, default=Path(".cache/language-coverage/prepared-v5/manifest.json"))
    validate.add_argument("--frozen-indic-manifest", type=Path)
    validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        result = prepare_fleurs_corpus(args.policy, args.cache, args.output, download=not args.offline)
        print(json.dumps({"prepared_languages": len(result["languages"]), "manifest": str(args.output / "manifest.json")}))
        return 0
    result = validate_language_coverage(args.policy, args.corpora, args.frozen_indic_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "missing_profiles", "coverage_gap_profiles", "roundtrip_or_normalizer_failure_profiles")}))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
