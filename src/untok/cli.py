"""Native SentencePiece Unigram tokenizer and checkpoint commands."""
from __future__ import annotations

import argparse
import json
import sys

from .sources import write_json


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="untok",
        description="Build and validate native SentencePiece Unigram tokenizer bundles for Nemotron.",
        epilog="Checkpoint migration requires a compatible NVIDIA NeMo runtime.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", aliases=["build-unigram"],
        help="Package fitted additions against a hash-pinned native Unigram model",
    )
    build.set_defaults(operation="build")
    build.add_argument("--base", required=True, help="Original SentencePiece model from the native checkpoint")
    build.add_argument("--selection", required=True, help="Ordered, scored additions and native base SHA256")
    build.add_argument("--output", required=True, help="Empty directory for a separate native candidate bundle")
    clean = commands.add_parser("clean", help="Rebuild original, Latin, Latin plus Indic and full tokenizer profiles")
    clean.set_defaults(operation="clean")
    clean.add_argument("--bundle", required=True, help="Original full bundle or a preserved bundle containing that source")
    clean.add_argument("--output", required=True, help="New or empty directory for all four tokenizer profiles")
    check = commands.add_parser(
        "check", aliases=["check-unigram"],
        help="Verify native bundle hashes, vocabulary and ID mapping on CPU",
    )
    check.set_defaults(operation="check")
    check.add_argument("--bundle", required=True)
    package = commands.add_parser("package", help="Export original, Latin, Latin plus Indic and full Unigram bundles")
    package.set_defaults(operation="package")
    package.add_argument("--bundle", required=True, help="Validated full native Unigram bundle")
    package.add_argument("--output", required=True, help="New destination for bundles and reproducible ZIPs")
    package.add_argument("--profiles", nargs="+", choices=["original", "latin", "latin-indic", "full"],
                         default=["original", "latin", "latin-indic", "full"])
    migrate = commands.add_parser("migrate", help="Create and verify a matching native NeMo checkpoint")
    migrate.set_defaults(operation="migrate")
    migrate.add_argument("--source", required=True, help="Pinned native base or original full Untok v1 .nemo checkpoint")
    migrate.add_argument("--bundle", required=True, help="Full or reduced Unigram bundle")
    migrate.add_argument("--source-sha256", required=True, help="Expected source checkpoint SHA256")
    migrate.add_argument("--output", required=True, help="New .nemo destination")
    migrate.add_argument("--seed", type=int, default=0)
    migrate.add_argument("--max-new-mass-ratio", type=float, default=0.05,
                         help="Initial text-donor added-output mass bound (default: 0.05)")
    migrate.add_argument("--model-config", help="YAML native model settings applied before model construction")
    migrate.add_argument("--training-template", help="Native NVIDIA training YAML used to write a .train.yaml sidecar")
    migrate.add_argument("--training-overrides", help="YAML overrides for --training-template")
    validate = commands.add_parser(
        "validate", aliases=["validate-unigram"],
        help="Validate the full native candidate and pinned text corpora on CPU",
    )
    validate.set_defaults(operation="validate")
    validate.add_argument("--bundle", required=True)
    validate.add_argument("--policy", required=True, help="Explicit native validation policy JSON")
    validate.add_argument("--corpora", help="Frozen corpus manifest; omitted inputs mean incomplete")
    validate.add_argument("--phase", choices=["dev", "reserve"], default="dev")
    validate.add_argument("--selection-receipt", help="Required artifact receipt for reserve evaluation")
    validate.add_argument("--max-examples", type=int, default=0, help="Dev text examples only; reserve always omits text")
    validate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        status = 0
        if args.operation == "build":
            from .unigram import build_native_tokenizer

            result = build_native_tokenizer(args.base, args.selection, args.output)
        elif args.operation == "clean":
            from .clean import build_clean_bundles

            result = build_clean_bundles(args.bundle, args.output)
        elif args.operation == "check":
            from .bundles import load_tokenizer_bundle

            adapter = load_tokenizer_bundle(args.bundle)
            result = {"structural_passed": True, "native_vocabulary_size": adapter.vocab_size,
                      "native_blank_id": adapter.blank_id, "tokenizer_sha256": adapter.id_map.tokenizer_sha256,
                      "checkpoint_validated": False, "asr_validated": False}
        elif args.operation == "package":
            from .bundles import package_tokenizer_bundles

            result = package_tokenizer_bundles(args.bundle, args.output, profiles=tuple(args.profiles))
        elif args.operation == "migrate":
            from .native_checkpoint import migrate_native_checkpoint

            result = migrate_native_checkpoint(args.source, args.bundle, args.output,
                                               expected_source_sha256=args.source_sha256, seed=args.seed,
                                               max_new_mass_ratio=args.max_new_mass_ratio,
                                               model_config=args.model_config,
                                               training_template=args.training_template,
                                               training_overrides=args.training_overrides)
        elif args.operation == "validate":
            from .unigram_validation import validate_native_tokenizer

            report = validate_native_tokenizer(args.bundle, args.policy, args.corpora, phase=args.phase,
                                               selection_receipt_path=args.selection_receipt,
                                               max_examples=args.max_examples)
            write_json(args.output, report)
            status = 0 if report["passed"] else 2
            result = {"status": report["status"], "structural_passed": report["structural_passed"],
                      "corpus_status": report["corpus_status"], "report": args.output,
                      "checkpoint_validated": False, "asr_validated": False}
            for field in ("selection_quality_status", "selection_quality_passed"):
                if field in report:
                    result[field] = report[field]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return status
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"untok: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
