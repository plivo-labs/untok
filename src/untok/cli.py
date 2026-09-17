"""Inspect current bundles or migrate the native Nemotron checkpoint."""
import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="untok")
    commands = parser.add_subparsers(dest="operation", required=True)
    check = commands.add_parser("check", help="Verify a current tokenizer bundle")
    check.add_argument("--bundle", required=True)
    migrate = commands.add_parser("migrate", help="Initialize Nemotron with a current bundle")
    for name in ("source", "bundle", "source-sha256", "output"):
        migrate.add_argument("--" + name, required=True)
    migrate.add_argument("--seed", type=int, default=0)
    migrate.add_argument("--max-new-mass-ratio", type=float, default=0.05)
    args = parser.parse_args(argv)
    try:
        if args.operation == "check":
            from .bundles import load_tokenizer
            adapter = load_tokenizer(args.bundle)
            result = {"profile": adapter.profile, "structural_passed": True,
                      "native_vocabulary_size": adapter.vocab_size,
                      "active_vocabulary_size": adapter.active_vocab_size,
                      "native_blank_id": adapter.blank_id, "tokenizer_sha256": adapter.tokenizer_sha256}
        else:
            from .native_checkpoint import migrate_native_checkpoint
            result = migrate_native_checkpoint(args.source, args.bundle, args.output,
                expected_source_sha256=args.source_sha256, seed=args.seed,
                max_new_mass_ratio=args.max_new_mass_ratio)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"untok: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
