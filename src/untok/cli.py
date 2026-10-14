"""Inspect bundles, migrate Nemotron, and prepare Indic languages for training."""
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
    init = commands.add_parser("init-languages", help="Start prompt slots and untrained rows for Indic languages")
    for name in ("source", "source-sha256", "output"):
        init.add_argument("--" + name, required=True)
    init.add_argument("--bundle", default="latin-indic")
    init.add_argument("--languages", default="all", help="'all' (default) or codes/locales, e.g. ta,te,bn-IN")
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--force-prompt-slots", action="store_true", help="also overwrite slots that look trained")
    calibrate = commands.add_parser("calibrate", help="Fit the initialised rows' token offsets from audio (required)")
    for name in ("source", "source-sha256", "output"):
        calibrate.add_argument("--" + name, required=True)
    calibrate.add_argument("--manifest", action="append", required=True, metavar="LOCALE=PATH",
                           help="training manifest per language, e.g. ta-IN=ta.jsonl (repeat)")
    calibrate.add_argument("--bundle", default="latin-indic")
    calibrate.add_argument("--hours", type=float, default=12.0)
    calibrate.add_argument("--epochs", type=int, default=15)
    calibrate.add_argument("--seed", type=int, default=20260928)
    calibrate.add_argument("--device", default="cuda")
    normalize = commands.add_parser("normalize-manifest", help="Apply the training-text policy to a manifest")
    for name in ("input", "output"):
        normalize.add_argument("--" + name, required=True)
    normalize.add_argument("--lang-field", default="target_lang")
    join = commands.add_parser("join-clips", help="Write joined two-language clips for auto training")
    join.add_argument("--manifest", action="append", required=True)
    join.add_argument("--output-dir", required=True)
    join.add_argument("--pairs", type=int, required=True)
    join.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args(argv)
    try:
        if args.operation == "check":
            from .bundles import load_tokenizer
            adapter = load_tokenizer(args.bundle)
            result = {"profile": adapter.profile, "structural_passed": True,
                      "native_vocabulary_size": adapter.vocab_size,
                      "active_vocabulary_size": adapter.active_vocab_size,
                      "native_blank_id": adapter.blank_id, "tokenizer_sha256": adapter.tokenizer_sha256}
        elif args.operation == "migrate":
            from .native_checkpoint import migrate_native_checkpoint
            result = migrate_native_checkpoint(args.source, args.bundle, args.output,
                expected_source_sha256=args.source_sha256, seed=args.seed,
                max_new_mass_ratio=args.max_new_mass_ratio)
        elif args.operation == "init-languages":
            from .language_init import init_languages
            result = init_languages(args.source, args.output, source_sha256=args.source_sha256,
                                    languages=args.languages, bundle=args.bundle, seed=args.seed,
                                    force_prompt_slots=args.force_prompt_slots)
        elif args.operation == "calibrate":
            from .calibration import calibrate
            manifests = {}
            for item in args.manifest:
                locale, sep, path = item.partition("=")
                if not sep or not locale or not path or locale in manifests:
                    raise ValueError(f"--manifest expects LOCALE=PATH once per language: {item}")
                manifests[locale] = path
            result = calibrate(args.source, args.output, source_sha256=args.source_sha256, manifests=manifests,
                               bundle=args.bundle, hours=args.hours, epochs=args.epochs, seed=args.seed,
                               device=args.device)
        elif args.operation == "normalize-manifest":
            from .text import normalize_manifest
            result = normalize_manifest(args.input, args.output, lang_field=args.lang_field)
        else:
            from .text import join_clips
            result = join_clips(args.manifest, args.output_dir, pairs=args.pairs, seed=args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"untok: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
