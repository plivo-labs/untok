# Original Nemotron ID preservation

The current v3 bundles preserve all **13,087 original SentencePiece entries at
IDs 0 through 13086**. This is an exhaustive byte comparison of each piece
message, including spelling, score and type. The original normalizer and every
other model metadata field are unchanged except the declared vocabulary size.
The original model is pinned to SHA256
`ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291`.

The earlier v2 compaction candidate violated this contract and is withdrawn.
Preservation of a source copy or a migration map never substitutes for keeping
the actual tokenizer entries at their original IDs.

## Profiles and additions

| Profile | Original entries unchanged | Text entries | Native RNNT blank |
| --- | ---: | ---: | ---: |
| `latin` | 13,087 | 13,573 | 13,573 |
| `latin-indic` | 13,087 | 20,784 | 20,784 |
| `full` | 13,087 | 20,784 | 20,784 |

Profiles filter **additions only**. All retain the complete original multilingual
bank, even its non-Latin pieces. The current full and Latin-plus-Indic vocabularies
are identical because every selected addition is admitted by Latin-plus-Indic.
The Latin extension admits 252 original-study additions; the other two admit
all 7,463. All three append 234 Unicode-derived Latin case/decomposition coverage
characters at score -32. This is not byte fallback or a joint score refit.

Cleanup may reject an added duplicate, unstable spelling or dominated piece.
It may never remove, renumber or edit an original piece. Consequently inherited
normalization aliases and dominated pieces remain and are reported separately.
The original normalizer still changes ZWNJ to space and leaves legacy Malayalam
chillu spellings distinct. No external spelling cleanup is applied implicitly.

The builder and loader call `validate_native_prefix`; tests independently compare
every original piece and all metadata. Tampering with original IDs, scores,
types, normalization or metadata must fail even after artifact hashes are updated.
Appending pieces can change segmentation despite unchanged original IDs and scores.

## Blank and public IDs

SentencePiece has original text IDs `0..13086`; its vocabulary contains no RNNT
blank. The original acoustic model used blank output **13087**. NeMo expects
blank after the text vocabulary, so checkpoint migration moves that learned row
to the final output index shown above. Every original text row stays at its
original index. Migration copies the learned blank predictor/output rows exactly
and verifies them before save and after reload; it does not reset them.

Keeping acoustic blank numerically fixed would require a separate acoustic ID
mapping and coordinated changes to NeMo's predictor, loss and decoding paths.
The standard blank-last layout avoids that runtime divergence. Consumers storing
acoustic blank IDs must apply the migration map.

Untok's separate **public** padding and blank IDs remain **13087 and 13088**.
New public text IDs begin at 13089. Public IDs must be converted with the supplied
mapping before use as acoustic labels; they are not the native SentencePiece IDs.
The existing v1 checkpoint receipts already recorded acoustic blank relocation,
including original 13087 to full-v1 20550.

## Rebuild and validate

```sh
untok clean --bundle src/untok/data/source --output artifacts/preserved-v3
untok check --bundle artifacts/preserved-v3/full
untok validate --bundle artifacts/preserved-v3/full \
  --policy configs/clean-validation.json \
  --corpora /path/to/original/frozen-data/manifest.json \
  --output reports/preserved-v3-dev.json
```

The extension policy is independently reproducible from the pinned original
models and Unicode 17 data, with the project's SentencePiece 0.2.1 runtime:

```sh
python scripts/build_extension_policy.py \
  --base src/untok/data/source/base-tokenizer.model \
  --full-model src/untok/data/source/tokenizer.model \
  --unicode-data /path/to/UnicodeData.txt \
  --output /path/to/new-policy.json
```

This builds a finite character inventory and never compiles a replacement
normalizer. Its output must equal `src/untok/data/extension-v3/policy.json`.
Source acquisition and pinned input locations are in [reproduction](native-reproduction.md).

Current [validation results](../configs/clean-validation-results.json),
[all-profile training usage](../configs/clean-profile-usage.json) and
[selection receipt](../configs/clean-selection-receipt.json) bind the v3 hashes.
Reserve is previously evaluated data and provides a regression check, not a new
unseen holdout. [Language coverage](language-coverage.md) reports remaining gaps.
All three real v3 checkpoints passed NeMo save/reload on 16 September 2026.
Every original text row and all 638,030,384 original learned values are exactly
preserved; only the acoustic blank row moves. See the
[migration verification receipt](../configs/native-checkpoint-v3-results.json)
and [checkpoint guide](native-checkpoint.md). Acoustic accuracy and training
have not been evaluated for v3.
