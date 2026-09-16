# Native Unigram tokenizer v5

The package includes `original`, `latin`, `latin-indic` and `full`. The profiles
share the original Nemotron normalizer but have different vocabulary and ID
contracts. See [profile and ID details](native-preservation.md).

- `original` is the exact 13,087-piece Nemotron SentencePiece model, byte for byte.
- `latin` keeps Latin pieces, shared punctuation and relevant existing locale tags.
- `latin-indic` keeps Latin and the 22 target Indic script profiles, shared
  punctuation and relevant existing locale tags.
- `full` keeps the entire original Nemotron bank at its original IDs and appends
  the selected Indic vocabulary and required shared characters.

Active counts are 13,087 (`original`), 2,653 (`latin`), 10,372 (`latin-indic`)
and 20,360 (`full`). Physical text slots are 13,087 for Original/Latin and 20,360
for Latin-plus-Indic/Full. Included original pieces retain their Nemotron IDs,
scores and types. Excluded subset IDs remain as reserved `UNUSED` placeholders;
they are not compacted. New Indic/shared pieces start at native ID 13087.

Stable IDs preserve physical checkpoint rows, including inactive rows, so a
restricted active vocabulary does not shrink the acoustic model.
The adapter's `active_vocab_size` counts enabled pieces; `vocab_size` counts
physical slots. Its `vocab` mapping contains active pieces with their stable IDs. Use the Untok
NeMo runtime: it must mask inactive logits as well as skip inactive tokenizer
pieces. Every profile preserves the original normalizer.

## Use and rebuild

```python
from untok.bundles import load_tokenizer

tokenizer = load_tokenizer("original")
ids = tokenizer.text_to_ids("Hello नमस्ते")
print(tokenizer.ids_to_text(ids))
```

Choose any of the four names or pass a bundle directory. `text_to_ids()` uses
native text IDs. `text_to_public_ids()` and `public_ids_to_text()` use the
separate public layout; do not pass those IDs directly as acoustic labels.

```sh
untok clean --bundle src/untok/data/source --output artifacts/profiles-v5
untok check --bundle artifacts/profiles-v5/original
untok check --bundle artifacts/profiles-v5/latin-indic
untok package --bundle artifacts/profiles-v5/full --output artifacts/export-v5
```

`clean` reconstructs all four profiles from the pinned historical source and
profile policy. It rejects a different fitted source. `package` exports all
four by default; `--profiles original full` selects a smaller set. `check`
verifies hashes, the deterministic recipe and ID maps without running corpus
or acoustic evaluation. Bundle source models are retained for reproduction.

## Vocabulary, tags and normalization

Included original strings, scores and piece types stay at their source IDs; there is
no joint score refit. Added strings must be unique, normalization-stable,
non-dominated and either required coverage with witnesses or used in training.
Inherited aliases and redundant native pieces are reported separately. Disabled
subset slots retain their positions with `UNUSED` placeholders; their original
spellings are absent from the active vocabulary.

Subsets filter locale tags explicitly, rather than treating their Latin spelling
as permission to retain every language tag. They retain existing tags only;
additional Indic output tags are not invented. Acoustic prompt slots are a
separate conditioning input and do not create tokenizer pieces.

The normalizer is unchanged: ZWNJ becomes space and legacy Malayalam chillu
spellings remain distinct. Decoding returns native-normalized text. The Latin
profile contains only retained original Nemotron pieces. No profile imports the
extra rare-Latin or Unicode case/decomposition inventories from the superseded
release. Coverage remains finite; no byte fallback is introduced.

## Text validation

```sh
untok validate \
  --bundle artifacts/profiles-v5/full \
  --policy configs/clean-validation.json \
  --corpora /path/to/frozen-data/manifest.json \
  --phase dev \
  --output reports/profiles-v5-dev.json
```

The policy must match the selected artifact's hashes and intended alphabets.
The full-model policy checks original-prefix identity, appended-piece quality,
required coverage and witnesses, fresh training use, normalized round trips,
unknowns and configured token-efficiency thresholds. Raw corpora are not packaged.
Missing or empty profile data means incomplete validation.

Reserve evaluation requires `--phase reserve --selection-receipt receipt.json`.
The receipt binds tokenizer, bundle, source selection, policy and corpus hashes.
A receipt for another version cannot validate the current artifact. These reserve
records have already been examined; repeated checks are regression evidence,
not a new unseen holdout. [Language coverage](language-coverage.md) reports
per-bundle results and remaining gaps.

## Bundle files

| File | Contents |
| --- | --- |
| `tokenizer.model` | Selected runtime tokenizer |
| `base-tokenizer.model` | Exact original NVIDIA tokenizer |
| `full-tokenizer.model` | Historical full v1 source tokenizer |
| `cleanup.json` | Profile selection and preservation evidence |
| `native-row-map.json` | Original/source-to-target rows, including retained inactive rows and acoustic blank |
| `nemo-id-map.json` | Public/native ID mapping |
| `vocabulary.json` | Active pieces, reserved inactive slots, scores, types and ID layouts |
| `manifest.json` | Artifact hashes, recipe identity and scope |

The internal `data/source` bundle preserves the historical fitted selection.
It is a build input, distinct from the public `original` profile.

## Historical selection and training

The original tokenizer comes from NVIDIA's `.nemo` archive at
[revision 1c8deae](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/1c8deaecc64b91f034d73e08dd8b64625eb3395d),
with SHA256 `ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291`.
It is distinct from the separately published BPE `tokenizer.json`.

The v1 study fitted additions against fixed native scores, comparing fresh and
donor-seeded string pools. Donors supplied candidate strings, not BPE rank scores
or normalizers. Shared strings received one physical ID; independently fitted
banks used the maximum provider score when merged. This is a constrained
extension, not a globally trained Unigram distribution.

[Historical results](native-unigram-results.md),
[corpus preparation](native-unigram-data.md) and
[reproduction](native-reproduction.md) describe that source study. A new fitted
selection needs a separately versioned profile recipe and evaluation.
`untok build` packages the original append-only format from a base and selection;
`build-unigram`, `check-unigram` and `validate-unigram` remain aliases.

The [v5 save/reload results](../configs/native-checkpoint-v5-results.json) verify
all four real CPU migrations, exact preservation of every original tensor value,
and inactive-output masks before save and after reload. The
[v4 results](../configs/native-checkpoint-v4-results.json) remain historical
measurements of compact subsets.

Text pieces do not teach acoustic meanings. [Checkpoint migration](native-checkpoint.md),
speech fine-tuning and held-out WER/CER evaluation are separate steps.
[Bounded native Runpod checks](https://github.com/plivo-labs/indic-asr/blob/main/docs/verification.md)
cover integration, not broad speech accuracy. Historical
v1/v3/v4 checkpoint or audio results apply only to their recorded artifact hashes.
