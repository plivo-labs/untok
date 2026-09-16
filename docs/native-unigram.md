# Native Unigram tokenizer v3

All three profiles preserve original Nemotron text IDs `0..13086`, including
complete piece messages, scores, types and normalization metadata. Profile
filtering and cleanup apply only to appended pieces. The complete original
multilingual bank stays in every bundle. See [the preservation contract](native-preservation.md).

| Profile | Native text entries | Native RNNT blank | Public padding | Public blank |
| --- | ---: | ---: | ---: | ---: |
| `latin` | 13,573 | 13,573 | 13,087 | 13,088 |
| `latin-indic` | 20,784 | 20,784 | 13,087 | 13,088 |
| `full` | 20,784 | 20,784 | 13,087 | 13,088 |

Native text IDs are dense `0..N-1`. The acoustic blank is `N`, so it moves during
checkpoint migration while every original text ID stays fixed. Public text IDs
reserve 13087 and 13088 for padding and blank; use the supplied conversion map.
The current full and Latin-plus-Indic models are identical because all selected
additions fall within the Latin-plus-Indic extension scope.

## Vocabulary and scores

V3 starts from the original 13,087-piece native bank and the pinned full v1
selection. Original entries are immutable, including inherited normalization
aliases and dominated pieces. Added strings must be unique, normalization-stable,
non-dominated and either required coverage with witnesses or used in training.
The validation report distinguishes inherited limitations from added-piece gates.

Retained scores are unchanged. The native bank and independently fitted additions
are not globally refitted. New Unicode case/decomposition characters use score
-32. Appending pieces can change segmentation without changing original IDs.
The native normalizer is unchanged: ZWNJ becomes space and Malayalam legacy
chillu spellings remain distinct. Decoding returns native-normalized text.

## Use and rebuild

```python
from untok.bundles import load_tokenizer

tokenizer = load_tokenizer("latin-indic")
ids = tokenizer.text_to_ids("நான் office போகிறேன்")
print(ids)
print(tokenizer.ids_to_text(ids))
```

Use `text_to_public_ids()` and `public_ids_to_text()` only when the public ID
layout is wanted. `load_tokenizer()` also accepts a custom bundle directory.

From a repository checkout, reproduce the current three profiles from the
preserved source bundle into a new directory:

```sh
untok clean --bundle src/untok/data/source --output artifacts/preserved-v3
untok check --bundle artifacts/preserved-v3/full
```

The clean recipe verifies both source tokenizer hashes and the packaged
extension policy. It rejects a different fitted source model. `check` verifies
hashes, the exact cleanup recipe and ID maps; it does not run corpus or acoustic
evaluation. An installed v3 bundle also carries the original full model and
can be supplied to `clean` as the source.

To run CPU validation against the frozen Indic corpus:

```sh
untok validate \
  --bundle artifacts/preserved-v3/full \
  --policy configs/clean-validation.json \
  --corpora /path/to/frozen-data/manifest.json \
  --phase dev \
  --output reports/clean-v3-dev.json
```

This policy pins the current full model, source identities, finite alphabets
and per-language thresholds. It checks the complete original prefix and appended NORMAL inventory,
required canonical coverage and witnesses, actual training use of surviving
optional fitted additions, and development unknowns, normalized round trips
and token efficiency. The frozen source corpus's normalizer hash is provenance;
v3 decoding uses that same original normalizer. Raw corpora are not packaged.
Missing or empty profile data is incomplete validation, not a pass.

Reserve evaluation requires `--phase reserve --selection-receipt receipt.json`.
Freeze the v3 `tokenizer_sha256`, `data_manifest_sha256`,
`bundle_manifest_sha256`, original-source `selection_sha256` and
`policy_sha256` before evaluating it. V1 receipts cannot validate v3. The old
reserve is already examined data, so rerunning it is a regression test rather
than a new unseen holdout. See [language coverage](language-coverage.md) for
current corpus results and remaining gaps.

The checked-in [full-profile dev/reserve results](../configs/clean-validation-results.json),
[selection receipt](../configs/clean-selection-receipt.json), and
[training usage for all three profiles](../configs/clean-profile-usage.json)
record the current artifact hashes. These are tokenizer checks, not speech
recognition results.

## Bundle files

| File | Contents |
| --- | --- |
| `tokenizer.model` | V3 Unigram vocabulary and the original normalizer |
| `base-tokenizer.model` | Exact original NVIDIA native tokenizer |
| `full-tokenizer.model` | Exact full v1 source tokenizer |
| `cleanup.json` | Added-piece filtering, Unicode coverage, and inherited limitations |
| `native-row-map.json` | Identity map for original text IDs; full-v1 added rows and acoustic blank map explicitly |
| `nemo-id-map.json` | Public/native ID mapping and blank placement |
| `vocabulary.json` | Pieces, scores, types and ID layouts |
| `manifest.json` | Hashes, recipe identity and validation scope |

The internal `data/source` bundle retains the historical selection and its
inputs for reproducibility. It is not a fourth named public profile.

## Historical selection and further training

The original native tokenizer comes from the `.nemo` archive at NVIDIA
[revision 1c8deae](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/1c8deaecc64b91f034d73e08dd8b64625eb3395d),
with SHA-256 `ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291`.
It is distinct from NVIDIA's published BPE `tokenizer.json`.

The v1 study fitted new scores against fixed native scores, comparing fresh
and donor-seeded string pools. BPE donor strings did not carry their rank scores
or normalizers into the native Unigram model. Required characters and approved
Hindi pieces counted inside group budgets; duplicate strings received one
physical ID. This produced a constrained extension, with independently fitted
banks and maximum provider score used for shared cross-bank strings.

[Historical results](native-unigram-results.md) document that study, including
its old normalizer and old IDs. [Corpus preparation](native-unigram-data.md)
describes its frozen data. [Reproduction](native-reproduction.md) supplies
portable acquisition, preparation, fitting and merge scripts for a new study;
a newly fitted model does not automatically satisfy the pinned v3 cleanup recipe.
`untok build` reconstructs the original append-only format from a base and
selection. The older `build-unigram`, `check-unigram` and `validate-unigram`
names remain aliases. Historical BPE commands use the `legacy-bpe` prefix.

The tokenizer does not teach a model acoustic meanings for new pieces.
[Checkpoint migration](native-checkpoint.md), speech fine-tuning and held-out
WER/CER evaluation are separate requirements. Historical v1 migration and audio
checks do not establish v3 checkpoint compatibility or speech accuracy.
