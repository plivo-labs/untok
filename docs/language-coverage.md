# Language coverage evidence

The 24/47/56 language lists describe intended vocabulary scope. They are not a
certificate of complete orthography, dialect coverage, or speech accuracy. The
new language-level validation measures all three shipped bundles against each
supplied language and retains unknowns as failures of coverage on those texts.
Every bundle now retains the original multilingual bank. The policy's minimum
evaluation scopes still select required bundle/language combinations; additional
combinations are measured without claiming complete coverage.

## What was evaluated

The original-bundle baseline was measured on 16 September 2026, before the
subsequent tokenizer rebuild. Its model hashes are part of the report. It covers:

- **34 inherited languages:** 100 distinct FLEURS development references per
  language, selected before tokenizer evaluation. These cover all 24 listed
  Latin languages, Arabic, and the other nine languages in Full. Both original
  and publisher-normalized transcripts are evaluated as two forms of the same
  3,400 records.
- **22 Indic profiles:** all 33,027 development and 20,055 reserve records from
  the original frozen corpus. These are the previously evaluated records,
  with their existing source annotation cleanup. They are not new holdouts.
- **55 synthetic language/English combinations:** ten concatenations per
  non-English profile. Each joins an unaltered source reference and an English
  reference with one space. This exercises mixed-script handling; it does not
  establish performance on natural conversational code switching.

[FLEURS](https://huggingface.co/datasets/google/fleurs/blob/70bb2e84b976b7e960aa89f1c648e09c59f894dd/README.md)
is public read-speech reference text derived from parallel FLORES sentences,
licensed CC BY 4.0. Only 7,588,052 bytes of TSV metadata were downloaded; no audio
is needed. Source files are pinned to commit
`70bb2e84b976b7e960aa89f1c648e09c59f894dd` and individual SHA256 digests in
[the policy](../configs/language-coverage.json).

FLEURS provides Bokmål for the Norwegian profile. **Nynorsk remains unvalidated.**
Regional English, Spanish, Portuguese, Arabic, and other variants are not each
represented by independent corpora. Mandarin uses the `cmn_hans_cn` configuration.
Language-to-source details and these limits are recorded in the policy.

## Original v1 baseline

In the v1 baseline, all 56 profiles have evidence, and all representable normalized round trips and
normalizer comparisons pass. **Coverage is not complete:** Full has unknowns in
100/3,400 raw FLEURS references and 14/3,400 publisher-normalized references.
The two totals must not be added as independent samples. The original Indic
development corpus still has 46 unknown-bearing records; reserve has none.

Raw text reveals useful gaps that publisher normalization sometimes removes:
uppercase Maltese `Ċ Ġ Ħ`, uppercase Latvian `Ā Ī Ū`, low double quotation mark
`„`, Japanese quotation marks/middle dot, Thai `ฯ`, Chinese book-title brackets,
and two Chinese characters. Publisher-normalized Turkish introduces combining
dot above (`U+0307`), producing unknowns in nine records even though all 100 raw
Turkish records are representable. Text preparation is therefore part of the
coverage contract; silently switching transcript fields would hide failures.

The machine-readable [baseline result](../configs/language-coverage-results.json) contains
per-language, per-view metrics for **every bundle**, including unknown record and
token rates, unknown codepoints, normalized round trips, mean token lengths, and
p95/p99 tokens per normalized character. The denominator includes SentencePiece
metaspace boundaries and its dummy prefix. Unknown-free rows are the only rows
eligible for round-trip checks, and the eligible count is explicit.

The validation command never fits or changes a tokenizer. A replacement model
must be evaluated again, with its hashes and results kept separate from this
baseline. Repertoire changes should follow independently chosen linguistic or
Unicode inventories; selecting additions from these examples would make them
selection data. Existing checkpoints need matching migration and training work.

## Preserved v3 results

The same frozen text was evaluated again after replacing the bundles. The
[version 3 report](../configs/language-coverage-v3-results.json) binds the new
model hashes and keeps every language, bundle and transcript view explicit.
All 56 profiles have evidence; all representable normalized round trips and
normalizer comparisons pass.

| Full bundle evidence | v1 unknown-bearing records | v3 unknown-bearing records | Records |
| --- | ---: | ---: | ---: |
| FLEURS raw transcripts | 100 | 87 | 3,400 |
| FLEURS publisher-normalized transcripts | 14 | 4 | 3,400 |
| Original Indic development | 46 | 45 | 33,027 |
| Original Indic reserve | 0 | 0 | 20,055 |

The raw and publisher-normalized rows are two views of the same references.
The v3 repertoire uses Unicode 17 case and canonical decomposition closure of
existing single Latin characters; the builder does not select additions from
these evaluation sentences. This closes the measured Maltese, Latvian uppercase,
and Turkish combining-dot gaps. All 13,087 original Nemotron text IDs and the original normalizer are
preserved. ZWNJ still becomes space and Malayalam spelling variants remain
distinct. The earlier compact v2 candidate is withdrawn.

Coverage is still finite. Raw FLEURS gaps remain for low quotation marks in
several European languages, Japanese punctuation, Thai `ฯ`, Chinese book-title
brackets and two Chinese characters. The four publisher-normalized failures are
one Japanese, one Thai and two Chinese records. The machine report lists exact
codepoints and counts. Indic development also retains 45 unknown-bearing rows.
The all-language result is therefore **complete_with_coverage_gaps**, not a
zero-unknown pass. Adding a larger independent alphabet/punctuation repertoire
or byte fallback would require another explicit vocabulary and training change.

## Reproduce

From the repository root, fetch only the pinned transcript metadata:

```sh
uv run python scripts/validate_language_coverage.py fetch
```

Selection is deterministic and independent of tokenizer output. It deduplicates
using an NFC/whitespace key on the publisher-normalized transcript, chooses one
source-ID representative, sorts distinct keys by SHA256, and takes the first 100.
The raw and normalized source fields are preserved. Both source TSVs and selected
JSONL files have pinned hashes. Existing conflicting cache/output bytes are
rejected. No raw corpus is committed to Git.

Evaluate the 34 new language corpora and, when available, the original Indic
freeze:

```sh
uv run python scripts/validate_language_coverage.py validate \
  --frozen-indic-manifest /path/to/original/frozen-data/manifest.json \
  --output reports/language-coverage.json
```

Without `--frozen-indic-manifest`, the report still evaluates all 34 FLEURS
languages and explicitly marks the 22 Indic profiles missing. It cannot claim
all-language completeness. The Indic manifest must have the exact original
freeze hash; missing data is not reconstructed from another script or source.
Use `fetch --offline` to verify and prepare already downloaded TSVs without
network access.

The script exits 0 only for `passed_on_supplied_text`. Unknowns in an in-scope
profile produce `complete_with_coverage_gaps` and exit 2; missing profiles produce
`incomplete`; normalizer or round-trip failures produce `failed`. A completed
measurement with coverage gaps remains valid evidence and must not be presented
as a zero-unknown pass.

## Limits

The sample is balanced by language and distinct transcript count, not dialect,
domain, or frequency of real-world usage. Parallel translations are correlated
across languages. FLEURS was not used to refit this tokenizer during this check,
but absence from Nemotron or donor pretraining is not established. Earlier
English FLEURS controls also mean this is not a wholly untouched evaluation
source. Original Indic development data was used during selection, and its
reserved records all come from IndicVoices.

The tests cover source/selection integrity, literal TSV parsing, distinct-text
sampling, immutable preparation, missing evidence, unknowns, and all-bundle
reporting. They do not replace the full corpus run. Natural code-switched
conversations, alternate scripts, Nynorsk, and broader dialect/domain sampling
remain separate validation work.
