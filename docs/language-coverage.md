# Text coverage evidence

These measurements describe the unchanged current tokenizer models, not speech
accuracy or complete language support. The four bundles preserve included
Nemotron IDs and the original normalizer. Subsets replace excluded original
slots with inactive placeholders; neither Indic bundle adds a rare Latin bank.

The [recorded current-model report](../configs/language-coverage-v5-results.json)
contains model hashes and all per-language results. Evidence covers 56 text
profiles across all four bundles. Normalizer comparisons and representable
normalized round trips pass, but unknowns remain. Its status is
**complete_with_coverage_gaps**, not a zero-unknown pass.

## Measured scope

The inherited-language check uses 100 distinct FLEURS development references
per language, for 34 languages and 3,400 total records. Raw and publisher-normalized
text are two forms of the same records, not independent samples. The 22 Indic
profiles use the frozen study's 33,027 development and 20,055 reserved records.
These are reused evaluation sets, not fresh holdouts. Synthetic English joins
exercise mixed-script text but do not establish natural code-switching quality.

| In-scope FLEURS text | Records | Raw unknown-bearing records | Publisher-normalized unknown-bearing records |
| --- | ---: | ---: | ---: |
| `original` | 3,400 | 635 | 222 |
| `latin` | 2,400 | 464 | 195 |
| `latin-indic` | 2,500 | 159 | 108 |
| `full` | 3,400 | 191 | 113 |

Latin measures the 24 listed Latin languages. Latin + Indic adds Arabic to that
FLEURS scope; the other Indic languages use their separate frozen corpus.
Original and Full measure all 34 inherited-language FLEURS profiles. Hindi
adds the 35th Original text profile, and the full Indic set brings Full to 56.

| All 22 Indic profiles | Development unknown-bearing records | Reserve unknown-bearing records |
| --- | ---: | ---: |
| `latin-indic` | 61 / 33,027 | 0 / 20,055 |
| `full` | 51 / 33,027 | 0 / 20,055 |

Full has 154 unknown development tokens and Latin + Indic has 181. These counts
exceed the earlier, broader Latin-inventory study's thresholds; zero reserve
unknowns does not erase those gaps. The original Hindi baseline has unknowns in
201/1,753 development records and 23/898 reserved records.

## Limits and reproduction

Nynorsk remains unvalidated; FLEURS supplies Bokmål for Norwegian. Regional
variants, alternate Indic scripts and broader dialect/domain coverage are not
established. The parallel FLEURS sentences are correlated across languages,
and their absence from upstream pretraining is not established. Reserved Indic
records all come from one source.

Source and selection hashes are retained in the
[pinned coverage policy](../configs/language-coverage-v5.json). FLEURS metadata
is pinned to revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, under CC BY 4.0.
The immutable [evaluation instructions](https://github.com/plivo-labs/untok/blob/e2f8acd1bab8edf1dc72267678d4a2ba41949439/docs/language-coverage.md)
retain the historical data-acquisition and measurement commands. Those commands
are retired from this checkout; [current verification](native-reproduction.md)
checks byte identity against the measured tokenizer artifacts.
