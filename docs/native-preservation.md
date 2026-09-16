# Tokenizer profiles and original Nemotron IDs

Version 4 has four profiles with different preservation contracts.

| Profile | Text entries | Vocabulary | Original native text IDs |
| --- | ---: | --- | --- |
| `original` | 13,087 | Exact original Nemotron SentencePiece model | All 13,087 unchanged; model bytes identical |
| `latin` | 2,653 | Latin, shared punctuation and relevant existing tags | Compact subset IDs; use the row map |
| `latin-indic` | 10,372 | Latin and the 22 target Indic profiles, shared punctuation and relevant existing tags | Compact subset IDs; use the row map |
| `full` | 20,360 | Complete original Nemotron bank plus selected Indic additions | Original IDs `0..13086`, scores and types unchanged |

No profile imports the extra rare-Latin or Unicode case/decomposition inventories
from the superseded release. `latin` is derived only from the original Nemotron
vocabulary; Indic expansion is kept in `latin-indic` and `full`.

The original model is pinned to SHA256
`ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291`.
`original/tokenizer.model` must match those exact bytes; the Untok bundle adds
wrapper metadata and ID maps without modifying that model.

`full` retains every original piece message at its original ID. Its model
metadata is unchanged apart from vocabulary length. Subsets preserve retained
piece messages and the normalizer, while remapping text and special IDs to the
smaller inventory. Removed original pieces have a `null` source-to-target map.
Their labels require retokenization from text, not a numeric substitution.

## Script and tag scope

The `latin` and `latin-indic` profiles filter the whole vocabulary, including
original pieces. Latin-plus-Indic includes the selected Indic/shared additions
needed by the 22 target profiles. They do not retain the complete multilingual bank. Script
membership follows the pinned Unicode table, including shared punctuation and
Script_Extensions. This is a script restriction, not a language detector.

Language-tag pieces need a separate whitelist because their spellings use Latin
letters. `latin` retains 28 existing Latin-language locale tags; `latin-indic`
retains those tags plus the existing Hindi tag. `original` and `full` retain all
39 original tags. No new output tags are invented for the additional Indic
languages. Acoustic prompt slots are a separate model input.

All profiles retain the original SentencePiece normalization behavior: ZWNJ
becomes a space and legacy Malayalam chillu spellings remain distinct. Native
aliases and redundant original pieces remain where required by the preservation
contract. Added-piece quality checks do not claim that the inherited vocabulary
contains no such limitations. Retained scores are unchanged; no joint score
refit is performed.

## Blank and public IDs

SentencePiece contains text IDs, not an RNNT blank token. NeMo places its
acoustic blank at the text vocabulary size. Thus `original` keeps native blank
13087, while other profiles use their own final output row. Checkpoint migration
copies the learned blank weights to that row; it does not reset them.

| Profile | Native text IDs | Native acoustic blank | Placement |
| --- | --- | ---: | --- |
| `original` | `0..13086` | 13087 | Final row; unchanged from Nemotron |
| `latin` | `0..2652` | 2653 | Final row; remapped from 13087 |
| `latin-indic` | `0..10371` | 10372 | Final row; remapped from 13087 |
| `full` | `0..20359` | 20360 | Final row; remapped from 13087 |

`original` and `full` use public padding 13087 and public blank 13088, with new
public text IDs beginning at 13089. For the compact `latin` and `latin-indic`
profiles, public text IDs are `0..N-1`, public padding is `N`, public blank is
`N+1`, and native acoustic blank is `N`. Use each bundle's `nemo-id-map.json`
when converting public IDs to acoustic labels.

## Rebuild

```sh
untok clean --bundle src/untok/data/source --output artifacts/profiles-v4
untok check --bundle artifacts/profiles-v4/original
untok check --bundle artifacts/profiles-v4/full
```

The recipe pins the original base, historical full source and its vocabulary
policy. Each bundle contains `base-tokenizer.model` and `full-tokenizer.model`
as reproducibility inputs; `tokenizer.model` is the selected runtime profile.
See [reproduction](native-reproduction.md) for source acquisition and fitting.

## Evidence by version

All four v4 checkpoints passed real NeMo construction, exact retained-tensor
checks, save and reload on 16 September 2026. `original` keeps the tokenizer
byte-identical and retains every original learned value. The two subsets omit
excluded rows and preserve every retained value exactly; `full` retains every
original value and initializes the added rows. Blank weights are verified at
the final indices above. The [v4 migration receipt](../configs/native-checkpoint-v4-results.json)
binds the current tokenizer, bundle, checkpoint and implementation hashes.
Speech accuracy and training have not been evaluated.

The former v2 compaction candidate was withdrawn. The superseded v3 release kept
the original multilingual bank in every profile. Its
[checkpoint migration report](../configs/native-checkpoint-v3-results.json) and
[language report](../configs/language-coverage-v3-results.json) remain historical
measurements of their recorded hashes. They do not describe the v4 subsets.

Current checks must bind the exact v4 model and bundle hashes. Structural
identity, text coverage, checkpoint migration and speech accuracy are distinct
claims; consult the [tokenizer guide](native-unigram.md),
[checkpoint guide](native-checkpoint.md) and [language evidence](language-coverage.md).
