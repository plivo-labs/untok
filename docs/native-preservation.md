# Stable Nemotron IDs and tokenizer profiles

Version 5 keeps every included original Nemotron piece at its original ID,
with its original score and type. The subsets keep excluded IDs as inactive
physical slots instead of renumbering the remaining pieces.

| Profile | Active original pieces | Added pieces | Active total | Inactive slots | Physical text slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| `original` | 13,087 | 0 | 13,087 | 0 | 13,087 |
| `latin` | 2,653 | 0 | 2,653 | 10,434 | 13,087 |
| `latin-indic` | 3,099 | 7,273 | 10,372 | 9,988 | 20,360 |
| `full` | 13,087 | 7,273 | 20,360 | 0 | 20,360 |

`original/tokenizer.model` is byte-identical to NVIDIA's embedded model, pinned
to SHA256 `ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291`.
`full` retains its entire prefix. Inactive subset slots use reserved `UNUSED`
placeholders rather than the excluded language spellings. They are not active
vocabulary, and the tokenizer does not emit them.

Every included original ID remains in `0..13086`. The 7,273 Indic/shared pieces
start at native ID 13087 and use the same IDs in `latin-indic` and `full`.
No profile adds the rare-Latin or Unicode case/decomposition inventories from
superseded releases. Latin is drawn entirely from original Nemotron pieces.

## Acoustic rows and blank

Stable IDs retain the original physical row positions. They do **not** produce
a smaller acoustic model. Migration copies every original checkpoint row,
including rows made inactive in a subset. The Untok NeMo runtime must mask
inactive output logits; retaining a row's weights does not make it an allowed
prediction. SentencePiece's `UNUSED` status alone cannot enforce this acoustic
restriction.

| Profile | Physical text slots | Native acoustic blank | Change from Nemotron |
| --- | --- | ---: | --- |
| `original` | `0..13086` | 13087 | Unchanged |
| `latin` | `0..13086` | 13087 | Unchanged; inactive outputs masked |
| `latin-indic` | `0..20359` | 20360 | Blank weights move to final row |
| `full` | `0..20359` | 20360 | Blank weights move to final row |

The blank is always the final acoustic output row, following **physical slots**,
not the number of active pieces. The 2,653 active Latin pieces keep sparse
original IDs; they are not renumbered to `0..2652`.

All four profiles retain public padding 13087 and public blank 13088.
New public text IDs start at 13089. Use `nemo-id-map.json` to convert public IDs
to native acoustic labels. The native and public blank IDs are different spaces.
Original labels referring to included pieces retain their meaning; labels for
inactive pieces are invalid for that subset and must be regenerated from text.

## Script, tags and normalization

The active Latin subset contains Latin/shared pieces and 28 existing Latin
locale tags. Latin-plus-Indic contains those tags plus the existing Hindi tag.
Original and Full retain all 39 original tags. Inactive tag spellings are removed
from the subsets, rather than retained because their text uses Latin letters.
No new output tags are invented. Acoustic prompt slots are separate model inputs.

Script membership follows the pinned Unicode table, including shared punctuation
and Script_Extensions. It restricts available pieces, not the language of every
sentence. Every profile keeps original SentencePiece normalization: ZWNJ becomes
space and legacy Malayalam chillu spellings remain distinct. Retained scores are
unchanged; inherited aliases and redundant native pieces are reported separately.
There is no joint score refit.

## Rebuild and evidence

```sh
untok clean --bundle src/untok/data/source --output artifacts/profiles-v5
untok check --bundle artifacts/profiles-v5/original
untok check --bundle artifacts/profiles-v5/latin-indic
```

Each bundle retains the original base and historical full v1 source as rebuild
inputs. Its manifest must bind the selected model, inactive-slot policy and ID
maps. See [reproduction](native-reproduction.md) for source acquisition and fitting.

The [v5 ID audit](../configs/native-id-v5-results.json) compares every original
piece with the current models and confirms zero moved retained IDs or altered
retained piece records. Fresh [v5 text coverage](../configs/language-coverage-v5-results.json)
binds those model hashes and retains measured unknowns.

The [v5 checkpoint report](../configs/native-checkpoint-v5-results.json) verifies
all four real migrations on CPU: all 638,030,384 original tensor values remain
exact before save and after reload, including inactive rows. Inactive-output
mask checks pass at both stages. Speech accuracy, full acoustic training and
GPU execution remain outside that evidence.

The [v4 migration report](../configs/native-checkpoint-v4-results.json) and
[v4 language report](../configs/language-coverage-v4-results.json) are historical:
v4 compacted subset IDs and removed acoustic rows. V3 retained the whole
multilingual vocabulary, and the earlier v2 candidate was withdrawn.
