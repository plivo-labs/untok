# Native IDs and preservation

Every retained Nemotron piece keeps its original ID, score and type. All models
keep the original SentencePiece normalizer. Subsets reserve excluded IDs as
inactive slots instead of renumbering the remaining pieces.

| Profile | Active original pieces | Additions | Active total | Inactive slots | Native blank |
| --- | ---: | ---: | ---: | ---: | ---: |
| `original` | 13,087 | 0 | 13,087 | 0 | 13,087 |
| `latin` | 2,653 | 0 | 2,653 | 10,434 | 13,087 |
| `latin-indic` | 3,099 | 7,273 | 10,372 | 9,988 | 20,360 |
| `full` | 13,087 | 7,273 | 20,360 | 0 | 20,360 |

The original model is byte-identical to NVIDIA's embedded model, with SHA256
`ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291`.
Original text IDs are `0..13086`; additions occupy `13087..20359` identically
in Latin + Indic and Full. RNNT blank is the next row after physical text slots.

Inactive slots contain UNUSED placeholders. They are not emitted by text
tokenization. Acoustic use also requires Untok's inactive-output mask, because
SentencePiece piece types cannot restrict a model's logits. Migration retains
inactive checkpoint weights as well as active weights, so subsets do not shrink
the acoustic model.

Latin has 28 original language tags; Latin + Indic adds the original Hindi tag.
Original and Full retain all 39 original tags. No new output tags or extra Latin
character bank are added. Acoustic prompt slots are separate model inputs.

Selection follows the frozen approved inventory, not runtime script detection.
Shared scripts do not identify one language exclusively. ZWNJ still normalizes
to a space, and legacy Malayalam chillu spellings remain distinct. Decoding
returns native-normalized text rather than necessarily the input spelling.

New additions can change segmentation even where the original model knew every
character. Re-encode training transcripts after changing bundles. Stable IDs
preserve row meaning, not speech accuracy or an unchanged training objective.

See [assembly and verification](native-reproduction.md),
[coverage evidence](language-coverage.md) and
[checkpoint usage](native-checkpoint.md).
