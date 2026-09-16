# untok

SentencePiece Unigram tokenizers for ASR and TTS, based on NVIDIA Nemotron's
native tokenizer and extended with Indic vocabulary.

## Install

```sh
git clone https://github.com/plivo-labs/untok.git
cd untok
uv sync --locked
```

Or install with Python 3.11 or later: `python -m pip install .`.

## Choose a bundle

| Bundle | Active pieces | Text slots | RNNT blank ID | Contents |
| --- | ---: | ---: | ---: | --- |
| `original` | 13,087 | 13,087 | 13,087 | Exact, unmodified Nemotron base |
| `latin` | 2,653 | 13,087 | 13,087 | Original Latin/shared pieces and relevant tags; no additions |
| `latin-indic` | 10,372 | 20,360 | 20,360 | 3,099 original pieces + 7,273 Indic/shared additions |
| `full` | 20,360 | 20,360 | 20,360 | All 13,087 original pieces + the same Indic/shared additions |

**Included Nemotron pieces keep their original IDs, scores and types.** Subsets
reserve excluded IDs as inactive slots; they do not compact IDs or shrink the
model. Untok's NeMo runtime masks those inactive outputs. The original normalizer
is unchanged, and RNNT blank is always the final acoustic row.

## Use

```python
from untok.bundles import load_tokenizer

tokenizer = load_tokenizer("latin-indic")
ids = tokenizer.text_to_ids("நான் office போகிறேன்")
print(ids)
print(tokenizer.ids_to_text(ids))
```

Select `"original"`, `"latin"` or `"full"` the same way, or pass a bundle directory.
The example returns native text IDs. Public IDs use a separate mapping.

## Scope and limitations

- The 22 Indic profiles are Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi,
  Kannada, Kashmiri (Arabic), Konkani, Maithili, Malayalam, Manipuri (Meetei Mayek),
  Marathi, Nepali, Odia, Punjabi (Gurmukhi), Sanskrit, Santali (Ol Chiki), Sindhi
  (Devanagari), Tamil, Telugu and Urdu. Shared scripts permit other languages.
- Use a [matching checkpoint and Untok runtime](docs/native-checkpoint.md).
  [All four CPU migrations](configs/native-checkpoint-v5-results.json) verified
  exact original weights and inactive masks after reload. Only the Indic profiles
  add rows and move blank; new pieces require speech training.
- Script selection does not guarantee an inference language or recognition
  accuracy. Acoustic prompt slots are separate from tokenizer language tags.
- The original normalizer converts ZWNJ to a space and leaves legacy Malayalam
  chillu variants distinct. Decoding returns normalized text.
- Coverage is finite; unknown characters can produce `<unk>`.
  [Fresh text checks](docs/language-coverage.md) retain the measured coverage gaps.

[Profile and ID contracts](docs/native-preservation.md) ·
[Language evidence](docs/language-coverage.md) ·
[Reproduction](docs/native-reproduction.md) ·
[Source provenance](THIRD_PARTY.md)

Project code: [Apache-2.0](LICENSE). Tokenizer assets retain their
[upstream terms](THIRD_PARTY.md).
