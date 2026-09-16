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

All four bundles are included.

| Bundle | Text entries | RNNT blank ID | Contents |
| --- | ---: | ---: | --- |
| `original` | 13,087 | 13,087 | Exact, unmodified Nemotron base |
| `latin` | 2,653 | 2,653 | Original Nemotron Latin/shared pieces and relevant existing tags; no additions |
| `latin-indic` | 10,372 | 10,372 | 3,099 original Latin/Indic/shared pieces + 7,273 Indic/shared additions for the 22 target profiles |
| `full` | 20,360 | 20,360 | All 13,087 original pieces + the same 7,273 Indic/shared additions |

`original` is byte-identical to the original 13,087-piece model. `full` preserves
all original text IDs, scores and types. The two script subsets use compact IDs
and explicit checkpoint row maps; their IDs differ from the original model.
All four use the original Nemotron normalizer. The RNNT blank is always the
final acoustic output row, separate from text tokens and public IDs.

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

All four [NeMo migrations](configs/native-checkpoint-v4-results.json) passed exact
retained-weight checks and save/reload; speech accuracy remains unevaluated.

## Scope and limitations

- The Indic profiles are Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada,
  Kashmiri (Arabic), Konkani, Maithili, Malayalam, Manipuri (Meetei Mayek), Marathi,
  Nepali, Odia, Punjabi (Gurmukhi), Sanskrit, Santali (Ol Chiki), Sindhi (Devanagari),
  Tamil, Telugu and Urdu. Shared scripts also permit other languages.
- Script selection limits the vocabulary; it does not guarantee an inference
  language or recognition accuracy. Language prompt slots are separate from
  tokenizer language-tag pieces.
- Use a [matching checkpoint](docs/native-checkpoint.md). Subsets require row
  remapping; migration preserves retained weights, including blank. Only the
  Indic/shared additions introduce new token rows that need speech training.
- The unchanged normalizer converts ZWNJ to a space and leaves legacy Malayalam
  chillu variants distinct. Decoding returns normalized text.
- Coverage is finite; unknown characters can produce `<unk>`. See the
  [language evidence and its limits](docs/language-coverage.md).

[Profile and ID contracts](docs/native-preservation.md) ·
[Tokenizer guide](docs/native-unigram.md) ·
[Reproduction](docs/native-reproduction.md) ·
[Source provenance](THIRD_PARTY.md)

Project code: [Apache-2.0](LICENSE). Tokenizer assets retain their
[upstream terms](THIRD_PARTY.md).
