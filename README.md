# untok

A SentencePiece Unigram tokenizer for ASR and TTS, extending NVIDIA Nemotron's
native vocabulary with Indic language support.

## Install

On Linux or macOS, clone the repository:

```sh
git clone https://github.com/plivo-labs/untok.git
cd untok
```

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
uv sync --locked
```

Or use pip with Python 3.11 or later:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Choose a bundle

All three bundles are included.

| Bundle | Text IDs | Includes |
| --- | ---: | --- |
| `latin` | 13,573 | Original Nemotron bank plus Latin/shared additions |
| `latin-indic` | 20,784 | Original Nemotron bank plus Latin and Indic additions |
| `full` | 20,784 | Original Nemotron bank plus all admitted additions |

**All 13,087 original Nemotron text IDs (`0..13086`) are unchanged**, along with
their pieces, scores, types and normalizer. Profiles filter additions only;
`full` and `latin-indic` currently have identical vocabularies.

## Use the tokenizer

Save this as `example.py`. Run `uv run example.py`, or `python example.py` if
you installed with pip:

```python
from untok.bundles import load_tokenizer

tokenizer = load_tokenizer("latin-indic")
ids = tokenizer.text_to_ids("நான் office போகிறேன்")
print(ids)
print(tokenizer.ids_to_text(ids))
```

Use `"latin"` or `"full"` to select another bundle. You can also pass the path to
a custom bundle. The example uses native text IDs.

## Languages

Text coverage is evaluated across these 56 language profiles. All bundles retain
the original multilingual vocabulary; see the [coverage report](docs/language-coverage.md)
for measured gaps.

**Latin (24):** Croatian, Czech, Danish, Dutch, English, Estonian,
Finnish, French, German, Hungarian, Italian, Latvian, Lithuanian, Maltese,
Norwegian (Bokmål), Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish,
Turkish and Vietnamese.

**Indic (22):**
Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri (Arabic script),
Konkani, Maithili, Malayalam, Manipuri (Meetei Mayek), Marathi, Nepali, Odia,
Punjabi (Gurmukhi), Sanskrit, Santali (Ol Chiki), Sindhi (Devanagari),
Tamil, Telugu and Urdu.

**Other (10):** Arabic, Bulgarian, Greek, Hebrew,
Japanese, Korean, Mandarin Chinese, Russian, Thai and Ukrainian.

## Limitations

- Existing Nemotron checkpoints need [migration](docs/native-checkpoint.md),
  which preserves original weights and moves the acoustic blank to the final row.
  All three migrations passed save/reload verification; speech accuracy remains untested.
- New pieces can change segmentation and require speech training.
- The original normalizer is unchanged: ZWNJ becomes a space and legacy Malayalam
  chillu spellings remain distinct. Decoding returns native-normalized text.
- The vocabulary is finite. Uncovered characters or emoji can produce `<unk>`;
  alternate scripts and dialects are not fully validated.
- Selecting a bundle selects a vocabulary, not an inference language lock.

See the [native tokenizer guide](docs/native-unigram.md),
[reproduction workflow](docs/native-reproduction.md), and
[source provenance](THIRD_PARTY.md) for implementation and source details.

Project code: [Apache-2.0](LICENSE). Tokenizer assets retain their [upstream terms](THIRD_PARTY.md).
