# untok

A SentencePiece Unigram tokenizer for speech-to-text (ASR) and TTS models, based
on NVIDIA Nemotron's native vocabulary and extended with Indic language support.
All bundles preserve every original Nemotron text-token ID, piece, score,
type and normalizer setting. New pieces are appended after the original bank.

## Install

On Linux or macOS, clone the repository:

```sh
git clone https://github.com/plivo-labs/untok.git
cd untok
```

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/).
It manages the Python environment for you:

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

All three bundles are included. Choose one when loading the tokenizer.

| Bundle | Text IDs | Includes |
| --- | ---: | --- |
| `latin` | 13,573 | Original Nemotron bank plus Latin/shared additions |
| `latin-indic` | 20,784 | Original Nemotron bank plus Latin and Indic additions |
| `full` | 20,784 | Original Nemotron bank plus all admitted additions |

**All 13,087 original text IDs (`0..13086`) are unchanged in every bundle.**
Profiles restrict added pieces only. The original multilingual bank is never
filtered, so the current `full` and `latin-indic` model bytes are identical.
The RNNT acoustic blank moves to the new final output row during checkpoint
migration; it is separate from SentencePiece text IDs. Retained weights are
copied, including the learned blank row. See [ID preservation](docs/native-preservation.md).
All three v3 checkpoints passed real NeMo save/reload with exact preservation of
every original learned value; [migration results](configs/native-checkpoint-v3-results.json)
record the hashes and scope. Speech accuracy is a separate evaluation.

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

These lists describe the minimum intended evaluation scope. All bundles also
retain the complete original multilingual vocabulary. Regional variants count
once. The [language coverage report](docs/language-coverage.md) records corpus
checks, remaining unknown characters and limits; Norwegian has a Bokmål text
sample, with Nynorsk untested.

**`latin`: 24 languages.** Croatian, Czech, Danish, Dutch, English, Estonian,
Finnish, French, German, Hungarian, Italian, Latvian, Lithuanian, Maltese,
Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish,
Turkish and Vietnamese.

**`latin-indic`: 47 languages.** All 24 Latin languages above, Arabic, and 22 Indic languages below:
Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri (Arabic script),
Konkani, Maithili, Malayalam, Manipuri (Meetei Mayek), Marathi, Nepali, Odia,
Punjabi (Gurmukhi), Sanskrit, Santali (Ol Chiki), Sindhi (Devanagari),
Tamil, Telugu and Urdu.
Arabic is retained because its script is shared with Kashmiri and Urdu.
Devanagari is shared across its languages; Bengali and Assamese share a vocabulary bank.

**`full`: 56 languages.** All 47 languages above, plus Bulgarian, Greek, Hebrew,
Japanese, Korean, Mandarin Chinese, Russian, Thai and Ukrainian.

## Limitations

- Text coverage does not mean a model can recognize or generate speech in those
  languages.
- Existing checkpoints need vocabulary expansion and blank-row migration. Original
  text labels retain their meanings; regenerate labels to use new segmentation.
  New pieces need speech training. V3 has no completed
  acoustic validation; see [checkpoint usage](docs/native-checkpoint.md).
- The original normalizer is unchanged: ZWNJ becomes a space and legacy Malayalam
  chillu spellings remain distinct. Decoding returns native-normalized text.
- The vocabulary is finite. Uncovered characters or emoji can produce `<unk>`;
  alternate scripts and every dialect are not validated.
- Selecting a bundle selects a vocabulary, not an inference language lock.

See the [native tokenizer guide](docs/native-unigram.md),
[reproduction workflow](docs/native-reproduction.md), and
[source provenance](THIRD_PARTY.md) for implementation and source details.
