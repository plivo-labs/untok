# untok

SentencePiece Unigram tokenizers for ASR and TTS, based on NVIDIA Nemotron's
native tokenizer and extended with Indic vocabulary.

## Install

On Linux or macOS, use Python 3.11 or later:

```sh
git clone https://github.com/plivo-labs/untok.git
cd untok
uv sync --locked
source .venv/bin/activate
```

Or install with `python -m pip install .`. All four bundles are included.

## Choose a bundle

| Bundle | Active pieces | Text slots | RNNT blank ID | Contents |
| --- | ---: | ---: | ---: | --- |
| `original` | 13,087 | 13,087 | 13,087 | Exact Nemotron base |
| `latin` | 2,653 | 13,087 | 13,087 | Original Latin/shared pieces and relevant tags |
| `latin-indic` | 10,372 | 20,360 | 20,360 | 3,099 original pieces + 7,273 Indic/shared additions |
| `full` | 20,360 | 20,360 | 20,360 | All original pieces + the same additions |

Included Nemotron pieces keep their original IDs, scores and types. Subsets
reserve excluded IDs as inactive slots, so they do not shrink the acoustic
model. Untok's NeMo runtime masks those outputs. The original normalizer stays
unchanged; RNNT blank is the final acoustic row.

## Use

```python
from untok.bundles import load_tokenizer

tokenizer = load_tokenizer("latin-indic")
ids = tokenizer.text_to_ids("நான் office போகிறேன்")
print(tokenizer.ids_to_text(ids))
```

Select another bundle name the same way, or pass a bundle directory. These are
native text IDs. Text tokenization needs neither NeMo nor a GPU.

## Nemotron checkpoint

In a compatible NeMo environment:

```sh
untok migrate \
  --source nemotron-3.5-asr-streaming-0.6b.nemo \
  --source-sha256 210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74 \
  --bundle latin-indic \
  --output nemotron-latin-indic.nemo
```

Migration retains existing weights and initializes new rows from existing text
pieces. New pieces still need speech training. See [checkpoint usage](docs/native-checkpoint.md)
and [indic-asr](https://github.com/plivo-labs/indic-asr) for NVIDIA's training
and evaluation scripts. No trained replacement checkpoint is included.

## Languages and limitations

These counts describe the documented text profiles, not validated ASR support:

- **Latin: 24.** Croatian, Czech, Danish, Dutch, English, Estonian, Finnish,
  French, German, Hungarian, Italian, Latvian, Lithuanian, Maltese, Norwegian,
  Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish, Turkish
  and Vietnamese.
- **Latin + Indic: 47.** The 24 above, Arabic, and these 22 Indic profiles:
  Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri (Arabic),
  Konkani, Maithili, Malayalam, Manipuri (Meetei Mayek), Marathi, Nepali, Odia,
  Punjabi (Gurmukhi), Sanskrit, Santali (Ol Chiki), Sindhi (Devanagari), Tamil,
  Telugu and Urdu.
- **Original: 35.** The 24 Latin profiles, Arabic, Hindi, Bulgarian, Greek,
  Hebrew, Japanese, Korean, Mandarin Chinese, Russian, Thai and Ukrainian.
- **Full: 56.** All Original and Indic profiles.

Coverage is finite, and unknown characters can produce `<unk>`. Nynorsk,
alternate Indic scripts and broad dialect coverage are not established. The
original normalizer turns ZWNJ into a space; decoding returns normalized text.
Language prompts condition the model but do not enforce an output language.
Preserved token IDs and weights do not guarantee unchanged speech accuracy.

[Coverage evidence](docs/language-coverage.md) ·
[ID contract](docs/native-preservation.md) ·
[Rebuild](docs/native-reproduction.md) ·
[Source provenance](THIRD_PARTY.md)

Project code: [Apache-2.0](LICENSE). Tokenizer assets retain their upstream terms.
