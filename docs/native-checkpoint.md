# V4 checkpoint compatibility and migration

Choose a checkpoint whose tokenizer matches the selected profile:

| Profile | Compatibility with original Nemotron |
| --- | --- |
| `original` | Exact original tokenizer bytes and text IDs; use the original checkpoint |
| `latin` | Compact Latin subset; retained rows require remapping and excluded rows are removed |
| `latin-indic` | Compact Latin-plus-Indic vocabulary; retained rows require remapping and added rows need training |
| `full` | Original text IDs remain fixed; added Indic/shared rows expand the vocabulary |

The native text vocabulary sizes and acoustic blank IDs are 13,087 (`original`),
2,653 (`latin`), 10,372 (`latin-indic`) and 20,360 (`full`).

Every profile uses the original SentencePiece normalizer. The native acoustic
blank is the final row at the selected text vocabulary size. Migration copies
the source blank weights to that row. An unchanged original text ID does not
imply unchanged segmentation or predictions after extending the vocabulary.
For subsets, old token arrays need an explicit row map or retokenization from
original text; removed labels cannot be fixed by substituting another numeric ID.

Migration accepts the pinned original native `.nemo` checkpoint from
`nvidia/nemotron-3.5-asr-streaming-0.6b`, revision
`1c8deaecc64b91f034d73e08dd8b64625eb3395d`, with SHA256
`210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`.
The embedded model is Unigram; NeMo's use of `BPE` in its SentencePiece integration
class names does not change that algorithm.

A source checkpoint containing the exact historical **full v1** tokenizer is
also supported after its checkpoint hash and embedded tokenizer are verified.
Original-base and full-v1 migration use different row maps. Migration from a
historical reduced Latin or Latin-plus-Indic checkpoint is not supported.

## Current migration evidence

All four v4 checkpoints passed real NeMo construction, save and reload on
16 September 2026. Exact retained-tensor comparisons passed across 657 tensors,
both before save and after reload; new-row initialization and tokenizer identity
also passed. `original` remains byte-identical to the original tokenizer.

| Profile | Original text rows retained | Source text rows omitted | New text rows initialized | Native blank |
| --- | ---: | ---: | ---: | ---: |
| `original` | 13,087 | 0 | 0 | 13,087 |
| `latin` | 2,653 | 10,434 | 0 | 2,653 |
| `latin-indic` | 3,099 | 9,988 | 7,273 | 10,372 |
| `full` | 13,087 | 0 | 7,273 | 20,360 |

`original` and `full` preserve all 638,030,384 original learned values. Subsets
intentionally omit excluded text rows and verify every retained value exactly.
Each native blank is the final output row; `original` keeps index 13087 unchanged.
The [v4 results](../configs/native-checkpoint-v4-results.json) record checkpoint,
model, bundle and receipt hashes, plus the pinned CPU NeMo runtime. Generated
checkpoints and complete receipts are local artifacts under
`artifacts/native-checkpoints-v4/`, outside Git. This run did not evaluate speech
accuracy, a training forward pass or logit parity.

## Historical evidence

The [v3 migration report](../configs/native-checkpoint-v3-results.json) records
real NeMo save/reload verification of the superseded v3 models. Those models
retained the entire original bank in every profile; they are not the v4 subsets.
The [v1 speech results](native-compatibility-results.md) are historical as well.
Neither report qualifies a newly built v4 artifact. A current migration's
`.migration.json` receipt must bind its exact source, target model and bundle
hashes before it is treated as verified. Speech accuracy and training require
separate evaluation.

## Migrate

After installing `untok` in your NeMo environment, select the installed bundle's
directory. Migration takes a directory path; the short names accepted by
`load_tokenizer()` are for text tokenization.

```sh
BUNDLE=$(python -c 'from importlib.resources import files; print(files("untok").joinpath("data", "latin-indic"))')
untok check --bundle "$BUNDLE"
```

Migration requires the compatible NVIDIA NeMo Speech runtime. The historical
v1 integration environment used Python 3.12, PyTorch 2.8.0 with CUDA 12.8, SentencePiece 0.2.1
and NeMo Speech revision `ca4daa1470f6c01068c4e6a9a73b19b9a91dc366`.
The `checkpoint` extra supplies tensor utilities, not the complete NeMo stack.

```sh
untok migrate \
  --source nemotron-3.5-asr-streaming-0.6b.nemo \
  --source-sha256 210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74 \
  --bundle "$BUNDLE" \
  --output nemotron-latin-indic.nemo
```

Choose `full` or `latin` in the path command to migrate those bundles.
`original` can use the pinned original checkpoint directly. The destination
and migration report must not exist. The source checkpoint is never overwritten.

Migration verifies the source hash, actual native tokenizer bytes, every
retained tensor row, new-row initialization, saved tokenizer artifacts and
restored checkpoint. The adjacent `.migration.json` records the checks and
the explicit source-to-target row map. A removed source text row maps to `null`;
the source RNNT blank maps to the target's final output row. A successful run
verifies that particular migration; it does not establish speech accuracy.

For a full-v1 source, pass that checkpoint's path and independently recorded
SHA-256 to the same command. Its tokenizer must match
`f987a99ce9448ca72bb2da11f36744254f9f9b12f5596fcb742ddedf950886a8`.
Surviving trained v1 additions then retain their rows; only target pieces absent
from that source require initialization.

Restore a migrated checkpoint with the installed `untok` package:

```python
from untok.native_runtime import get_native_nemo_model_class, transcribe_native_file

model = get_native_nemo_model_class().restore_from(
    "nemotron-latin-indic.nemo", map_location="cuda"
)
hypotheses, prompt_evidence = transcribe_native_file(
    model, "speech.wav", target_lang="hi-IN"
)
print(hypotheses[0].text)
```

This helper verifies the language prompt actually supplied to the model. The
pinned NeMo file-list loader can select a unified prompt, so this path loads
the file as tensor audio and sets the requested prompt explicitly. It also
leaves all model layers in evaluation mode. NeMo's transcription teardown can
otherwise re-enable training mode in submodules before a later streaming call.

## Language preferences and choosing two languages

Language preferences belong in model inference and decoding, not in
SentencePiece normalization or piece scores. Each checkpoint has one shared
output vocabulary. A language prompt changes the model's conditioning without
automatically masking other languages' pieces.

The current helper accepts one `target_lang` string. For example,
`target_lang="en-US"` uses English prompt slot 0, and `target_lang="hi-IN"`
uses Hindi slot 6. `target_lang="auto"` selects slot 101; it is an explicit
conditioning mode, not an absent or all-zero prompt. The default greedy
decoder chooses the highest-scoring output rather than sampling randomly.

| Desired behavior | Where it belongs | Available now |
| --- | --- | --- |
| Nemotron's learned English preference | Pass `target_lang="en-US"` to the inference helper | Yes |
| An adjustable additional English preference | A decoder policy that adjusts token scores before selection | No |
| Restrict available pieces to an English-Hindi set | A decoder policy that excludes pieces outside the permitted set | No |

An additional decoder policy could be exposed through
[`transcribe_native_file`](../src/untok/native_runtime.py), with its scoring
logic in a separate decoder module. It would operate at runtime without
rewriting the tokenizer or checkpoint weights. It would need validation for
both offline and streaming decoding, including any enabled CUDA graph path.
The masks in the existing checkpoint tests are validation controls, not a
supported language-lock feature.

For an English-Hindi option, keep the model prompt and the permitted output
set separate. A future `allowed_languages=["en-US", "hi-IN"]` setting could
be paired with `target_lang="auto"`, or with one primary language such as
`"hi-IN"`. This setting is a proposal and is not an accepted argument today.
Do not pass a list to `target_lang` or activate two one-hot positions; the
current interface and checkpoint expect one prompt index.

A practical output filter could keep Latin and Devanagari pieces, shared
punctuation, numbers, word-boundary pieces and RNNT blank. Output language tags
need explicit handling as well. This is a script restriction, not a guarantee
of English and Hindi wording: Latin also represents French and romanized
Hindi, while Devanagari is shared with Marathi, Nepali and other languages.
Shared pieces must not be assigned exclusively to one language.

## What is preserved

`full` retains every original text row at its original index. The two subsets
retain only selected rows and use their explicit compact mappings. Migration
verifies retained tensor values exactly and relocates the learned acoustic blank.
For a full-v1 source, only surviving source additions keep their learned rows.
The unchanged source normalizer is verified for every profile.

Native text IDs are dense `0..N-1`, and native acoustic blank is `N`. `original`
and `full` retain public pad/blank IDs 13087/13088. The compact subsets use public
pad `N` and public blank `N+1`. Use each bundle's supplied public/native map.
Retained logits may match while removed or added classes change softmax
probabilities and unrestricted predictions.

New output weights initially copy the original blank row with a lower bias.
The combined new-output mass is bounded relative to retained old outputs by
`1e-6`, before training. New predictor rows start from the mean retained text
embedding. These rows remain independent and trainable.

The original prompt slots stay fixed. Missing target identities receive unused
slots in the existing prompt dimension. This enables the conditioning path;
it does not teach a new language or impose an output-language mask.

## Using newly added Hindi pieces

Migration makes the new IDs valid decoder outputs, but does not teach the
model when to emit them. This applies to new pieces for Hindi even though the
original checkpoint already recognizes Hindi through its original vocabulary.

For example, the actual native tokenizers encode `भारत` as follows. The `▁`
symbol marks a word boundary.

| Tokenizer | Pieces |
| --- | --- |
| Original | `▁`, `भा`, `र`, `त` |
| V4 full or Latin plus Indic | `▁भारत` |

The expanded v4 text tokenizer can select `▁भारत` immediately. When migrating from
the original NVIDIA checkpoint, that new output row has no learned acoustic
association with the word yet. Under
the current initialization, new rows have the blank row's weights and a lower
bias, so ordinary greedy decoding will not select them without subsequent
weight changes or an additional scoring intervention. Changing the language
prompt does not close that score gap. Simply boosting a new row's score would
not teach it the word's acoustic meaning.

To learn meaningful use of the new pieces, a future checkpoint fine-tuning
step would use speech transcripts encoded by the updated tokenizer. This
teaches the new predictor and output rows; it is not another tokenizer change.
`original` and `full` retain the entire original multilingual bank.
Latin-plus-Indic retains the permitted Indic pieces; Latin excludes Indic scripts.
Full and Latin-plus-Indic add the selected Indic bank. A full-v1 source may already have trained rows for surviving additions. Passing
compatibility checks establishes preservation on the tested inputs, not that
the new pieces have been learned.

## Speech integration checks for current artifacts

`scripts/native_checkpoint_probe.py` contains controls that require all source
rows to survive. It is appropriate only for mappings with that property.
`scripts/native_subset_probe.py` supports retained-output comparisons for compact
subsets as well as full-source-preserving layouts. Source checks bind the actual
original-base or full-v1 tokenizer and row map to the migration receipt.

`inference` exercises explicit target prompts and reports raw diagnostic WER/CER.
Its success flag means inference executed, not that recognition accuracy passed.
Small samples and regional proxy recordings do not establish all-locale accuracy.

`scripts/native_subset_probe.py` provides the retained-output comparison path
for the current derived profiles, including full. It selects the original
NVIDIA or full-v1 source inventory from exact tokenizer bytes and verifies that
the migration receipt declares the same source hash and row mapping. It requires
a completed migration receipt and immutable audio hashes. This is a command for
a fresh current-artifact run, not historical passing evidence:

```sh
python scripts/native_subset_probe.py --mode both \
  --source nemotron-3.5-asr-streaming-0.6b.nemo \
  --checkpoint nemotron-latin-indic.nemo \
  --migration nemotron-latin-indic.migration.json \
  --manifest audio-evaluation.jsonl \
  --output reports/native-subset.json
```

This compares the target model with the original model's retained outputs.
The manifest should include the desired locale paths and immutable audio hashes.
Removed output rows can change unrestricted predictions even when every
retained row was copied exactly. For a full-v1 migration, supply the same full-v1
source checkpoint used during migration. Source selection and receipt checks
must bind the actual source inventory and current profile. A passing historical
run does not establish paired audio behavior for the v4 subsets.

## Work required for a trained release

Fine-tuning is a separate task, outside compatibility validation.

Use the Full migration for the initial continuation-training experiment, with
new-language speech and replay from original languages. Keep speaker-disjoint
development and test audio, preserve complete transcript labels, and record
the training hours and source mix per language. Tune learning rate, sampling
and stopping only on development audio.

Evaluate final WER/CER against the original checkpoint on sufficiently large
held-out sets for the original locales and the 22 new text profiles. Include
conversation, read speech, code-switching and streaming. Report absent dialects
or scripts explicitly. A few optimizer steps establish the training path but
cannot replace this fine-tuning and accuracy evaluation.
