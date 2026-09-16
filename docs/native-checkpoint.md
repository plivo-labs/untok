# V3 native checkpoint migration

All three packaged profiles preserve all 13,087 original Nemotron text IDs,
complete piece messages and normalization metadata. The acoustic blank moves
from 13087 to the expanded vocabulary's final output row; migration copies its
learned weights exactly. Existing original text labels retain their meaning.
Regenerate labels to use new segmentation. The earlier compact v2 candidate is
withdrawn and must not be used for the original-ID requirement.

All three v3 checkpoints were migrated from the pinned original NVIDIA checkpoint
and verified through real NeMo save/reload on 16 September 2026. Each preserved all
638,030,384 original learned values across 657 tensors with exact equality, before
save and after reload. No original text row was removed or renumbered.
[Migration results and checkpoint hashes](../configs/native-checkpoint-v3-results.json)
record the runtime, bundle identities and complete receipt hashes.

| Local checkpoint | Original text IDs | Acoustic blank migration |
| --- | --- | --- |
| `artifacts/native-checkpoints-v3/full.nemo` | `0..13086` unchanged | `13087` → `20784` |
| `artifacts/native-checkpoints-v3/latin-indic.nemo` | `0..13086` unchanged | `13087` → `20784` |
| `artifacts/native-checkpoints-v3/latin.nemo` | `0..13086` unchanged | `13087` → `13573` |

Checkpoints and their full `.migration.json` receipts are local generated artifacts,
outside Git and the Python package. This verification used CPU execution with the
pinned NeMo source revision below, Python 3.12.14, PyTorch 2.8.0 and SentencePiece 0.2.1.
Paired acoustic validation and training have not been run for v3.
[Published speech compatibility results](native-compatibility-results.md) apply
to v1, whose migration also moved blank to its final output row.

Migration accepts the pinned original native `.nemo` checkpoint from
`nvidia/nemotron-3.5-asr-streaming-0.6b`, revision
`1c8deaecc64b91f034d73e08dd8b64625eb3395d`.

Expected checkpoint SHA256:
`210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`.
The embedded SentencePiece model is Unigram. NeMo's class names use `BPE`
for this SentencePiece integration, but that does not determine the model's
actual segmentation algorithm.

It also accepts a checkpoint containing the exact original **full v1** Untok
tokenizer, after verifying the supplied checkpoint SHA-256 and embedded
tokenizer bytes. Migration from a v1 reduced Latin or Latin-plus-Indic
checkpoint is not supported. An original-base migration and a full-v1 migration
use different source row maps; they cannot share an assumed numeric ID layout.

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

Choose `full` or `latin` in the path command to migrate the other bundles. The destination
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

Every v3 profile retains all original text rows at their original indices.
Migration copies those rows and relocates the original blank row. For a full-v1
source, added rows outside the selected extension profile may be omitted; no
original Nemotron text row may be omitted. The exact original model metadata and
normalizer are verified. Retained raw logits can stay equal while adding output
classes changes softmax probabilities or unrestricted predictions.

Full and Latin-plus-Indic contain 20,784 text entries; Latin contains 13,573.
The native acoustic blank is respectively 20,784 or 13,573. The public pad and
blank IDs remain 13,087 and 13,088; use the supplied mapping rather than passing
public IDs directly as native training labels.

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
| V3 full or Latin plus Indic | `▁भारत` |

The v3 text tokenizer can select `▁भारत` immediately. When migrating from
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
Every profile retains all original Hindi pieces and other original scripts.
Full and Latin-plus-Indic add the selected Indic bank; Latin filters only new
additions. A full-v1 source may already have trained rows for surviving additions. Passing
compatibility checks establishes preservation on the tested inputs, not that
the new pieces have been learned.

## Speech integration checks still required for v3

`scripts/native_checkpoint_probe.py` contains full-source-preservation controls.
Original-base to v3 mappings retain every source row, so that path is applicable.
`scripts/native_subset_probe.py` also accepts these identity-prefix mappings
and historical reduced layouts. Its source checks bind the actual original-base
or full-v1 tokenizer and row map to the migration receipt.

`inference` exercises explicit target prompts and reports raw diagnostic WER/CER.
Its success flag means inference executed, not that recognition accuracy passed.
Small samples and regional proxy recordings do not establish all-locale accuracy.

`scripts/native_subset_probe.py` provides the retained-output comparison path
for all three v3 profiles, including full. It selects the original
NVIDIA or full-v1 source inventory from exact tokenizer bytes and verifies that
the migration receipt declares the same source hash and row mapping. It requires
a completed migration receipt and immutable audio hashes. This is a command for
a new v3 run, not an already-passed result:

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
have CPU tests for both source inventories and all three profiles; no actual v3
paired audio run has been completed.

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
