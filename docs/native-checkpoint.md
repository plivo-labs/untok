# V5 checkpoint compatibility and migration

Every included Nemotron piece keeps its original native ID, score and type.
Subset exclusions occupy reserved inactive slots rather than compacting IDs or
removing checkpoint rows. The Untok NeMo runtime must mask those inactive outputs.

| Profile | Active text pieces | Physical text slots | Inactive rows | Native RNNT blank |
| --- | ---: | ---: | ---: | ---: |
| `original` | 13,087 | 13,087 | 0 | 13,087 |
| `latin` | 2,653 | 13,087 | 10,434 | 13,087 |
| `latin-indic` | 10,372 | 20,360 | 9,988 | 20,360 |
| `full` | 20,360 | 20,360 | 0 | 20,360 |

`original` contains the exact original tokenizer bytes and can use the original
checkpoint. Latin retains its physical shape but requires the subset tokenizer
and mandatory inactive-output mask. The Indic profiles add 7,273 native text rows
beginning at 13087 and move the learned acoustic blank from 13087 to 20360.
Blank remains the final output row after physical text slots, not active pieces.

Migration copies every original Nemotron row value, including inactive rows.
Keeping those rows does not permit predicting their excluded spellings: the
SentencePiece slots are `UNUSED`, and acoustic logits for those IDs are masked.
The runtime mask is part of the model contract during training and decoding;
a generic NeMo loader that omits it does not implement the subset correctly.
Stable IDs do not reduce the acoustic model's vocabulary dimension.

All profiles keep the original normalizer. Original labels for included pieces
remain meaningful at the same IDs; labels for inactive pieces are not valid
subset labels. Regenerate labels from text when changing active repertoire or
using the new Indic segmentation.

Migration accepts the pinned original `.nemo` checkpoint from
`nvidia/nemotron-3.5-asr-streaming-0.6b`, revision
`1c8deaecc64b91f034d73e08dd8b64625eb3395d`, with SHA256
`210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`.
Its embedded tokenizer is Unigram despite NeMo's `BPE` class naming.

The exact historical full-v1 tokenizer is also an accepted source after its
checkpoint hash and embedded tokenizer are verified. Full-v1 additions use their
explicit source-to-target maps; the original-Nemotron identity guarantee does
not promise stable IDs for every historical Untok addition. A historical reduced
checkpoint is not a supported source.

## Evidence by version

The [v5 migration report](../configs/native-checkpoint-v5-results.json) records
real construction, save and reload for all four profiles on CPU. Each migration
preserved all 638,030,384 original values across 657 tensors exactly, both before
saving and after reloading, with no omitted values. This includes inactive rows
and the learned blank row. New-row initialization also passed both checks.

The runtime mask passed before save and after reload for all 10,434 inactive
Latin rows and 9,988 inactive Latin-plus-Indic rows. Original and Full have no
inactive outputs. The report binds model, manifest, migration receipt, checkpoint
and implementation hashes.

Four [optional RNNT integration tests](../tests/test_native_runtime.py) also passed
in the pinned NeMo environment on CPU: standard and fused joint/loss paths with
the PyTorch and numba loss backends produced finite forward/backward results and
zero gradients for inactive output rows. These small joint/loss tests do not
establish complete acoustic-model training. GPU execution, CUDA graph replay,
paired speech-output parity and speech accuracy have not been validated for v5.

The [v4 migration report](../configs/native-checkpoint-v4-results.json) records
successful construction and save/reload of compact v4 subsets, which omitted
excluded rows. The [v3 report](../configs/native-checkpoint-v3-results.json) and
[v1 speech checks](native-compatibility-results.md) are historical too. None
validates the current v5 artifacts. Speech accuracy and training remain separate
from the current structural and tensor-preservation checks.

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

Migration verifies the source hash, actual native tokenizer bytes, original
tensor rows including inactive rows, new-row initialization, saved tokenizer
artifacts and restored checkpoint. The adjacent `.migration.json` records the checks and
the explicit source-to-target row map. Original Nemotron rows keep their numeric
positions even when inactive; the source RNNT blank maps to the target's final
output row. Historical source additions outside the target may map to `null`. A successful run
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
The mandatory v5 inactive-slot mask enforces the selected profile. It does not
provide an arbitrary per-request English-Hindi filter inside Latin-plus-Indic.

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

All four profiles retain original Nemotron row positions. The subset models
copy inactive rows as well as active rows; only active rows are allowed outputs.
Physical text slots span `0..13086` for Original/Latin and `0..20359` for the
Indic profiles. Only added Indic rows expand the model and move blank.

Public padding is 13087 and public blank is 13088 for every profile. New public
text IDs begin at 13089. Use the supplied public/native mapping. Inactive labels
must be rejected rather than treated as ordinary text or blank.

Original row values can be preserved while masking or adding output classes
changes probabilities and predictions. Token-ID identity is not a claim of
unrestricted speech-output parity.

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

For example, the original and historical v4 tokenizers encoded `भारत` as follows. The `▁`
symbol marks a word boundary.

| Tokenizer | Pieces |
| --- | --- |
| Original | `▁`, `भा`, `र`, `त` |
| Historical v4 full or Latin plus Indic | `▁भारत` |

That expanded text tokenizer can select `▁भारत` immediately. When migrating from
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
`scripts/native_subset_probe.py` provides source-bound comparison controls.
Its current-artifact run must account for inactive rows and their runtime mask. Source checks bind the actual
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
Masked output rows can change unrestricted predictions even when every
retained row was copied exactly. For a full-v1 migration, supply the same full-v1
source checkpoint used during migration. Source selection and receipt checks
must bind the actual source inventory and current profile. A passing historical
run does not establish paired audio behavior for v5.

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
