# Nemotron checkpoint usage

Untok prepares NVIDIA Nemotron 3.5 ASR Streaming 0.6B for a selected tokenizer.
NVIDIA's native scripts then own training and evaluation. Use the environment
and pinned NVIDIA revision documented in [indic-asr](https://github.com/plivo-labs/indic-asr).
Installing Untok's `checkpoint` extra alone does not install NeMo.

The supported source is the original `.nemo` checkpoint from
[NVIDIA's pinned release](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/ea30d66debe3740a08b573244286791d423d6b3e).
Its embedded SentencePiece model is Unigram, despite NeMo's BPE class names.
Migration from historical Untok checkpoints is not supported.

## Migrate

```sh
untok migrate \
  --source nemotron-3.5-asr-streaming-0.6b.nemo \
  --source-sha256 210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74 \
  --bundle latin-indic \
  --output nemotron-latin-indic.nemo
```

Choose `original`, `latin`, `latin-indic` or `full`, or a bundle directory.
The output checkpoint and its `.migration.json` report must not already exist.
The source is never overwritten. The `original` tokenizer can also use NVIDIA's
original checkpoint directly.

Migration checks the source hash and embedded tokenizer, copies every original
weight, initializes additions, and verifies the saved checkpoint after reload.
The learned blank row moves only when the vocabulary expands:

| Bundle | Physical text slots | Inactive slots | Native RNNT blank |
| --- | ---: | ---: | ---: |
| `original` | 13,087 | 0 | 13,087 |
| `latin` | 13,087 | 10,434 | 13,087 |
| `latin-indic` | 20,360 | 9,988 | 20,360 |
| `full` | 20,360 | 0 | 20,360 |

Original text IDs remain `0..13086`. New native text IDs are `13087..20359`.
Blank follows all physical text slots, not just active pieces. The predictor's
padding row is the acoustic blank; SentencePiece itself keeps `pad_id=-1`.

Subsets retain excluded checkpoint rows but mark their tokenizer slots UNUSED
and mask their acoustic outputs. Keep the Untok runtime installed when restoring
a migrated checkpoint. SentencePiece UNUSED entries alone cannot mask a model's
output head. Regenerate training labels from text with the chosen tokenizer.

## New-token initialization

For each addition, migration finds the highest-scoring exact sequence of active
original NORMAL pieces and averages their embedding, output-weight and bias
rows. If no sequence exists, it uses the mean of all active original NORMAL
pieces. Blank, special and inactive pieces are excluded as donors.

New output biases receive a single shift that bounds their combined initial
mass relative to retained text outputs. The default ratio is `0.05`, adjustable
with `--max-new-mass-ratio`. For Latin + Indic, 1,462 additions have exact donor
paths and 5,811 use the fallback; the default shift is **-8.064528728662534**.
Original rows, including blank, are unchanged. There is no blank penalty.

This is an initialization bound, not a training constraint or accuracy guarantee.
An averaged embedding does not reproduce the recurrent state of its donor
sequence. During training, the native RNNT loss rewards the updated target IDs,
and the predictor receives those IDs as its reference history. Newly added Hindi
pieces need acoustic learning even though the base model already recognizes Hindi.

## Training

From the [indic-asr checkout](https://github.com/plivo-labs/indic-asr), migration
can embed model settings and write a complete native training YAML:

```sh
untok migrate \
  --source nemotron-3.5-asr-streaming-0.6b.nemo \
  --source-sha256 210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74 \
  --bundle latin-indic \
  --model-config configs/model-settings.yaml \
  --training-template vendor/nemo/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml \
  --training-overrides configs/untok-latin-indic.yaml \
  --output models/untok-initial.nemo
```

The paths above belong to indic-asr. The generated `untok-initial.train.yaml`
uses the restored checkpoint architecture and prompts, plus native dataset,
optimizer and Trainer settings. Tokenizer replacement is disabled because
migration has already installed it. Follow indic-asr's native training command.

The frozen-encoder recipe freezes encoder and preprocessor parameters while
leaving the native training mode and context sampling active. The predictor
(also called the decoder), joint and language prompt network train. Freezing the
encoder does not preserve their learned output mappings during optimization.

## Evaluation and prompts

Use NVIDIA's cache-aware evaluator through indic-asr's registration launcher:

```sh
python -m indic_asr.evaluate \
  model_path=models/untok-initial.nemo \
  dataset_manifest=data/test-hi.jsonl \
  output_path=runs/evaluation-hi \
  target_lang=hi \
  'att_context_size=[56,0]'
```

The manifest must describe complete recordings with matching reference text.
The evaluator applies one global `target_lang` to the whole run; it does not
route each row by its language field. Use separate language runs, or `auto` for
mixed speech. See [native evaluator details](https://github.com/plivo-labs/indic-asr/blob/main/docs/evaluation.md)
for scoring and input limitations.

`en`/`en-US` select prompt slot 0, `hi`/`hi-IN` select slot 6, and explicit
`auto` selects slot 101. Prompts provide learned conditioning over the shared
vocabulary. They do not guarantee the output script or enable a hard language
lock. Passing two languages or setting two prompt positions is not supported.
Such output restrictions would be a separate decoder feature, not a tokenizer
setting. The mandatory inactive-slot mask only enforces the selected bundle.

## Validation limits

The recorded single-GPU integration checks covered migration, restoration,
native training, optimizer resume and streaming evaluation. They did not
establish accuracy preservation: after 36 updates on 0.627 hours of speech,
explicit-Hindi WER rose from 21.52% to 35.63%, mainly through deletions. The cause
and eventual convergence remain unqualified; no trained replacement is released.

The restricted Latin + Indic profile can change original predictions when they
contain excluded characters. Exact weight copying does not guarantee identical
unrestricted transcripts. See [recorded verification](https://github.com/plivo-labs/indic-asr/blob/main/docs/verification.md)
for the evaluated cohorts, comparison settings and multi-GPU gap. Historical
results belong to their recorded code and artifact hashes, not future changes.
