# Adding Indic languages to a checkpoint

An Untok `latin-indic` checkpoint can write every Indic script, but two parts of it are not ready to learn a
language it was never trained on:

- **Prompt slots.** Every Indic language has a slot (`ta-IN` 39, `te-IN` 40, `bn-IN` 36, ... `pa-IN` 79), but only
  the slots NVIDIA or a later fine-tune trained (for example `hi-IN`, `en-US`, `auto`) carry weights. The others are
  near zero, and a near-zero slot erases most of the audio information after the encoder.
- **Rows for scripts NVIDIA never wrote.** Migration gives an added piece the average of the original pieces that
  spell it. Tamil, Telugu, Bengali, Gujarati, Gurmukhi, Odia, Kannada, Malayalam, Ol Chiki and Meetei Mayek pieces
  (and most added Arabic pieces) have no such spelling, so all of their rows start as one identical copy. Identical
  rows score every frame the same, blank keeps winning, and training stalls.

Two commands fix both, before training, on a fresh migration or on a checkpoint that was already fine-tuned:

```sh
untok init-languages --source trained.nemo --source-sha256 <sha256> --output languages.nemo
untok calibrate --source languages.nemo --source-sha256 <sha256 of languages.nemo> --hours 12 \
  --output languages-calibrated.nemo \
  --manifest ta-IN=data/ta-IN.jsonl --manifest te-IN=data/te-IN.jsonl --manifest bn-IN=data/bn-IN.jsonl ...
```

Pass `calibrate` manifests only for languages of new scripts (scripts NVIDIA never wrote). Devanagari and Arabic
languages need none (see [Leftover Devanagari and Arabic rows](#leftover-devanagari-and-arabic-rows)).

Both write a new checkpoint, its `.tokenizer` directory and a JSON report next to it, and never overwrite. Check the
result against the [gate](#gate-before-training) before training.

## init-languages

By default it prepares all 22 Indic languages of the profile (`--languages ta,te,bn-IN` for a subset):

- **Prompt slot.** Each untrained slot becomes a copy of the trained `auto` column; each language then learns its own
  slot from its own data. Slots that already look trained (norm above 2, against about 0.57 for untrained and 13.5
  for trained slots) are kept unless `--force-prompt-slots` is given. Copying `auto` was the starting point that
  worked best for Tamil and Telugu: starting from Hindi's slot hurt Hindi under `auto`, and pushing slots apart
  stopped `auto` from ever choosing the new languages.
- **Rows.** In each language's script, the rows that are still identical copies are re-initialised. They start at
  the mean of trained rows with biases 8 below those rows' mean bias (silent), and `calibrate` then gives each token
  its own offset, scaled so the rows spread out as much as the trained rows do, with biases at the trained rows' level.
  The trained rows come from the same script when NVIDIA's vocabulary had it (Arabic, Devanagari), otherwise from
  Devanagari. Decoder embedding rows of these tokens are drawn from N(0, 1). In a checkpoint that already writes
  Devanagari or Arabic, only the leftover copies are re-initialised, and they stay dormant (next section).
- Every other value stays bit-identical; the command verifies this.

Until calibration, every such row is exactly the mean of the trained rows with its bias 8 lower, so it stays silent:
in a pilot on `plivo/untok-finetune-nemotron`, giving uncalibrated rows random offsets instead made Odia and Bengali
pieces fire on 90% of Hindi clips. Rows of a new script without fitted offsets cannot learn either (Tamil and Telugu
wrote nothing after 1,500 updates from random offsets), which is why calibration is required for new scripts.

## Leftover Devanagari and Arabic rows

Rows are shared by script. In a checkpoint fine-tuned on Hindi, a few rare Devanagari pieces (57 in
`plivo/untok-finetune-nemotron`) are still copies, and so are the 696 leftover Arabic copies. Their script is one NVIDIA
wrote, so `init-languages` re-initialises them for Hindi, Marathi, Nepali, Sanskrit and the other Devanagari languages
(and for Urdu and Kashmiri) like any other copy, but then:

- they stay dormant: exactly the trained rows' mean, bias 8 below the trained rows' mean bias;
- those languages are marked `calibrated: dormant_leftovers` in the record and the `.languages.json` report;
- `calibrate` skips their manifests and says so in its report, and the training guard accepts them;
- they learn from training targets: `NewSliceLR` puts them in the fast slice with the new scripts' rows. In the
  fine-tune that produced `plivo/untok-finetune-nemotron`, the `ळ` row (a character NVIDIA never had) started as such a
  copy and was learned from training targets; it is no longer a copy.

Calibrating them from audio broke the checkpoint. The 57 Devanagari copies are a rare tail: in a sample of 765
Marathi sentences only 12 of them occur (`ळी`, `▁मिळ`, `▁वेळ`, `मुळे`, ... and `ॲ`). A 1.5-hour Marathi calibration gave
Devanagari a scale (gamma) of 7.13, where the new scripts got 1.29 to 1.66 from the same amount of audio, and pushed
their biases up to +26.9 (the highest trained Devanagari bias is 0.91). The checkpoint then wrote these pieces over
everything: on 20 English and 20 Hindi check clips (`en-US` and `hi-IN` prompts, attention context [56, 13]), WER went
from the source checkpoint's 27.03 and 28.92 to 5,015.54 and 3,573.49. With the leftovers dormant and the new scripts
calibrated on 12 hours each, the same clips give 27.03 and 28.92 again. Arabic follows the same rule; its leftovers
were not tried with audio calibration.

## calibrate

`calibrate` fits each token's offset from audio, for every new script (one NVIDIA never wrote) that you pass a
manifest for. Calibrate only the new scripts, with 12 hours of audio each (the default `--hours`):

- It runs the frozen checkpoint on up to 12 hours of the language's audio (`--hours`) with the `auto` prompt and
  attention context [56, 1], and takes the joint network's hidden states with the prediction network at its start.
- It trains a linear CTC probe on those states (15 epochs, AdamW, learning rate 2e-3, warmup then cosine).
- Each token's probe weights relative to the probe's blank become its offset. Offsets are centred and scaled to the
  trained rows' spread as above. Tokens the probe never saw get the mean offset and the lowest seen bias offset minus 2, before the same scale.
- Only the initialised rows and biases change; their biases move from dormant to the trained rows' level. The report
  records the clips, hours, probe dev CER, scale and the manifest hashes.

Use the same text policy as training (below); calibration applies it to the manifest text itself.

Bengali and Assamese share the Bengali rows, so their manifests are pooled into one 12-hour calibration (about 6 hours
each when both are given).

## Gate before training

Check the calibrated checkpoint before spending compute on it:

- **Scale.** Every script's `gamma` in the `.calibration.json` report should be at most about 0.85. On
  `plivo/untok-finetune-nemotron` with 12 hours per script, the eight new scripts got 0.60 to 0.81. With 1.5 hours
  they got 1.29 to 1.66 and Gujarati pieces leaked into Hindi output (Hindi WER on the 20 check clips 28.92 -> 39.16).
- **Start equals source.** On Hindi and English clips, with the `hi-IN`, `en-US` and `auto` prompts, the calibrated
  checkpoint must give the same output as the checkpoint `init-languages` started from. Dormant and well-calibrated
  rows do not win on those languages, so any difference means some rows fire where they should not. The 12-hour
  checkpoint above matched its source on 20 English, 20 Hindi and 20 more Hindi clips.

## Training guard and learning rates

`untok.training.NewSliceLR` is a Lightning callback listed in the training YAML (see indic-asr's
`configs/untok-new-languages.yaml`). Before training it reads the init checkpoint's `untok_languages` record and
refuses any listed language whose initialised rows were not calibrated (`calibrated: false`); languages marked
`dormant_leftovers` pass. During training it gives the initialised rows, their decoder embeddings and their prompt
slots their own learning rates (3e-3, 1e-3, 1e-3 by default) while the shared layers keep the YAML's rates: NeMo can
set a rate per module, not for a few rows inside one layer. `untok.nemo.register()` allows this target in NeMo's YAML
instantiation.

## Training text

`untok normalize-manifest` applies the policy used with this recipe:

- no language tags (`<hi-IN>`, `<unintelligible>`, ...) in any target;
- Indic text without punctuation, and numbers spelled out in the row's language with AI4Bharat's
  [indic_numtowords](https://pypi.org/project/indic-numtowords/) (`pip install untok[text]`). Thousands separators are
  dropped first and each run of digits is read as one number;
- English keeps its case, punctuation and digits.

`indic_numtowords` 1.1.0 has number words for all 22 Indic languages of the profile. Runs longer than nine digits
are read digit by digit. A number it cannot spell (for example some Sindhi numbers) keeps its digits, and the report
counts it.

`untok join-clips` writes two clips of different languages joined by 0.1-0.5 s of silence, as code-switched audio
for the `auto` prompt. Normalise the source manifests first; the joined text keeps each part as written.

## Record

Both commands store their record in the checkpoint config under `untok_languages`:

```yaml
untok_languages:
  languages:
    ta-IN: {script: Tamil, prompt_slot: 39, prompt_init: auto_copy, rows: initialized, calibrated: true}
    gu-IN: {script: Gujarati, prompt_slot: 42, prompt_init: auto_copy, rows: initialized, calibrated: false}
    mr-IN: {script: Devanagari, prompt_slot: 41, prompt_init: auto_copy, rows: initialized, calibrated: dormant_leftovers}
  scripts:
    Tamil: {rows: [...], donors: [...], donor_script: Devanagari, init: {...}, calibration: {...}}
    Devanagari: {rows: [...], donors: [...], donor_script: Devanagari, init: {...}, calibration: null}
```

`calibrated` is `true` or `false` for languages of a new script, `dormant_leftovers` for leftover Devanagari and
Arabic copies, and `not_required` when the language's script has no copies left.

Training with NVIDIA's script loads only the weights of `init_from_nemo_model`, so the callback reads this record
from that file and copies it into the checkpoints it saves.

## Limits

- Calibration quality grows with data; 12 hours per script is the tested amount. 1.5 hours was too little (see the
  gate).
- Scripts that share rows (Bengali for Assamese and Bengali) are calibrated once from all manifests given for that
  script.
- Leftover Devanagari and Arabic rows start silent and depend on training targets alone; how fast Marathi's `ळ` and
  `ॲ` pieces are learned this way has not been measured.
- These steps prepare a checkpoint for training. They do not make it recognise a new language by themselves.
