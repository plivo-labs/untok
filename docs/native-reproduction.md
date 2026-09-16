# Reproducing native acquisition, preparation and selection

The scripts in `scripts/native_study/` run source acquisition, corpus preparation,
freezing, candidate-pool training, constrained score fitting and joint bank merging.
They operate on explicit input/output directories. They never overwrite the four
packaged tokenizer models. Corpus text stays in the chosen study directory.

There are two distinct reproducibility contracts:

1. **Historical integrity verification.** `native-study-historical.json` publishes
   the original 67 derived-file hashes, counts and model pins without corpus text.
   `native-study-historical-intake.json` publishes the 91 intake hashes and roles.
   These identify the original research data if its snapshots are available.
   The base-plus-selection build reconstructs the full v1 source byte for byte.
   Reconstructing the current packaged v4 profiles then requires the pinned
   [normalization and cleanup recipe](native-preservation.md), via `untok clean`.
2. **A corrected new study.** The portable acquisition and fitting scripts produce
   a new freeze and selection. The written split recipe is explicitly versioned;
   cleanup metadata now survives successive processing. It is not an assertion
   that a fresh download reproduces every historical row or score. Older written
   research splits, mutable HTML/API snapshots, unavailable viewer revisions and
   incomplete historical Punjabi requests prevent that claim. No finished fitted
   model is copied into the fitting process.

The scripts port the original September 2026 study's primary/independent intake,
split and leakage checks, source weighting, candidate training and merge audits.
They use the maintained `untok.unigram_fit` implementation, including its current
numerical checks. Changes to that implementation, runtime versions or input
manifests can change the selected scores; each fit records their hashes.

## Sources and local setup

`configs/native-study-sources.lock.json` includes the 22 BPE donor inventories,
both IndicBART Unigram model files, supporting Unicode/CLDR inputs and the Bhasha
archive. Each downloadable file has an immutable URL and SHA-256. Transcript
repository revisions and the NVIDIA checkpoint/extracted-tokenizer hashes are
also recorded. `native-study-indicvoices.json` pins the shard catalog;
`native-study-spring.json` pins the nine supplemental source repositories.

Install the package and study dependencies in a separate environment:

```sh
uv sync --extra test
uv pip install 'pyarrow>=20,<24' 'huggingface_hub>=0.34,<2'
source .venv/bin/activate

study_dir=/absolute/path/to/new-native-study
mkdir -p "$study_dir"
python scripts/native_study/run.py fetch --cache "$study_dir/cache"
```

Record `python --version` and `uv pip freeze` with the run. Fitting uses the
repository's pinned SentencePiece version. Exact scores across different Python,
NumPy or SentencePiece environments have not been established.

The packaged `src/untok/data/source/base-tokenizer.model` is a verified original
native input. To verify its extraction independently, obtain the pinned NVIDIA
`.nemo` file and run:

```sh
python scripts/native_study/run.py extract-base \
  --checkpoint /absolute/path/to/nemotron-3.5-asr-streaming-0.6b.nemo \
  --output "$study_dir/base-tokenizer.model"
```

This verifies the entire checkpoint hash and reads only an embedded tokenizer
whose hash matches the lock. It does not extract archive paths onto the filesystem.
For the following commands, set `base` to that output or the packaged original:

```sh
base=src/untok/data/source/base-tokenizer.model
```

## Acquire transcripts and written inputs

IndicVoices requires an existing Hugging Face account grant/token. The reader
requests pinned Parquet metadata and transcript column byte ranges; it refuses
servers that ignore ranges. It does not decode or download audio columns. The
`plan` command is local and shows the shard/sampling policy before downloading.

```sh
python scripts/native_study/intake_primary.py plan \
  --output-root "$study_dir" --metadata configs/native-study-indicvoices.json
python scripts/native_study/intake_primary.py ingest \
  --output-root "$study_dir" --metadata configs/native-study-indicvoices.json
python scripts/native_study/intake_primary.py reserve \
  --output-root "$study_dir" --metadata configs/native-study-indicvoices.json

python scripts/native_study/intake_independent.py meta urdu spring \
  --output "$study_dir/independent-intake" \
  --spring-metadata configs/native-study-spring.json

python scripts/native_study/intake_written.py \
  --cache "$study_dir/cache" --output "$study_dir/written" \
  --build-config configs/build.json
```

Primary defaults request 15,000 train and 1,000 official-valid rows per profile,
with the original deterministic shard/rowgroup and speaker/region sampling.
These are convenience samples, not representative dialect coverage. Official
valid is only the reserve intake: freeze still removes cross-role speaker,
recording and text overlaps.

Independent intake preserves the original Meta, Urdu and SPRING sampling rules.
Meta and SPRING's viewer serves the default revision, so fresh acquisition
checks the pinned revision before and after requests and stops on drift. An
authorized cached replay can use `--cached-only`; cached response identities are
verified using hash sidecars, or `--cache-receipt` with the prior source-audit
JSON receipts for historical caches. A fresh run can acquire more Punjabi blocks than the
historical partial run; this correctly produces different input hashes.

Written intake excludes identifiable FLORES rows and the Arabic Sindhi
IndicCorp file from the Devanagari Sindhi profile. It reads the pinned Bhasha
archive and at most 8 MiB per IndicCorp source to retain 2,000 complete nonempty
lines. Each prefix must match the published historical byte count and SHA-256
in the source lock. It strips record terminators only. Its new deterministic 90/10 split uses
source document/record identifiers and does not claim the historical written
split. The later freeze performs normalization only for audits and weighting.

Original external written snapshots, or separately acquired additional samples,
can be included with `--external manifest.json --external-root /path/to/snapshots`.
The manifest has this schema (hash and filename are illustrative):

```json
{"inputs":[{"path":"hi.jsonl","sha256":"64-character-SHA256",
  "language":"hi","domain":"wikimedia","role":"dev"}]}
```

Each JSONL row needs `text` and a document identity in `document_id`,
`revision_url`, `source_url` or `source`. Keep `source`, revision, raw-text and
language metadata where available. Roles may be `dev` or `pending`; written
snapshots never introduce reserve records. Original Wikimedia revision URLs,
UDHR snapshots and Sindhi author prose are optional new-study inputs rather than
silently copied historical files. Omitting them changes the source mixture and
must be reported when comparing new results with the historical candidate.

## Freeze with corrected annotation lineage

```sh
python scripts/native_study/run.py manifest \
  --root "$study_dir" --output "$study_dir/intake.json"
python scripts/native_study/freeze.py \
  --inputs "$study_dir/intake.json" --input-root "$study_dir" \
  --base "$base" --annotation-policy configs/native-study-annotations.json \
  --output "$study_dir/frozen-data"
```

The input contract verifies every intake file before use. Freeze retains the
original source text, removes only listed source annotations from lexical text,
builds speaker/recording/document components, removes cross-role exact and
discovered near duplicates, and calculates per-language/source character weights.
Near-duplicate discovery is approximate, with exact verification of candidates.
All 22 profiles must have nonempty train/dev/reserve files and surviving
IndicVoices training data. An existing freeze directory is never overwritten.

Version 2 preserves previous annotation counts and partial-transcript flags. For
Meta it replays the four documented markers against preserved `raw_text`, so
already-cleaned intake cannot reset the flags to zero. Reselecting raw text does
not double-count the same cleanup. The historical 219 inconsistent rows and the
historical manifest remain immutable. Corrected rows must not be presented as a
byte-identical historical freeze. Partial lexical transcripts are not complete
paired-audio training labels.

Freeze uses relative intake identifiers for stable record IDs across directory
relocation. It prints the resulting manifest hash; use that exact hash below.

## Fit and merge a new selection

```sh
data_sha=THE_PRINTED_FROZEN_MANIFEST_SHA256
python scripts/native_study/run.py fit-all \
  --data "$study_dir/frozen-data" --accept-data-sha256 "$data_sha" \
  --base "$base" --donors "$study_dir/cache/donors" \
  --output "$study_dir/fits"
python scripts/native_study/merge.py --run \
  --data "$study_dir/frozen-data" --accept-data-sha256 "$data_sha" \
  --fits "$study_dir/fits" --base "$base" \
  --output "$study_dir/joint-merge"
```

This is substantial CPU work. `fit-all` runs the twelve jobs sequentially;
individual `fit.py --job ...` invocations allow independent scheduling. Defaults
compare fresh and donor-seeded pools, seeds 0/1 and mass multipliers 0.8/1.2 at
full data. `configs/native-study-recipe.json` supplies these experiment settings
and records the hashes of the original algorithms adapted for this workflow.
Each job selects against the common smallest-mass seed-zero fresh
development reference. Failed settings remain failures in the reports.

Donor files must match the source lock. Only NORMAL strings enter the pool;
donor normalizers and BPE rank scores do not. Native pieces/scores stay fixed.
Per-group budgets, required alphabets and Hindi/Latin lists are checked. The
fitter trains reservoirs, applies forward-backward score estimation and pruning,
and measures development behavior; it does not copy an already-fitted selection.

Merge validates winner receipts and input hashes, deduplicates strings, resolves
shared cross-job scores by maximum provider score, and adds coverage-only pieces
at −32. It freshly checks native prefix/metadata preservation, all-training usage,
required witnesses, literal score dominance and per-language development tails.
Failed gates leave a pending selection. Reserve contents are not used to choose
winners. The optional historical comparison model is absent by default.

After joint gates pass, build and validate a separate output bundle using the
normal `untok build` and `untok validate` commands. Freeze the final selection
receipt before reserved evaluation. A new corpus/selection is a candidate; it
does not inherit the historical tokenizer's validation or acoustic evidence.
This fitting path produces the append-only source format. The shipped v4
cleanup pins the historical full source and rejects a different newly fitted
model. Applying cleanup to a new selection requires an explicit new recipe,
normalizer/input pins and its own evaluation; it must not silently replace the
current release.

## Verify historical data without changing it

```sh
python scripts/native_study/run.py verify \
  --root /path/to/original/frozen-data \
  --manifest configs/native-study-historical.json
python scripts/native_study/run.py verify \
  --root /path/to/staged/original-intake \
  --manifest configs/native-study-historical-intake.json
```

The historical intake root uses `primary-intake/`, `independent-intake/` and
`written/` as listed by the contract. Verification checks identity, not source
sentence correctness. It neither cleans nor writes the corpus. These contracts
allow archived research inputs to be checked independently while the new recipe
remains explicit about changes and unavailable historical acquisition snapshots.
