# Rebuild the current tokenizers

The maintained builder assembles the four current models from the pinned native
Nemotron base, the ordered 7,273 approved additions and fixed inactive-ID lists.
It preserves the approved spelling, float32 score, piece type and model metadata.
It does not download corpora or repeat the historical candidate search.

From the repository checkout:

```sh
uv run python scripts/build_tokenizers.py --output artifacts/rebuilt
uv run python scripts/verify_tokens.py \
  --data artifacts/rebuilt --baseline-dir src/untok/data
```

The output directory must be new or empty. Verification independently compares
model bytes, retained IDs/scores/types, normalizer, inactive slots and native
blank positions with the approved bundle baseline. It also checks the complete
artifact hashes. To inspect the shipped artifacts directly:

```sh
uv run python scripts/verify_tokens.py
uv run untok check --bundle latin-indic
```

These commands establish tokenizer identity and structure, not acoustic accuracy.
No checkpoint weights or corpus text are downloaded by assembly.

## Selection provenance

The original fitted selection and source locks remain provenance inputs. The
current profiles exclude the historical extra Latin inventory and retain stable
Nemotron IDs. Current assembly uses the approved final selection rather than
re-running script classification or the old chain of profile transformations.

The immutable [research snapshot](https://github.com/plivo-labs/untok/tree/e2f8acd1bab8edf1dc72267678d4a2ba41949439)
preserves corpus acquisition, fitting, pruning, comparisons and historical
reports. See its [original reproduction instructions](https://github.com/plivo-labs/untok/blob/e2f8acd1bab8edf1dc72267678d4a2ba41949439/docs/native-reproduction.md)
for that separate study. It requires pinned external data and source access;
a byte-exact rebuild of the approved tokenizers does not repeat that study.

Keep [source licenses and attribution](../THIRD_PARTY.md) with redistributed
tokenizer materials. The archived research code is no longer an installed
runtime dependency or a maintained command in this checkout.
