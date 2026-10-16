"""Start Indic languages in an Untok checkpoint: prompt slots and untrained rows.

Works on a fresh `untok migrate` output and on checkpoints already fine-tuned with the same profile.
For every requested language:

- prompt slot: an untrained slot starts as a copy of the trained `auto` column (slot 101); trained slots are kept;
- rows: rows of the language's script that are still identical copies (Untok's start for pieces with no spelling in
  NVIDIA's vocabulary: every Tamil, Telugu, Bengali, ... row, and most added Arabic rows) start at the mean of trained
  rows plus token offsets scaled to those rows' spread, biases at their mean bias, and decoder embedding rows N(0, 1).
  Donors are the script's own trained rows when NVIDIA's vocabulary had the script (Arabic, Devanagari), else the
  Devanagari rows. `init-languages` places every such row exactly at the donor mean with its bias DORMANT_SHIFT below
  the donors' mean, so uncalibrated rows stay silent (random offsets made them fire on Hindi audio); `calibrate`
  (required before training) adds offsets fitted from audio and lifts the biases to the trained rows' level. That
  holds for scripts NVIDIA never wrote. Leftover copies of a script NVIDIA wrote (Devanagari, Arabic) stay dormant,
  are marked `dormant_leftovers` and learn from training targets.
- output tags: latin-indic v6 has `<bn-IN>` ... `<as-IN>` in reused original tag slots, which a fresh migration still
  fills with NVIDIA's rows of the replaced tags (`<bg-BG>`, `<zh-CN>`, ...). The first `init-languages` therefore starts
  every new-language tag in the tokenizer, requested or not, as a copy of the trained `<hi-IN>` output row and decoder
  embedding row with its bias DORMANT_SHIFT lower, so it never ties with `<hi-IN>` (an exact copy would win greedy ties
  from the lower ID). `NewSliceLR` trains the tag rows of the listed languages at the new-row rate, so a recipe with
  tag targets lifts them. Checkpoints made with the untagged v5 profile are still accepted and keep their tokenizer.

Everything else stays bit-identical. Metadata goes to `cfg.untok_languages` and a `.languages.json` sidecar.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

from .checkpoint import _sha256, _torch, inspect_nemo_layout
from .languages import DONOR_SCRIPT, original_size, registry_languages, resolve_languages, script_rows

AUTO_SLOT = 101
TAG_DONOR = "hi-IN"  # new output-language tags start from NVIDIA's trained Hindi tag
INIT_POLICY = "devanagari_mean_scaled_offsets_v1"
PROMPT_KEY = "prompt_kernel.0.weight"
# Untrained prompt columns have norm ~0.57; trained ones ~13.5. Anything above this is treated as trained.
TRAINED_PROMPT_NORM = 2.0
# Uncalibrated rows sit this far below the trained rows' mean bias (about Untok migration's -8.06 new-row shift).
DORMANT_SHIFT = 8.0


def donor_statistics(weight, bias, donors):
    """Mean row, mean distance from it ('spread') and mean bias of the donor rows, in float64."""
    torch = _torch()
    index = torch.tensor(donors)
    rows = weight.index_select(0, index).double()
    mean = rows.mean(0)
    return {"mean": mean, "spread": float((rows - mean).norm(dim=1).mean()),
            "bias_mean": float(bias.index_select(0, index).double().mean())}


def place_rows(weight, bias, rows, stats, offsets, bias_offsets):
    """rows <- donor mean + gamma * centred offsets, gamma matching the donor spread. Edits weight/bias in place."""
    torch = _torch()
    offsets, bias_offsets = offsets.double(), bias_offsets.double()
    centred = offsets - offsets.mean(0, keepdim=True)
    centred_bias = bias_offsets - bias_offsets.mean()
    size = float(centred.norm(dim=1).mean())
    if not size > 0:
        raise ValueError("Offsets must not all be equal")
    gamma = stats["spread"] / size
    index = torch.tensor(rows)
    new_rows = stats["mean"][None] + gamma * centred
    weight.index_copy_(0, index, new_rows.to(weight.dtype))
    bias.index_copy_(0, index, (stats["bias_mean"] + gamma * centred_bias).to(bias.dtype))
    return {"gamma": round(gamma, 6), "row_norm": round(float(new_rows.norm(dim=1).mean()), 3),
            "rows": len(rows)}


def copy_prompt_columns(prompt_weight, enc_hidden, slots, *, force=False):
    """Copy the auto column into untrained slots. Returns slot -> 'auto_copy' or 'kept_trained'."""
    result = {}
    auto = prompt_weight[:, enc_hidden + AUTO_SLOT].clone()
    for slot in slots:
        if slot == AUTO_SLOT:
            raise ValueError("The auto slot cannot be a language slot")
        column = enc_hidden + slot
        if float(prompt_weight[:, column].norm()) > TRAINED_PROMPT_NORM and not force:
            result[slot] = "kept_trained"
            continue
        prompt_weight[:, column] = auto
        result[slot] = "auto_copy"
    return result


def init_language_state(state, layout, enc_hidden, plan, *, seed=0, force_prompt_slots=False):
    """Pure state-dict version of init-languages. Returns (new_state, script_reports, prompt_reports).

    plan = {"slots": [...], "scripts": {script: {"rows": [...], "donors": [...]}},
            "tags": {locale: {"row": id, "donor": id or None}}} ("tags" optional). Input state and global RNG stay
    untouched.
    """
    torch = _torch()
    new = {key: value for key, value in state.items()}
    changed = {layout.output_weight_key, layout.output_bias_key, layout.embedding_key, PROMPT_KEY}
    for key in changed:
        if key not in new or new[key] is None:
            raise ValueError(f"Checkpoint lacks {key}")
        new[key] = new[key].detach().clone()
    prompts = copy_prompt_columns(new[PROMPT_KEY], enc_hidden, plan["slots"], force=force_prompt_slots)
    weight, bias, embedding = new[layout.output_weight_key], new[layout.output_bias_key], new[layout.embedding_key]
    for item in tag_copies(plan):
        weight[item["row"]], embedding[item["row"]] = weight[item["donor"]], embedding[item["donor"]]
        bias[item["row"]] = bias[item["donor"]] - DORMANT_SHIFT
    generator = torch.Generator().manual_seed(seed)
    scripts = {}
    for script, item in plan["scripts"].items():
        rows = item["rows"]
        stats = donor_statistics(weight, bias, item["donors"])
        index = torch.tensor(rows)
        weight.index_copy_(0, index, stats["mean"][None].expand(len(rows), -1).to(weight.dtype))
        bias.index_copy_(0, index, torch.full((len(rows),), stats["bias_mean"] - DORMANT_SHIFT, dtype=bias.dtype))
        vectors = torch.randn(len(rows), embedding.shape[1], generator=generator, dtype=torch.float64)
        embedding.index_copy_(0, index, vectors.to(embedding.dtype))
        scripts[script] = {"rows": len(rows), "offsets": "none until calibrated", "bias": f"dormant (-{DORMANT_SHIFT})"}
    verify_untouched(state, new, layout, plan, enc_hidden)
    return new, scripts, prompts


def verify_untouched(old, new, layout, plan, enc_hidden):
    """Every value outside the initialised rows and prompt columns must be bit-identical."""
    torch = _torch()
    rows = sorted({r for item in plan["scripts"].values() for r in item["rows"]}
                  | {item["row"] for item in tag_copies(plan)})
    columns = [enc_hidden + s for s in plan["slots"]]
    for key, before in old.items():
        after = new[key]
        if key in (layout.output_weight_key, layout.output_bias_key, layout.embedding_key):
            keep = torch.ones(before.shape[0], dtype=torch.bool)
            keep[rows] = False
            before, after = before[keep], after[keep]
        elif key == PROMPT_KEY:
            keep = torch.ones(before.shape[1], dtype=torch.bool)
            keep[columns] = False
            before, after = before[:, keep], after[:, keep]
        if not torch.equal(before, after):
            raise ValueError(f"Values outside the new-language rows changed: {key}")


def tag_copies(plan):
    """Tag rows that start as copies of the donor tag's rows (languages other than the donor's own)."""
    return [item for item in plan.get("tags", {}).values() if item["donor"] is not None]


def tag_plan(model_bytes, locales):
    """locale -> {"row", "donor"} for every language of `locales` whose `<locale>` tag is a piece of the checkpoint's
    tokenizer. The donor is `<hi-IN>`'s row; None for Hindi itself, whose tag NVIDIA trained."""
    import sentencepiece as spm

    backend = spm.SentencePieceProcessor(model_proto=model_bytes)

    def find(locale):
        index = backend.piece_to_id(f"<{locale}>")
        return index if backend.id_to_piece(index) == f"<{locale}>" else None

    donor = find(TAG_DONOR)
    if donor is None:
        return {}
    return {locale: {"row": find(locale), "donor": None if locale == TAG_DONOR else donor}
            for locale in locales if find(locale) is not None}


def identical_copies(weight, rows):
    """Rows whose output vector equals the most frequent vector among them (untrained copies), if repeated."""
    torch = _torch()
    if len(rows) < 2:
        return []
    values, inverse, counts = torch.unique(weight.index_select(0, torch.tensor(rows)), dim=0,
                                           return_inverse=True, return_counts=True)
    top = int(counts.argmax())
    if int(counts[top]) < 2:
        return []
    return [r for r, k in zip(rows, inverse.tolist()) if k == top]


def build_plan(adapter, registry, locales, weight, recorded_scripts=()):
    """Which prompt slots and which untrained rows the requested languages need, and their donor rows.

    Scripts already recorded by an earlier init-languages keep their rows and record (they are no longer copies).
    """
    known = registry_languages(registry)
    cutoff = original_size(adapter)
    languages, scripts = {}, {}
    for locale in locales:
        info = known[locale]
        script = info["script"]
        if script in recorded_scripts:
            scripts[script] = "recorded"
        elif script not in scripts:
            rows = list(script_rows(adapter, script))
            copies = identical_copies(weight, rows)
            if copies:
                own = [r for r in rows if r not in set(copies)]
                if not any(r < cutoff for r in own):  # NVIDIA never wrote this script: borrow Devanagari
                    donor_rows = list(script_rows(adapter, DONOR_SCRIPT))
                    own = [r for r in donor_rows if r not in set(identical_copies(weight, donor_rows))]
                    donor_script = DONOR_SCRIPT
                else:
                    donor_script = script
                scripts[script] = {"rows": copies, "donors": own, "donor_script": donor_script}
            else:
                scripts[script] = None
        languages[locale] = {"prompt_slot": info["prompt_slot"], "script": script,
                             "rows": "initialized" if scripts[script] else "trained"}
    return {"languages": languages, "scripts": {k: v for k, v in scripts.items() if v and v != "recorded"},
            "slots": [info["prompt_slot"] for info in languages.values()]}


def read_language_metadata(nemo_path):
    """cfg.untok_languages from a .nemo file without restoring the model (None if absent)."""
    import yaml

    with tarfile.open(nemo_path) as archive:
        for member in archive:
            if member.name.split("/")[-1] == "model_config.yaml":
                cfg = yaml.safe_load(io.TextIOWrapper(archive.extractfile(member), encoding="utf-8"))
                return cfg.get("untok_languages")
    raise ValueError(f"No model_config.yaml in {nemo_path}")


def require_calibrated(metadata, locales):
    """Refuse training a language whose initialised rows were not calibrated from audio.

    Languages marked `dormant_leftovers` (leftover Devanagari or Arabic copies) pass: they learn from training targets.
    """
    if metadata is None:
        raise ValueError("Checkpoint has no Untok language metadata; run `untok init-languages` and `untok calibrate`")
    missing = [l for l in locales if l not in metadata["languages"]]
    if missing:
        raise ValueError(f"Languages were not initialised with `untok init-languages`: {', '.join(missing)}")
    uncalibrated = [l for l in locales if metadata["languages"][l]["calibrated"] is False]
    if uncalibrated:
        raise ValueError(f"Run `untok calibrate` first; uncalibrated languages: {', '.join(uncalibrated)}")


def _write_tokenizer_dir(directory, model_bytes):
    """The checkpoint's own tokenizer (a superseded profile version stays as it was)."""
    import sentencepiece as spm

    backend = spm.SentencePieceProcessor(model_proto=model_bytes)
    directory.mkdir(parents=True)
    (directory / "tokenizer.model").write_bytes(model_bytes)
    pieces = [backend.id_to_piece(i) for i in range(backend.get_piece_size())]
    (directory / "vocab.txt").write_text("\n".join(pieces) + "\n", encoding="utf-8")


def _outputs(output, sidecar_suffix):
    output = Path(output).resolve()
    tokenizer_dir, sidecar = output.with_suffix(".tokenizer"), output.with_suffix(sidecar_suffix)
    if output.suffix != ".nemo" or any(p.exists() for p in (output, tokenizer_dir, sidecar)):
        raise ValueError("Checkpoint, tokenizer directory and report require new output paths")
    return output, tokenizer_dir, sidecar


def restore_profile_checkpoint(source, expected_sha256, bundle):
    """Restore an Untok-profile checkpoint and check its tokenizer, blank and prompt slots against the bundle."""
    from .bundles import load_tokenizer_bundle
    from .native_checkpoint import _source_native_bytes
    from .nemo import register
    from .prompts import profile_registry
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

    source = Path(source)
    if source.suffix != ".nemo" or _sha256(source) != expected_sha256:
        raise ValueError("Provide the .nemo checkpoint and its matching SHA256")
    adapter = load_tokenizer_bundle(bundle)
    register()
    model = ASRModel.restore_from(str(source), map_location="cpu")
    if type(model) is not EncDecRNNTBPEModelWithPrompt:
        raise ValueError("Checkpoint must be NVIDIA's prompted RNNT model")
    layout = inspect_nemo_layout(model)
    native_sha256 = hashlib.sha256(_source_native_bytes(model)).hexdigest()
    if layout.blank_id != adapter.blank_id or native_sha256 not in adapter.accepted_tokenizer_sha256:
        raise ValueError("Checkpoint tokenizer or blank differs from the bundle")
    registry = profile_registry(adapter.profile)
    if dict(model.cfg.model_defaults.prompt_dictionary) != registry["prompt_dictionary"]:
        raise ValueError("Checkpoint prompt slots differ from the profile's prompt registry")
    return model, adapter, registry, layout


def save_with_metadata(model, metadata, output, tokenizer_dir):
    from omegaconf import open_dict
    from .native_checkpoint import _source_native_bytes

    cfg = model.cfg
    with open_dict(cfg):
        cfg.untok_languages = metadata
    model.cfg = cfg
    model.eval()
    model.save_to(str(output))
    _write_tokenizer_dir(tokenizer_dir, _source_native_bytes(model))


def init_languages(source, output, *, source_sha256, languages="all", bundle="latin-indic", seed=0,
                   force_prompt_slots=False):
    """Prompt slots and untrained rows for every Indic language of the profile (default), or a subset."""
    output, tokenizer_dir, sidecar = _outputs(output, ".languages.json")
    model, adapter, registry, layout = restore_profile_checkpoint(source, source_sha256, bundle)
    from omegaconf import OmegaConf

    previous = model.cfg.get("untok_languages")
    previous = OmegaConf.to_container(previous, resolve=True) if previous is not None else None
    recorded = previous["languages"] if previous else {}
    locales = [l for l in resolve_languages(registry, languages) if l not in recorded]
    if not locales:
        raise ValueError("Every requested language was already initialised in this checkpoint")
    from .native_checkpoint import _source_native_bytes

    plan = build_plan(adapter, registry, locales, model.state_dict()[layout.output_weight_key].detach(),
                      recorded_scripts=set(previous["scripts"]) if previous else ())
    native = _source_native_bytes(model)
    # Tags are started once, all of them: a later run must not reset a tag that training has lifted since.
    recorded_tags = previous.get("tags") if previous else None
    plan["tags"] = {} if recorded_tags is not None else tag_plan(native, list(registry_languages(registry)))
    enc_hidden = int(model.cfg.model_defaults.enc_hidden)
    state, scripts, prompts = init_language_state(model.state_dict(), layout, enc_hidden, plan, seed=seed,
                                                  force_prompt_slots=force_prompt_slots)
    model.load_state_dict(state, strict=True)
    metadata = {"schema_version": 1, "profile": adapter.profile,
                "tokenizer_sha256": hashlib.sha256(native).hexdigest(),
                "init_policy": INIT_POLICY, "source_checkpoint_sha256": source_sha256, "seed": seed,
                "languages": {}, "scripts": {}}
    if previous is not None:  # keep earlier languages of a checkpoint initialised before
        metadata["languages"].update(previous["languages"])
        metadata["scripts"].update(previous["scripts"])
    started = f"copy_of_<{TAG_DONOR}>_dormant"
    metadata["tags"] = recorded_tags if recorded_tags is not None else {
        locale: {"id": item["row"], "init": "kept_trained" if item["donor"] is None else started}
        for locale, item in plan["tags"].items()}
    for script, report in scripts.items():
        item = plan["scripts"][script]
        metadata["scripts"][script] = {"rows": item["rows"], "donors": item["donors"],
                                       "donor_script": item["donor_script"], "init": report, "calibration": None}
    for locale, info in plan["languages"].items():
        calibrated = "not_required"
        if info["rows"] == "initialized":  # a script recorded earlier may already be calibrated
            item = metadata["scripts"][info["script"]]
            if item["donor_script"] == info["script"]:
                # NVIDIA wrote this script (Devanagari, Arabic): its copies are a rare tail (57 Devanagari rows, 45 of
                # them never in Marathi text). Calibrating them from audio blew them up (gamma 7.13, bias +26.9); they
                # stay dormant and learn from training targets with NewSliceLR, as ळ did in Plivo's fine-tune.
                calibrated = "dormant_leftovers"
            else:
                calibrated = item.get("calibration") is not None
        metadata["languages"][locale] = {**info, "prompt_init": prompts[info["prompt_slot"]], "calibrated": calibrated}
    save_with_metadata(model, metadata, output, tokenizer_dir)
    report = {"checkpoint_sha256": _sha256(output), "tokenizer_directory": str(tokenizer_dir),
              "languages": metadata["languages"], "tags": metadata["tags"],
              "scripts": {k: v["init"] for k, v in metadata["scripts"].items()},
              "next_step": "untok calibrate --manifest LOCALE=train.jsonl ... (required for languages with initialised rows)"}
    sidecar.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
