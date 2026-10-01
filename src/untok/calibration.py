"""Required audio calibration of initialised language rows before training them.

For each script with initialised rows, a linear CTC probe is fitted on the frozen checkpoint's joint hidden states under the `auto`
prompt (prediction network at its start state, attention context [56, 1]). Each token's probe weights relative to
the probe's blank become that token's offset; offsets are centred and scaled to the Devanagari rows' spread, so a
token's row points at the sound of that token instead of a random direction. Tokens the probe never saw get the
mean offset and a low bias. Only the initialised output rows and biases change; the sidecar records the receipt.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .checkpoint import _sha256, _torch
from .language_init import (AUTO_SLOT, _outputs, donor_statistics, place_rows, restore_profile_checkpoint,
                            save_with_metadata)
from .text import normalize_text

CALIBRATION_POLICY = "auto_prompt_joint_ctc_probe_v1"


def read_manifest(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def split_rows(rows, *, hours, holdout_speakers, seed, max_duration=20.0, max_dev_clips=400):
    """Speaker-held-out dev clips (if speakers are known), then up to `hours` of shuffled training clips."""
    rng = random.Random(seed)
    rows = [r for r in rows if 0 < r["duration"] <= max_duration and r.get("text", "").strip()]
    speakers = sorted({str(r["speaker_id"]) for r in rows if r.get("speaker_id") is not None})
    if len(speakers) >= 3 * holdout_speakers:
        rng.shuffle(speakers)
        held = set(speakers[:holdout_speakers])
        dev = [r for r in rows if str(r.get("speaker_id")) in held]
        train = [r for r in rows if str(r.get("speaker_id")) not in held]
    else:
        rows = rows[:]
        rng.shuffle(rows)
        cut = max(1, len(rows) // 20)
        dev, train = rows[:cut], rows[cut:]
    rng.shuffle(train)
    rng.shuffle(dev)
    chosen, total = [], 0.0
    for row in train:
        if total >= hours * 3600:
            break
        chosen.append(row)
        total += row["duration"]
    return chosen, dev[:max_dev_clips]


def offsets_from_probe(weight, bias, mean, std, vocab, rows):
    """Blank-relative token offsets in raw feature space from a probe trained on standardised features."""
    torch = _torch()
    weight_eff = weight.double() / std.double()[None]
    dead = std < 1e-2
    weight_eff[:, dead] = 0.0
    bias_eff = bias.double() - weight_eff @ mean.double()
    position = {token: i + 1 for i, token in enumerate(vocab)}  # class 0 is the probe's CTC blank
    seen = torch.tensor([r in position for r in rows])
    if not bool(seen.any()):
        raise ValueError("The probe saw none of this script's tokens")
    offsets = torch.zeros(len(rows), weight.shape[1], dtype=torch.float64)
    bias_offsets = torch.zeros(len(rows), dtype=torch.float64)
    for k, row in enumerate(rows):
        if row in position:
            offsets[k] = weight_eff[position[row]] - weight_eff[0]
            bias_offsets[k] = bias_eff[position[row]] - bias_eff[0]
    offsets[~seen] = offsets[seen].mean(0)
    bias_offsets[~seen] = bias_offsets[seen].min() - 2.0
    return offsets, bias_offsets, {"dead_units": int(dead.sum()), "missing_from_probe": int((~seen).sum())}


def _batches(lengths, max_frames):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, current, longest = [], [], 0
    for i in order:
        if current and max(longest, lengths[i]) * (len(current) + 1) > max_frames:
            batches.append(current)
            current, longest = [], 0
        current.append(i)
        longest = max(longest, lengths[i])
    if current:
        batches.append(current)
    return batches


def _pad(features, index, device):
    torch = _torch()
    longest = max(features[i].shape[0] for i in index)
    out = torch.zeros(len(index), longest, features[index[0]].shape[1], device=device)
    for b, i in enumerate(index):
        out[b, :features[i].shape[0]] = features[i].to(device).float()
    return out, torch.tensor([features[i].shape[0] for i in index], device=device)


def joint_features(model, rows, *, att_context, device, batch=32):
    """Per clip: joint hidden states (frames x joint_hidden) under the auto prompt, decoder at its start state."""
    torch = _torch()
    from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

    model.encoder.set_default_att_context_size(list(att_context))
    with torch.no_grad():
        start, _ = model.decoder.predict(None, None, add_sos=False, batch_size=1)
        prediction = model.joint.pred(start)
        hidden = model.joint.joint_net[:-1]
        out = [None] * len(rows)
        order = sorted(range(len(rows)), key=lambda i: rows[i]["duration"])
        for s in range(0, len(order), batch):
            chunk = order[s:s + batch]
            signals = [torch.as_tensor(AudioSegment.from_file(
                rows[i]["audio_filepath"], target_sr=16000, offset=rows[i].get("offset") or 0,
                duration=rows[i]["duration"] if rows[i].get("offset") else 0).samples, dtype=torch.float32) for i in chunk]
            lengths = torch.tensor([len(x) for x in signals])
            audio = torch.zeros(len(signals), int(lengths.max()))
            for b, x in enumerate(signals):
                audio[b, :len(x)] = x
            processed, processed_len = model.preprocessor(input_signal=audio.to(device), length=lengths.to(device))
            encoded, encoded_len = model.encoder(audio_signal=processed, length=processed_len)
            encoded = encoded.transpose(1, 2)
            prompt = torch.zeros(len(chunk), encoded.shape[1], model.num_prompts, device=device)
            prompt[:, :, AUTO_SLOT] = 1.0
            states = hidden(model.joint.enc(model.prompt_kernel(torch.cat([encoded, prompt], -1))) + prediction)
            for b, i in enumerate(chunk):
                out[i] = states[b, :int(encoded_len[b])].to(torch.bfloat16).cpu()
    return out


def _edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def fit_probe(features, targets, vocab, *, epochs, lr, max_frames, seed, device):
    """Linear CTC head over standardised features. Returns (weight, bias, mean, std, steps)."""
    torch = _torch()
    import torch.nn.functional as F

    torch.manual_seed(seed)
    frames = torch.cat([f.float() for f in features]).to(device)
    mean = frames.double().mean(0)
    std = frames.double().var(0, unbiased=False).sqrt() + 1e-3
    del frames
    mean_f, std_f = mean.float(), std.float()
    position = {token: i + 1 for i, token in enumerate(vocab)}
    labels = [torch.tensor([position[t] for t in target]) for target in targets]
    head = torch.nn.Linear(features[0].shape[1], len(vocab) + 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    batches = _batches([f.shape[0] for f in features], max_frames)
    total = epochs * len(batches)
    warmup = max(1, min(300, total // 10))
    schedule = lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * s / total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(batches)
        for index in batches:
            x, x_len = _pad(features, index, device)
            y = [labels[i] for i in index]
            log_probs = F.log_softmax(head((x - mean_f) / std_f), -1).transpose(0, 1)
            loss = F.ctc_loss(log_probs, torch.cat(y).to(device), x_len, torch.tensor([len(t) for t in y], device=device),
                              blank=0, reduction="mean", zero_infinity=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
    return head.weight.detach().cpu(), head.bias.detach().cpu(), mean.cpu(), std.cpu(), total


def probe_cer(head_weight, head_bias, mean, std, features, rows, vocab, tokenizer):
    """Greedy CTC character error rate of the probe on held-out clips (spaces ignored)."""
    torch = _torch()
    errors = chars = 0
    for f, row in zip(features, rows):
        logits = ((f.float() - mean.float()) / std.float()) @ head_weight.float().T + head_bias.float()
        best = logits.argmax(-1).tolist()
        ids = [vocab[k - 1] for j, k in enumerate(best) if k != 0 and (j == 0 or k != best[j - 1])]
        hyp = tokenizer.ids_to_text(ids).replace(" ", "")
        ref = row["text"].replace(" ", "")
        errors += _edit_distance(hyp, ref)
        chars += len(ref)
    return round(errors / max(1, chars), 4)


def calibrate(source, output, *, source_sha256, manifests, bundle="latin-indic", seed=20260928, hours=12.0,
              epochs=15, att_context=(56, 1), holdout_speakers=60, max_frames=4000, lr=2e-3, device="cuda"):
    """Fit the token offsets of every initialised script in `manifests` ({locale: NeMo JSONL}) from audio."""
    torch = _torch()
    from omegaconf import OmegaConf

    output, tokenizer_dir, sidecar = _outputs(output, ".calibration.json")
    model, adapter, registry, layout = restore_profile_checkpoint(source, source_sha256, bundle)
    raw = model.cfg.get("untok_languages")
    if raw is None:
        raise ValueError("Run `untok init-languages` before `untok calibrate`")
    metadata = OmegaConf.to_container(raw, resolve=True)
    by_script, skipped = {}, {}
    for locale, path in manifests.items():
        info = metadata["languages"].get(locale)
        if info is None:
            raise ValueError(f"{locale} was not initialised with `untok init-languages`")
        if info["rows"] != "initialized":
            skipped[locale] = "the script's rows are trained; nothing to calibrate"
            continue
        by_script.setdefault(info["script"], []).append((locale, Path(path)))
    if not by_script:
        raise ValueError("None of the manifests belongs to a language with initialised rows")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = model.to(device).eval()
    head = model.joint.joint_net[-1]
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    weight, bias = head.weight.detach().cpu().clone(), head.bias.detach().cpu().clone()
    inactive = set(adapter.inactive_native_ids)
    receipts = {}
    for script, items in by_script.items():
        rows = list(metadata["scripts"][script]["rows"])
        clips, sources = [], {}
        for locale, path in items:
            data = read_manifest(path)
            sources[locale] = {"manifest": str(path), "manifest_sha256": _sha256(path), "rows": len(data)}
            for row in data:
                text = normalize_text(row.get("text", ""), locale)[0]
                ids = model.tokenizer.text_to_ids(text)
                if ids and adapter.unk_id not in ids and not inactive.intersection(ids):
                    clips.append({**row, "text": text, "ids": ids})
        train, dev = split_rows(clips, hours=hours, holdout_speakers=holdout_speakers, seed=seed)
        if not train:
            raise ValueError(f"No usable {script} training clips")
        vocab = sorted({t for row in train for t in row["ids"]})
        train_features = joint_features(model, train, att_context=att_context, device=device)
        probe = fit_probe(train_features, [r["ids"] for r in train], vocab, epochs=epochs, lr=lr,
                          max_frames=max_frames, seed=seed, device=device)
        del train_features
        dev_features = joint_features(model, dev, att_context=att_context, device=device)
        cer = probe_cer(*probe[:4], dev_features, dev, vocab, model.tokenizer)
        offsets, bias_offsets, info = offsets_from_probe(*probe[:4], vocab, rows)
        stats = donor_statistics(weight, bias, list(metadata["scripts"][script]["donors"]))
        placed = place_rows(weight, bias, rows, stats, offsets, bias_offsets)
        receipts[script] = {"policy": CALIBRATION_POLICY, "languages_used": sources, "seed": seed,
                            "attention_context": list(att_context), "prompt_slot": AUTO_SLOT,
                            "train_clips": len(train), "train_hours": round(sum(r["duration"] for r in train) / 3600, 3),
                            "dev_clips": len(dev), "probe_classes": len(vocab) + 1, "probe_steps": probe[4],
                            "probe_dev_cer": cer, **info, **placed}
    with torch.no_grad():
        head.weight.copy_(weight.to(head.weight.device))
        head.bias.copy_(bias.to(head.bias.device))
    model = model.cpu()
    changed = sorted({r for script in receipts for r in metadata["scripts"][script]["rows"]})
    after = model.state_dict()
    for key, old in before.items():
        new = after[key].cpu()
        if key in (layout.output_weight_key, layout.output_bias_key):
            keep = torch.ones(old.shape[0], dtype=torch.bool)
            keep[changed] = False
            old, new = old[keep], new[keep]
        if not torch.equal(old, new):
            raise ValueError(f"Calibration changed values outside the calibrated rows: {key}")
    for script, receipt in receipts.items():
        metadata["scripts"][script]["calibration"] = receipt
        metadata["scripts"][script]["init"]["offsets"] = "calibrated"
        metadata["scripts"][script]["init"]["bias"] = "trained level"
        for info in metadata["languages"].values():
            if info["script"] == script and info["rows"] == "initialized":
                info["calibrated"] = True
    save_with_metadata(model, metadata, output, tokenizer_dir, adapter)
    report = {"checkpoint_sha256": _sha256(output), "source_checkpoint_sha256": source_sha256,
              "tokenizer_directory": str(tokenizer_dir), "calibrated_scripts": receipts, "skipped": skipped,
              "languages": metadata["languages"]}
    sidecar.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
