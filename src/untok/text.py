"""Training-text policy for adding Indic languages, and joined two-language clips.

- No language tags (`<hi-IN>`, `<unintelligible>`, ...) in any target.
- Indic text: no punctuation; digits (ASCII or native) spelled out in the row's language with AI4Bharat's
  indic_numtowords (MIT). Thousands separators are dropped first; each run of digits is read as one number, so
  "12.5" becomes "twelve five" in the row's language; runs longer than nine digits are read digit by digit. A
  number the library cannot spell keeps its digits and is counted.
- English keeps its case, punctuation and digits.
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path

from .languages import LANGUAGE_SCRIPT

TAG = re.compile(r"<[^<>]*>")
THOUSANDS = re.compile(r"(?<=\d)[,٬](?=\d)")
DIGITS = re.compile(r"\d+")
NUMBER_LANGUAGES = tuple(LANGUAGE_SCRIPT)  # every Indic language of the profile has number words in the library


def _language(locale):
    return str(locale).split("-")[0].lower()


def spell_numbers(text, language):
    """Digits to words in the language. Returns (text, numbers_spelled, numbers_left)."""
    text = THOUSANDS.sub("", text)
    if not DIGITS.search(text):
        return text, 0, 0
    if language not in NUMBER_LANGUAGES:
        return text, 0, len(DIGITS.findall(text))
    from indic_numtowords import num2words

    spelled = left = 0

    def words(match):
        nonlocal spelled, left
        ascii_digits = "".join(str(unicodedata.decimal(c)) for c in match.group())
        try:  # the library takes the digit string; an int fails beyond nine digits
            result = num2words(ascii_digits, lang=language)
        except Exception:
            left += 1
            return match.group()
        spelled += 1
        return f" {result} "

    return DIGITS.sub(words, text), spelled, left


def normalize_text(text, locale):
    """Target text for one row. Returns (text, numbers_spelled, numbers_left)."""
    text = TAG.sub(" ", text)
    language = _language(locale)
    if language == "en":
        return " ".join(text.split()), 0, 0
    text, spelled, left = spell_numbers(text, language) if language != "auto" else (text, 0, 0)
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split()), spelled, left


def normalize_manifest(source, output, *, lang_field="target_lang"):
    """Apply normalize_text to every row of a NeMo JSONL manifest; rows left empty are dropped."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Output manifest already exists")
    counts = {"rows": 0, "changed": 0, "dropped_empty": 0, "numbers_spelled": 0, "numbers_left": 0, "languages": {}}
    with source.open(encoding="utf-8") as reader, output.open("x", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            locale = row.get(lang_field)
            if locale is None:
                raise ValueError(f"Row without {lang_field}: {row.get('audio_filepath')}")
            text, spelled, left = normalize_text(row.get("text", ""), locale)
            counts["rows"] += 1
            counts["languages"][locale] = counts["languages"].get(locale, 0) + 1
            counts["numbers_spelled"] += spelled
            counts["numbers_left"] += left
            if not text:
                counts["dropped_empty"] += 1
                continue
            counts["changed"] += text != row.get("text")
            writer.write(json.dumps({**row, "text": text}, ensure_ascii=False) + "\n")
    return counts


def join_clips(manifests, output_dir, *, pairs, seed=20260930, max_part=12.0, max_duration=25.0,
               gap=(0.1, 0.5), lang_field="target_lang"):
    """Two clips of different languages joined by a short silence, prompted with auto (code-switched audio).

    Pairs are drawn uniformly over ordered language pairs; each part is at most `max_part` seconds. Writes
    `<output_dir>/audio/*.flac` and `<output_dir>/joined.jsonl` (target_lang auto, prompt_mode auto).
    """
    import numpy as np
    import soundfile as sf
    from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

    output_dir = Path(output_dir)
    manifest = output_dir / "joined.jsonl"
    if manifest.exists():
        raise ValueError("Joined-clip manifest already exists")
    (output_dir / "audio").mkdir(parents=True, exist_ok=True)
    pools = {}
    for path in manifests:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if 0 < row["duration"] <= max_part and row.get("text", "").strip():
                        pools.setdefault(row[lang_field], []).append(row)
    languages = sorted(pools)
    combos = [(a, b) for a in languages for b in languages if a != b]
    if not combos:
        raise ValueError("Joined clips need at least two languages")
    rng, written, total = random.Random(seed), 0, 0.0
    with manifest.open("x", encoding="utf-8") as writer:
        while written < pairs:
            a, b = rng.choice(combos)
            first, second = rng.choice(pools[a]), rng.choice(pools[b])
            silence = rng.uniform(*gap)
            if first["duration"] + silence + second["duration"] > max_duration:
                continue
            parts = [AudioSegment.from_file(r["audio_filepath"], target_sr=16000, offset=r.get("offset") or 0,
                                            duration=r["duration"] if r.get("offset") else 0).samples
                     for r in (first, second)]
            audio = np.concatenate([parts[0], np.zeros(int(silence * 16000), dtype=parts[0].dtype), parts[1]])
            path = output_dir / "audio" / f"{written:07d}.flac"
            sf.write(path, audio, 16000, format="FLAC")
            duration = round(len(audio) / 16000, 3)
            writer.write(json.dumps({"audio_filepath": str(path), "duration": duration,
                                     "text": f"{first['text']} {second['text']}", lang_field: "auto",
                                     "prompt_mode": "auto", "languages": [a, b]}, ensure_ascii=False) + "\n")
            written += 1
            total += duration
    return {"manifest": str(manifest), "pairs": written, "hours": round(total / 3600, 2),
            "languages": {k: len(v) for k, v in pools.items()}}
