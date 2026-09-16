#!/usr/bin/env python3
"""Fetch a small real FLEURS corpus, stopping each archive stream early.

No model, predictions, transcription cleanup, or training is involved. This is
an intentionally short development corpus, never a release accuracy benchmark.
The immutable source revision and archive's published LFS hash are recorded;
because only a prefix is fetched, the full archive hash is NOT verified.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import urllib.request

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
DATASET = "google/fleurs"
PROFILES = {
    "en_us": ("en", "English", "Latin", "en-US"),
    "hi_in": ("hi", "Hindi", "Devanagari", "hi-IN"),
    "ta_in": ("ta", "Tamil", "Tamil", "ta-IN"),
    "ml_in": ("ml", "Malayalam", "Malayalam", "ml-IN"),
    "mr_in": ("mr", "Marathi", "Devanagari", "mr-IN"),
    "kn_in": ("kn", "Kannada", "Kannada", "kn-IN"),
    "bn_in": ("bn", "Bengali", "Bengali", "bn-IN"),
    "te_in": ("te", "Telugu", "Telugu", "te-IN"),
    "gu_in": ("gu", "Gujarati", "Gujarati", "gu-IN"),
    "or_in": ("or", "Odia", "Odia", "or-IN"),
    "pa_in": ("pa", "Punjabi", "Gurmukhi", "pa-IN"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "untok-development-smoke/1"})
    return urllib.request.urlopen(request, timeout=120)


def resolved(path: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{path}"


def get_json(url: str):
    with get(url) as response:
        return json.load(response)


class BudgetReader:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.downloaded = stream, limit, 0

    def read(self, size=-1):
        if size < 0:
            size = 10240
        if self.downloaded + size > self.limit:
            raise RuntimeError("Archive prefix download cap reached; increase the explicit cap")
        data = self.stream.read(size)
        self.downloaded += len(data)
        return data


def wav_info(data: bytes) -> dict:
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Expected an original RIFF/WAVE source file")
    fmt = None
    audio_size = None
    position = 12
    while position + 8 <= len(data):
        key, size = struct.unpack_from("<4sI", data, position)
        begin = position + 8
        if begin + size > len(data):
            raise ValueError("Truncated WAV chunk")
        if key == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", data, begin)
        elif key == b"data":
            audio_size = size
        position = begin + size + (size % 2)
    if fmt is None or audio_size is None:
        raise ValueError("WAV has no fmt/data chunk")
    encoding, channels, sample_rate, byte_rate, block_align, bits = fmt
    if sample_rate != 16000 or channels != 1 or block_align <= 0:
        raise ValueError("Expected the documented original mono 16 kHz FLEURS audio")
    return {"sample_rate": sample_rate, "channels": channels, "encoding": encoding,
            "bits_per_sample": bits, "num_samples": audio_size // block_align,
            "duration": audio_size / block_align / sample_rate}


def prepare_split(config, split, count, args, registry):
    language, name, script, locale = PROFILES[config]
    tsv_url = resolved(f"data/{config}/{split}.tsv")
    with get(tsv_url) as response:
        tsv_bytes = response.read()
    # FLEURS stores literal quotation marks in transcripts, not CSV quoting.
    tsv_rows = list(csv.reader(io.StringIO(tsv_bytes.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE))
    rows = {}
    for row in tsv_rows:
        if len(row) != 7:
            raise ValueError(f"Unexpected FLEURS TSV schema: {config}/{split}")
        record_id, filename, raw_text, text, _, num_samples, gender = row
        if Path(filename).name != filename:
            raise ValueError("Audio filename must be a basename")
        if filename in rows:
            raise ValueError("Duplicate audio filename in source metadata")
        rows[filename] = {"source_id": record_id, "text": text, "raw_transcription": raw_text,
                          "num_samples": int(num_samples), "gender": gender}
    listing_url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/data/{config}/audio"
    listing = get_json(listing_url)
    archive_path = f"data/{config}/audio/{split}.tar.gz"
    entry = next(item for item in listing if item["path"] == archive_path)
    archive_url = resolved(archive_path)
    selected = []
    skipped = {"duration": 0, "empty_text": 0}
    with get(archive_url) as response:
        prefix = BudgetReader(response, args.max_download_mb_per_archive * 1024 * 1024)
        with tarfile.open(fileobj=prefix, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                filename = Path(member.name).name
                source = rows.get(filename)
                if source is None:
                    raise ValueError(f"Audio absent from pinned TSV: {member.name}")
                expected_duration = source["num_samples"] / 16000
                if not args.min_duration <= expected_duration <= args.max_duration:
                    skipped["duration"] += 1
                    continue
                if not source["text"].strip():
                    skipped["empty_text"] += 1
                    continue
                if member.size > 10 * 1024 * 1024:
                    raise ValueError("Selected short clip has unexpected file size")
                stream = archive.extractfile(member)
                data = stream.read()
                info = wav_info(data)
                if info["num_samples"] != source["num_samples"]:
                    raise ValueError("Audio length disagrees with pinned transcript metadata")
                clip_id = f"fleurs-{config}-{split}-{source['source_id']}-{Path(filename).stem}"
                relative = Path("audio") / config / split / filename
                path = args.output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.read_bytes() != data:
                    raise ValueError("Existing clip bytes differ; refusing overwrite")
                path.write_bytes(data)
                result = {"id": clip_id, "audio": str(relative), "audio_filepath": str(path.resolve()),
                    "audio_sha256": sha(data), "target_lang": locale,
                    "paired_migration_prompt_available": locale in registry,
                    "language": language, "language_name": name, "script": script,
                    **source, **info, "source_dataset": DATASET, "source_revision": REVISION,
                    "source_split": split, "source_audio_filename": filename,
                    "source_tsv_url": tsv_url, "source_tsv_sha256": sha(tsv_bytes),
                    "source_archive_url": archive_url,
                    "source_archive_published_sha256": entry.get("lfs", {}).get("oid"),
                    "source_archive_full_sha256_verified": False,
                    "license": "CC-BY-4.0", "speaker_id": None,
                    "purpose": "training_smoke" if split == "train" else "migration_development_check"}
                selected.append(result)
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise ValueError(f"Only found {len(selected)}/{count} eligible clips for {config}/{split}")
    provenance = {"config": config, "split": split, "selected": len(selected),
        "archive_prefix_bytes_downloaded": prefix.downloaded, "source_archive_bytes": entry["size"],
        "source_archive_published_sha256": entry.get("lfs", {}).get("oid"),
        "full_archive_hash_verified": False, "skipped_before_selection_complete": skipped,
        "selection": "first archive members with nonempty source text and declared duration interval"}
    return selected, provenance
