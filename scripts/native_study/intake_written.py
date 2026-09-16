"""Acquire bounded written inputs and create a new deterministic train/dev split.

This is a new corpus recipe, not a byte replay of the historical research split.
No Unicode normalization or changes to source spellings are applied here.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

from untok.sources import sha256

REVISION = "2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f"
FILES = dict(zip(
    "as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat ta te ur".split(),
    "as bn bd dg gu hi-1 kn gom ks mai ml mni mr ne or pa sa sat ta te ur".split()))
SEED = "untok-portable-written-v2"
SOURCE_LOCK = Path(__file__).resolve().parents[2] / "configs/native-study-sources.lock.json"


def role_for(document):
    return "dev" if int(hashlib.sha256((SEED + "\0" + document).encode()).hexdigest()[:16], 16) % 10 == 0 else "pending"


def prepare(cache, output, config, external=None, external_root=None, cached_only=False, source_lock=SOURCE_LOCK):
    if output.exists():
        raise FileExistsError("Written intake is immutable: " + str(output))
    targets = {t["language"]: t for t in json.loads(config.read_text())["targets"]}
    rows = defaultdict(list)
    sources = []
    prefixes = {item['language']: item for item in json.loads(source_lock.read_text())['written_prefixes']}
    aliases = {"Perso-Arabic": "Arabic", "Meetei-Mayek": "Meetei Mayek", "Ol-Chiki": "Ol Chiki", "Oriya": "Odia"}
    archive = cache / "bhasha.zip"
    if sha256(archive) != "be4bd82c5b9b54528393bcbf4542a4f7f2ac9ee4fc1a3609f0ed685a1d252c19":
        raise ValueError("Bhasha archive differs from pinned v1.0")
    with zipfile.ZipFile(archive) as z:
        entries = json.loads(z.read("bhasha-abhijnaanam.json"))["data"]
    for n, row in enumerate(entries):
        lang = row["unique_identifier"].split("_", 1)[0]
        lang = {"dg": "doi", "gom": "kok"}.get(lang, lang)
        text = row.get("native sentence")
        if lang not in targets or aliases.get(row["script"], row["script"]) != targets[lang]["script"]:
            continue
        if not isinstance(text, str) or not text.strip() or "flores" in str(row.get("source", "")).lower():
            continue
        document = "bhasha:" + lang + ":" + str(row.get("source")) + ":" + row["unique_identifier"]
        rows[lang].append(dict(text=text, raw_text=text, language=lang, id=row["unique_identifier"],
                               source=row.get("source"), domain="bhasha", script=targets[lang]["script"],
                               source_group=document, source_row=n, source_revision="IndicLID/v1.0", role=role_for(document)))
    sources.append(dict(path="bhasha.zip", sha256=sha256(archive)))
    for lang, filename in FILES.items():
        url = f"https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/{REVISION}/data/{filename}.txt"
        path = cache / f"indiccorp-{lang}-prefix.bin"
        if not path.exists():
            if cached_only:
                raise FileNotFoundError(path)
            request = urllib.request.Request(url, headers={"Range": "bytes=0-8388607"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(8_388_608)
            path.write_bytes(data)
        data = path.read_bytes()
        expected = prefixes[lang]
        if (expected['url'] != url or len(data) != expected['bytes'] or
                hashlib.sha256(data).hexdigest() != expected['sha256']):
            raise ValueError("IndicCorp prefix differs from pinned acquisition contract: " + lang)
        count = 0
        for line_no, line in enumerate(data.split(b"\n")[:-1], 1):
            text = line.removesuffix(b"\r").decode("utf-8")
            if not text.strip():
                continue
            document = f"indiccorp:{lang}:line:{line_no}"
            rows[lang].append(dict(text=text, raw_text=text, language=lang, id=document, domain="indiccorp",
                                   source="ai4bharat/IndicCorpV2", source_url=url, source_revision=REVISION,
                                   source_group=document, source_line=line_no, role=role_for(document)))
            count += 1
            if count == 2000:
                break
        if count < 2000:
            raise ValueError(f"Insufficient complete IndicCorp records for {lang}: {count}")
        sources.append(dict(path=path.name, sha256=sha256(path), url=url, selected_rows=count))
    # External written snapshots are explicit, hashed inputs. Retain document
    # identities so freeze can protect groups; never assign roles by paragraph.
    if external is not None:
        for item in json.loads(external.read_text())["inputs"]:
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("External snapshot path must be relative")
            path = external_root.resolve() / relative
            if not path.resolve().is_relative_to(external_root.resolve()) or sha256(path) != item["sha256"]:
                raise ValueError("External snapshot identity mismatch")
            for line in path.read_text().splitlines():
                row = json.loads(line)
                lang = item["language"]
                document = row.get("document_id") or row.get("revision_url") or row.get("source_url") or row.get("source")
                if not document or not isinstance(row.get("text"), str):
                    raise ValueError("External written rows need text and a source document identity")
                role = item.get("role", "dev")
                if role not in ("dev", "pending"):
                    raise ValueError("External written inputs cannot supply reserve")
                rows[lang].append(dict(row, language=lang, domain=item["domain"], source_group=str(document), role=role))
            sources.append(dict(item))
    output.mkdir(parents=True)
    files = []
    for lang in targets:
        for role, suffix in [("pending", "train"), ("dev", "dev")]:
            path = output / f"{lang}-{suffix}.jsonl"
            with path.open("w") as f:
                for row in rows[lang]:
                    if row["role"] == role:
                        f.write(json.dumps({k: v for k, v in row.items() if k != "role"}, ensure_ascii=False, sort_keys=True) + "\n")
            files.append(dict(path=path.name, sha256=sha256(path), kind="written", role=role))
    receipt = dict(schema_version=2, recipe=SEED, historical_byte_replay=False, sources=sources, inputs=files,
                   limits="Convenience samples; document identities in Bhasha are record proxies. Native freeze handles exact/near duplicates and final roles.")
    (output / "intake.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--build-config", type=Path, required=True)
    p.add_argument("--external", type=Path)
    p.add_argument("--external-root", type=Path)
    p.add_argument("--cached-only", action="store_true")
    p.add_argument("--source-lock", type=Path, default=SOURCE_LOCK)
    args = p.parse_args()
    if args.external and not args.external_root:
        p.error("--external requires --external-root")
    result = prepare(args.cache, args.output, args.build_config, args.external, args.external_root, args.cached_only, args.source_lock)
    print(json.dumps({"files": len(result["inputs"]), "historical_byte_replay": False}))


if __name__ == "__main__":
    main()
