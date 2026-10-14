"""Indic languages, their scripts and prompt slots in a tokenizer profile (no torch needed)."""
from __future__ import annotations

# Fixed Unicode blocks, so script membership never depends on the Python unicodedata version.
SCRIPT_BLOCKS = {
    "Devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "Bengali": ((0x0980, 0x09FF),),
    "Gurmukhi": ((0x0A00, 0x0A7F),),
    "Gujarati": ((0x0A80, 0x0AFF),),
    "Oriya": ((0x0B00, 0x0B7F),),
    "Tamil": ((0x0B80, 0x0BFF),),
    "Telugu": ((0x0C00, 0x0C7F),),
    "Kannada": ((0x0C80, 0x0CFF),),
    "Malayalam": ((0x0D00, 0x0D7F),),
    "Ol_Chiki": ((0x1C50, 0x1C7F),),
    "Meetei_Mayek": ((0xAAE0, 0xAAFF), (0xABC0, 0xABFF)),
    "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
}

# Writing script of every Indic language in the latin-indic prompt registry.
LANGUAGE_SCRIPT = {
    "hi": "Devanagari", "mr": "Devanagari", "ne": "Devanagari", "mai": "Devanagari", "kok": "Devanagari",
    "sa": "Devanagari", "brx": "Devanagari", "doi": "Devanagari", "sd": "Devanagari",
    "bn": "Bengali", "as": "Bengali", "pa": "Gurmukhi", "gu": "Gujarati", "or": "Oriya", "ta": "Tamil",
    "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "sat": "Ol_Chiki", "mni": "Meetei_Mayek",
    "ur": "Arabic", "ks": "Arabic",
}

# Rows of new scripts start from the mean of this script's trained rows (the closest Brahmic script the model writes).
DONOR_SCRIPT = "Devanagari"


def piece_script(piece):
    """The script of a piece, or None if it has no character from a listed script block."""
    for char in piece:
        code = ord(char)
        for script, blocks in SCRIPT_BLOCKS.items():
            if any(low <= code <= high for low, high in blocks):
                return script
    return None


def script_rows(adapter, script):
    """Active text IDs whose piece contains a character of the script."""
    if script not in SCRIPT_BLOCKS:
        raise ValueError(f"Unknown script: {script}")
    return tuple(i for i in adapter.active_native_ids
                 if piece_script(adapter.backend.id_to_piece(i)) == script)


def original_size(adapter):
    """Text IDs below this existed in NVIDIA's original vocabulary."""
    return len(adapter.source_native_to_target_native) - 1


def new_scripts(adapter):
    """Scripts with active pieces in this profile but none in NVIDIA's original vocabulary."""
    cutoff = original_size(adapter)
    result = []
    for script in SCRIPT_BLOCKS:
        rows = script_rows(adapter, script)
        if rows and not any(i < cutoff for i in rows):
            result.append(script)
    return tuple(result)


def registry_languages(registry):
    """locale -> {language, prompt_slot, script} for the registry's Indic target languages."""
    result = {}
    for entry in registry["target_assignments"]:
        language = entry["language"]
        if language not in LANGUAGE_SCRIPT:
            raise ValueError(f"No script recorded for registry language: {language}")
        result[entry["locale"]] = {"language": language, "prompt_slot": int(entry["prompt_id"]),
                                   "script": LANGUAGE_SCRIPT[language]}
    return result


def resolve_languages(registry, requested):
    """Turn 'all' or codes/locales such as 'ta' or 'ta-IN' into registry locales, in registry order."""
    known = registry_languages(registry)
    if requested in ("all", ["all"], ("all",)):
        return tuple(known)
    by_language = {info["language"]: locale for locale, info in known.items()}
    names = requested.split(",") if isinstance(requested, str) else list(requested)
    result = []
    for name in (n.strip() for n in names):
        locale = name if name in known else by_language.get(name)
        if locale is None:
            raise ValueError(f"Not an Indic language of this profile: {name}")
        if locale not in result:
            result.append(locale)
    if not result:
        raise ValueError("No languages requested")
    return tuple(result)
