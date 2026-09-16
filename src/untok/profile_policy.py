"""Explicit vocabulary and language-prompt scope for the four public profiles.

Locale output pieces and acoustic prompt names are separate inventories. In
particular, Slovenian uses <sl-SL> in the native vocabulary but sl-SI in the
prompt registry; Maltese has a prompt without a dedicated vocabulary tag.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

from sentencepiece import sentencepiece_model_pb2 as pb

from .prompts import TARGET_LOCALES


PROFILES = ("original", "latin", "latin-indic", "full")
LATIN_TOKEN_LOCALES = frozenset("""
cs-CZ da-DK de-DE et-EE fi-FI fr-FR hu-HU hr-HR it-IT lt-LT lv-LV nl-NL
pl-PL pt-BR ro-RO sk-SK sl-SL sv-SE en-GB en-US es-ES es-US fr-CA
nb-NO nn-NO pt-PT tr-TR vi-VN
""".split())
ORIGINAL_TOKEN_LOCALES = LATIN_TOKEN_LOCALES | frozenset("""
bg-BG el-GR ru-RU uk-UA ar-AR he-IL hi-IN ja-JP ko-KR th-TH zh-CN
""".split())
INDIC_PROMPT_LOCALES = frozenset(TARGET_LOCALES.values())
LATIN_PROMPT_LOCALES = (LATIN_TOKEN_LOCALES - {"sl-SL"}) | {"sl-SI", "mt-MT"}
# Norwegian's older no-NO identity has its own native prompt slot; it must not
# be silently reinterpreted as the later, distinct nb-NO or nn-NO slots.
LATIN_LEGACY_PROMPT_LOCALES = frozenset({"no-NO"})
LATIN_PROMPT_ALIASES = {
    "cs": "cs-CZ", "da": "da-DK", "de": "de-DE", "en": "en-US",
    "enGB": "en-GB", "es": "es-US", "esES": "es-ES", "et": "et-EE",
    "fi": "fi-FI", "fr": "fr-FR", "hr": "hr-HR", "hu": "hu-HU",
    "it": "it-IT", "lt": "lt-LT", "lv": "lv-LV", "nb": "nb-NO",
    "nl": "nl-NL", "nn": "nn-NO", "no": "no-NO", "pl": "pl-PL",
    "pt": "pt-PT", "ro": "ro-RO", "sk": "sk-SK", "sl": "sl-SI",
    "sv": "sv-SE", "tr": "tr-TR",
}
INDIC_PROMPT_ALIASES = {"hi": "hi-IN", "hi-HI": "hi-IN"}
_LOCALE_TAG = re.compile(r"<([a-z]{2,3}(?:-[A-Za-z0-9]+)+)>")


def _check_profile(profile: str) -> None:
    if profile not in PROFILES:
        raise ValueError(f"Unknown tokenizer profile: {profile}")


def allowed_token_locales(profile: str) -> frozenset[str]:
    _check_profile(profile)
    if profile in {"original", "full"}:
        return ORIGINAL_TOKEN_LOCALES
    if profile == "latin":
        return LATIN_TOKEN_LOCALES
    # Only existing output tags may be selected. Prompt identities do not
    # synthesize the twenty-one absent Indic output tokens.
    return LATIN_TOKEN_LOCALES | (ORIGINAL_TOKEN_LOCALES & INDIC_PROMPT_LOCALES)


def profile_policy(profile: str) -> dict:
    """Serializable selection contract; it never changes the legacy v1 policy."""
    _check_profile(profile)
    scopes = {
        "original": "Exact original Nemotron tokenizer bytes; no additions or filtering",
        "latin": "Original Nemotron Latin/shared pieces and relevant source specials only; no additions",
        "latin-indic": "Original Nemotron Latin/Indic/shared pieces plus approved Indic/shared additions; relevant source specials",
        "full": "Complete original Nemotron bank at unchanged IDs plus approved Indic/shared additions only",
    }
    return {
        "schema_version": 1,
        "profile_version": 4,
        "profile": profile,
        "scope": scopes[profile],
        "special_tokens": "Filter existing locale tags by explicit language scope before protobuf type; retain technical specials; invent no tags",
        "allowed_locale_tags": [f"<{locale}>" for locale in sorted(allowed_token_locales(profile))],
        "locale_filter_before_piece_type": profile in {"latin", "latin-indic"},
        "new_output_language_tags": False,
        "prompt_registry": "Separate from output-token inventory; preserve selected source prompt slots",
        "normalizer": "Original Nemotron normalization preserved exactly",
        "latin_extension_additions": False,
        "source_addition_policy": "Use hash-pinned source-selection entries; exclude approved_rare_latin_character provenance and all later Latin coverage extensions",
        "indic_shared_additions": profile in {"latin-indic", "full"},
        "compact_ids": profile in {"latin", "latin-indic"},
    }


def piece_allowed(piece: pb.ModelProto.SentencePiece, profile: str) -> bool:
    """Filter locale tokens before protobuf type or ordinary script checks."""
    _check_profile(profile)
    if profile in {"original", "full"}:
        return True
    locale = _LOCALE_TAG.fullmatch(piece.piece)
    if locale:
        return locale.group(1) in allowed_token_locales(profile)
    if piece.type != pb.ModelProto.SentencePiece.NORMAL:
        return True
    from .bundles import character_allowed

    return all(character_allowed(character, profile) for character in piece.piece)


def indic_addition_allowed(piece: pb.ModelProto.SentencePiece, source_addition: Mapping) -> bool:
    """Select Indic/shared source additions using their pinned provenance.

    The caller must verify the source-selection integrity before invoking this
    pure helper. This rejects the explicitly approved rare-Latin inventory,
    while retaining shared punctuation, joiners and Indic marks even where
    Unicode also admits them to the Latin character policy. New case-closure
    coverage is a separate inventory and must not be passed as source data.
    """
    if not isinstance(source_addition, Mapping) or source_addition.get("piece") != piece.piece:
        raise ValueError("Source addition provenance does not match its piece")
    reasons = source_addition.get("required_reasons")
    if not isinstance(reasons, list) or any(not isinstance(reason, Mapping) for reason in reasons):
        raise ValueError("Source addition must include its required-reason provenance")
    if any(reason.get("kind") == "approved_rare_latin_character" for reason in reasons):
        return False
    return piece.type == pb.ModelProto.SentencePiece.NORMAL and piece_allowed(piece, "latin-indic")


def allowed_prompt_locales(profile: str, source_registry: Mapping[str, int]) -> frozenset[str]:
    """Choose existing prompt names, preserving canonical and explicit aliases.

    Returns names only; callers retain the source numeric slots. Alias slots
    must equal their explicitly declared identity. Sharing a numeric slot with
    an admitted locale never makes an unrelated identity admissible.
    """
    _check_profile(profile)
    if profile in {"original", "full"}:
        return frozenset(source_registry)
    identities = LATIN_PROMPT_LOCALES | LATIN_LEGACY_PROMPT_LOCALES | {"auto"}
    aliases = dict(LATIN_PROMPT_ALIASES)
    if profile == "latin-indic":
        identities |= INDIC_PROMPT_LOCALES
        aliases.update(INDIC_PROMPT_ALIASES)
    selected = set(source_registry) & identities
    for alias, identity in aliases.items():
        if alias not in source_registry:
            continue
        if identity not in source_registry or source_registry[alias] != source_registry[identity]:
            raise ValueError(f"Source prompt alias {alias!r} disagrees with {identity!r}")
        selected.add(alias)
    return frozenset(selected)
