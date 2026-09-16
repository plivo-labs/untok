"""Profile scope must distinguish locale semantics from ASCII tag spelling."""
from pathlib import Path
import json

import pytest
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.profile_policy import (
    INDIC_PROMPT_LOCALES, LATIN_PROMPT_LOCALES, LATIN_TOKEN_LOCALES,
    ORIGINAL_TOKEN_LOCALES, allowed_prompt_locales, allowed_token_locales,
    indic_addition_allowed, piece_allowed,
)


DATA = Path(__file__).resolve().parents[1] / "src/untok/data/source"


def piece(text, kind=pb.ModelProto.SentencePiece.NORMAL):
    return pb.ModelProto.SentencePiece(piece=text, score=0, type=kind)


def test_locale_scope_precedes_even_user_defined_piece_type():
    assert not piece_allowed(piece("<bg-BG>", pb.ModelProto.SentencePiece.USER_DEFINED), "latin")
    assert not piece_allowed(piece("<bg-BG>", pb.ModelProto.SentencePiece.USER_DEFINED), "latin-indic")
    assert not piece_allowed(piece("<hi-IN>"), "latin")
    assert piece_allowed(piece("<hi-IN>"), "latin-indic")
    assert not piece_allowed(piece("<ar-AR>"), "latin-indic")
    assert not piece_allowed(piece("<xx-XX>", pb.ModelProto.SentencePiece.USER_DEFINED), "latin-indic")
    for profile in ("original", "latin", "latin-indic", "full"):
        assert piece_allowed(piece("<unk>", pb.ModelProto.SentencePiece.UNKNOWN), profile)
        assert piece_allowed(piece("<s>", pb.ModelProto.SentencePiece.CONTROL), profile)


def test_exact_native_locale_inventory_and_types_are_accounted_for():
    original = pb.ModelProto.FromString((DATA / "base-tokenizer.model").read_bytes())
    tags = [p for p in original.pieces if p.piece.startswith("<") and p.piece.endswith(">") and "-" in p.piece]
    assert {p.piece[1:-1] for p in tags} == ORIGINAL_TOKEN_LOCALES
    assert len(tags) == 39
    assert {p.type for p in tags} == {pb.ModelProto.SentencePiece.NORMAL, pb.ModelProto.SentencePiece.USER_DEFINED}
    for profile, expected in (("latin", 28), ("latin-indic", 29), ("full", 39), ("original", 39)):
        assert sum(piece_allowed(p, profile) for p in tags) == expected
        assert {p.piece[1:-1] for p in tags if piece_allowed(p, profile)} == allowed_token_locales(profile)


def test_script_policy_filters_mixed_pieces_and_preserves_shared_support():
    for text in ("hello", "▁foo", "[]", "\u200d", "॥", "॑"):
        assert piece_allowed(piece(text), "latin")
    for text in ("क", "ب", "ꯀ", "ᱚ", "aक"):
        assert not piece_allowed(piece(text), "latin")
        assert piece_allowed(piece(text), "latin-indic")
    for text in ("я", "α", "漢", "ー", "\u0342", "aя"):
        assert not piece_allowed(piece(text), "latin")
        assert not piece_allowed(piece(text), "latin-indic")


def test_pinned_addition_roles_remove_only_rare_latin_and_keep_shared_indic_support():
    selection = json.loads((DATA / "selection.json").read_text())
    selected = {entry["piece"] for entry in selection["additions"] if indic_addition_allowed(piece(entry["piece"]), entry)}
    rejected = {entry["piece"] for entry in selection["additions"] if entry["piece"] not in selected}
    assert len(selected) == 7273
    assert len(rejected) == 190
    assert {"\u200d", "॑", "॒", "॓", "॔", "॥", "₹", "▁،", "﴾", "﴿", "#"} <= selected
    assert {"ð", "þ", "ċ", "ġ", "ħ", "ḷ"} <= rejected
    assert selected.isdisjoint(rejected)
    with pytest.raises(ValueError, match="does not match"):
        indic_addition_allowed(piece("a"), {"piece": "b", "required_reasons": []})
    with pytest.raises(ValueError, match="required-reason"):
        indic_addition_allowed(piece("a"), {"piece": "a"})


def test_output_tags_and_prompt_identities_are_distinct():
    assert len(LATIN_TOKEN_LOCALES) == 28
    assert len(LATIN_PROMPT_LOCALES) == 29
    assert "sl-SL" in LATIN_TOKEN_LOCALES and "sl-SL" not in LATIN_PROMPT_LOCALES
    assert {"sl-SI", "mt-MT"} <= LATIN_PROMPT_LOCALES
    assert len(INDIC_PROMPT_LOCALES) == 22
    registry = {"auto": 101, "en-US": 0, "en": 0, "en-GB": 1, "enGB": 1,
                "mt-MT": 102, "sl-SI": 62, "sl": 62, "no-NO": 27, "no": 27,
                "hi-IN": 6, "hi": 6, "hi-HI": 6, "ur-PK": 37,
                "or-IN": 59, "or-KE": 59, "ar-AR": 7, "ru-RU": 11}
    latin = allowed_prompt_locales("latin", registry)
    assert {"auto", "en-US", "en", "enGB", "mt-MT", "sl-SI", "sl", "no-NO", "no"} <= latin
    assert {"hi-IN", "hi", "hi-HI", "ur-PK", "or-IN", "or-KE", "ar-AR", "ru-RU"}.isdisjoint(latin)
    indic = allowed_prompt_locales("latin-indic", registry)
    assert {"hi-IN", "hi", "hi-HI", "ur-PK", "or-IN"} <= indic
    assert {"or-KE", "ar-AR", "ru-RU"}.isdisjoint(indic)
    assert allowed_prompt_locales("full", registry) == frozenset(registry)
    assert allowed_prompt_locales("original", registry) == frozenset(registry)
    with pytest.raises(ValueError, match="alias"):
        allowed_prompt_locales("latin", {"en": 10, "en-US": 0})
