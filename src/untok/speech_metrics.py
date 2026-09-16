"""Text error counts and source locales used by native checkpoint probes."""
from __future__ import annotations

from typing import Any, Sequence


BASE_ASR_LOCALES = tuple("en-US en-GB es-US es-ES fr-FR fr-CA it-IT pt-BR pt-PT nl-NL de-DE tr-TR ru-RU ar-AR hi-IN ja-JP ko-KR vi-VN uk-UA pl-PL sv-SE cs-CZ nb-NO da-DK bg-BG fi-FI hr-HR sk-SK zh-CN hu-HU ro-RO et-EE".split())

ADAPTATION_LOCALES = tuple("el-GR lt-LT lv-LV mt-MT sl-SI he-IL th-TH nn-NO".split())


def normalize_for_scoring(text: str) -> str:
    """Collapse whitespace; preserve case, punctuation, Unicode and joiners."""
    return " ".join(text.split())


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance with linear auxiliary storage; repeated tokens count."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]
