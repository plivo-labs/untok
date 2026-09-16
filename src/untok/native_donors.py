"""Literal text-donor initialization for a verified native tokenizer and row map."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math

from sentencepiece import sentencepiece_model_pb2 as pb

from .checkpoint import _torch


class ExactDonors:
    """Maximize original Unigram score, then prefer the smallest source-ID path."""

    def __init__(self, base, source_to_target):
        self.base, self.trie = base, {}
        for index, piece in enumerate(base.pieces):
            if piece.type != pb.ModelProto.SentencePiece.NORMAL or source_to_target[index] is None:
                continue
            if not piece.piece or not math.isfinite(piece.score):
                raise ValueError("Donor pieces must have nonempty spellings and finite scores")
            node = self.trie
            for character in piece.piece:
                node = node.setdefault(character, {})
            if None in node:
                raise ValueError("Duplicate donor spelling")
            node[None] = index

    def segment(self, spelling):
        """Use literal internal spelling: no normalization or inserted boundary."""
        if not spelling:
            return None
        best = [None] * (len(spelling) + 1)
        best[0] = (0.0, ())
        for begin in range(len(spelling)):
            if best[begin] is None:
                continue
            score, prefix = best[begin]
            node = self.trie
            for end in range(begin, len(spelling)):
                node = node.get(spelling[end])
                if node is None:
                    break
                index = node.get(None)
                if index is None:
                    continue
                candidate = (score + self.base.pieces[index].score, prefix + (index,))
                previous = best[end + 1]
                if (previous is None or candidate[0] > previous[0]
                        or candidate[0] == previous[0] and candidate[1] < previous[1]):
                    best[end + 1] = candidate
        return best[-1]


def donor_paths(adapter, added, retained):
    """Build convex coefficients from verified original NORMAL pieces only."""
    base = pb.ModelProto.FromString(adapter.base_model_bytes)
    target = pb.ModelProto.FromString(adapter.model_bytes)
    inactive = set(getattr(adapter, "inactive_native_ids", ()))
    mapping = tuple(None if i in inactive or i not in retained else i
                    for i in adapter.source_native_to_target_native)
    normal = tuple(sorted(i for i in mapping[:-1] if i is not None
                          and target.pieces[i].type == pb.ModelProto.SentencePiece.NORMAL))
    if not normal:
        raise ValueError("Text-donor initialization requires active original NORMAL pieces")
    finder = ExactDonors(base, mapping)
    paths, loads, fallback_count = {}, dict.fromkeys(normal, Fraction(0)), 0
    for index in added:
        piece = target.pieces[index]
        if piece.type != pb.ModelProto.SentencePiece.NORMAL or not piece.piece:
            raise ValueError("Added pieces must be nonempty NORMAL pieces")
        found = finder.segment(piece.piece)
        path = tuple(mapping[i] for i in found[1]) if found else ()
        paths[index] = path
        if path:
            for donor, count in Counter(path).items():
                loads[donor] += Fraction(count, len(path))
        else:
            fallback_count += 1
    for index in normal:
        loads[index] += Fraction(fallback_count, len(normal))
    if sum(loads.values()) != len(added):
        raise ValueError("Convex donor loading is inconsistent")
    return normal, paths, max(loads.values())


def initialize_text_donor_rows(state, layout, adapter, source_to_target, *, max_new_mass_ratio=0.05):
    """Initialize only additions after native retained-state transfer.

    Identical convex coefficients in W and b give an initial exact-arithmetic
    bound Z_added/Z_active_retained <= exp(shift)*Cmax. FP32 storage and inference
    add rounding error. Embedding means do not reproduce donor recurrent states.
    The caller owns validated artifacts, layouts, retention and reload checks.
    """
    torch = _torch()
    if not math.isfinite(max_new_mass_ratio) or not 0 < max_new_mass_ratio < 1:
        raise ValueError("New-output mass ratio must be finite and between zero and one")
    retained = set(i for i in source_to_target if i is not None)
    added = sorted(set(range(layout.blank_id)) - retained)
    report = {"policy": "no_added_rows", "new_row_count": len(added), "new_model_rows": added}
    if not added:
        return dict(state), report
    if layout.output_bias_key is None:
        raise ValueError("Bounded added-output initialization requires a joint output bias")
    normal, paths, cmax = donor_paths(adapter, added, retained)
    shift = math.log(max_new_mass_ratio / float(cmax))
    result = dict(state)
    for key, ndim in zip(layout.row_keys, (2, 2, 1)):
        value = state[key]
        if (value.device.type != "cpu" or value.dtype != torch.float32 or value.ndim != ndim
                or value.shape[0] != layout.output_size or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Expected finite CPU FP32 vocabulary parameter: {key}")
        source = value.detach().double()
        fallback = source[list(normal)].mean(0)
        means = torch.stack([source[list(paths[i])].mean(0) if paths[i] else fallback for i in added])
        stored = means.float()
        if key == layout.output_bias_key:
            stored = stored + shift
        if not bool(torch.isfinite(stored).all()):
            raise ValueError("Nonfinite initialized rows")
        result[key] = value.detach().clone()
        result[key][added] = stored
    report.update(policy="retained_text_donor_mean_bound_v1", bias_shift=shift,
                  max_new_mass_ratio=max_new_mass_ratio, cmax=float(cmax), cmax_fraction=str(cmax),
                  retained_normal_count=len(normal), exact_decomposition_count=sum(bool(p) for p in paths.values()),
                  fallback_count=sum(not p for p in paths.values()),
                  bound="Z_added/Z_active_retained <= exp(bias_shift)*Cmax in exact arithmetic; inactive slots excluded",
                  limitation="Initial bound only; FP32 rounding and inference require validation; no acoustic or convergence guarantee")
    return result, report
