"""Public tokenizer IDs and native blank-last RNNT ID mapping."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class IdMap:
    """A complete, reversible mapping except for non-acoustic public padding."""

    canonical_to_model: tuple[int | None, ...]
    model_to_canonical: tuple[int, ...]
    hf_pad_id: int
    hf_blank_id: int
    model_blank_id: int
    tokenizer_sha256: str
    base_tokenizer_sha256: str | None = None

    @property
    def model_vocab_size(self) -> int:
        """Non-blank vocabulary size, as expected by NeMo RNNT modules."""
        return self.model_blank_id

    @property
    def model_output_size(self) -> int:
        return len(self.model_to_canonical)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "layout": "nemo_dense_blank_last", **asdict(self)}

    def to_model(self, ids: Sequence[int], *, allow_blank: bool = False) -> list[int]:
        result = []
        for index in ids:
            index = int(index)
            if not 0 <= index < len(self.canonical_to_model):
                raise ValueError(f"Canonical token ID out of range: {index}")
            mapped = self.canonical_to_model[index]
            if mapped is None:
                raise ValueError("Public padding is not an acoustic token or training label")
            if mapped == self.model_blank_id and not allow_blank:
                raise ValueError("RNNT blank is not a transcript label")
            result.append(mapped)
        return result

    def to_canonical(self, ids: Sequence[int], *, drop_blank: bool = True) -> list[int]:
        result = []
        for index in ids:
            index = int(index)
            if not 0 <= index < len(self.model_to_canonical):
                raise ValueError(f"Model token ID out of range: {index}")
            if drop_blank and index == self.model_blank_id:
                continue
            result.append(self.model_to_canonical[index])
        return result
