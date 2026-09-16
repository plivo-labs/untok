"""Shared native RNNT tensor-layout checks."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Checkpoint operations require the optional torch dependency") from exc
    return torch


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RNNTLayout:
    embedding_key: str
    output_weight_key: str
    output_bias_key: str | None
    blank_id: int
    output_size: int

    @property
    def row_keys(self) -> tuple[str, ...]:
        return tuple(k for k in (self.embedding_key, self.output_weight_key, self.output_bias_key) if k)


def inspect_nemo_layout(model: Any) -> RNNTLayout:
    """Recognize the standard blank-as-padding RNNT layout by live modules.

    Rejects CTC/hybrid/TDT heads, extra outputs, shared parameters and unusual
    embedding arrangements instead of guessing based on tensor dimensions.
    """
    torch = _torch()
    if hasattr(model, "ctc_decoder") or hasattr(model, "aux_ctc"):
        raise ValueError("Hybrid/CTC models are not supported by this RNNT migration")
    decoder, joint = getattr(model, "decoder", None), getattr(model, "joint", None)
    if decoder is None or joint is None or not hasattr(decoder, "prediction"):
        raise ValueError("Unrecognized NeMo RNNT model layout")
    if not getattr(decoder, "blank_as_pad", False):
        raise ValueError("Only RNNT blank_as_pad=True is supported")
    if getattr(joint, "_num_extra_outputs", 0) != 0:
        raise ValueError("Extra output classes (for example duration heads) are unsupported")
    embeddings = [(name, module) for name, module in decoder.named_modules() if isinstance(module, torch.nn.Embedding)]
    if len(embeddings) != 1:
        raise ValueError("Expected exactly one prediction-network embedding")
    _, embedding = embeddings[0]
    joint_net = getattr(joint, "joint_net", None)
    if not isinstance(joint_net, torch.nn.Sequential) or not isinstance(joint_net[-1], torch.nn.Linear):
        raise ValueError("Expected a final Linear layer in joint.joint_net")
    head = joint_net[-1]
    blank = int(decoder.blank_idx)
    if embedding.padding_idx != blank or embedding.num_embeddings != blank + 1:
        raise ValueError("Prediction embedding and blank index do not agree")
    if head.out_features != blank + 1:
        raise ValueError("Joint output rows and blank index do not agree")
    if getattr(joint, "num_classes_with_blank", head.out_features) != head.out_features:
        raise ValueError("Joint class count disagrees with output layer")
    parameters = list(model.named_parameters(remove_duplicate=False))

    def name_for(parameter):
        names = [name for name, value in parameters if value is parameter]
        if len(names) != 1:
            raise ValueError("Shared or unregistered vocabulary parameter is unsupported")
        return names[0]

    return RNNTLayout(name_for(embedding.weight), name_for(head.weight),
                      name_for(head.bias) if head.bias is not None else None,
                      blank, head.out_features)
