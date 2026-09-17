"""Native NeMo training and inference with one optional inactive-output mask."""
import torch
from nemo.collections.asr.modules.rnnt import RNNTJoint


class _MaskedLinear(torch.nn.Linear):
    """Preserve the native parameters and state keys; suppress only unused rows."""

    def __init__(self, source, inactive_ids):
        torch.nn.Module.__init__(self)
        self.in_features, self.out_features = source.in_features, source.out_features
        self.register_parameter("weight", source.weight)
        self.register_parameter("bias", source.bias)
        mask = torch.zeros(source.out_features, dtype=torch.bool, device=source.weight.device)
        mask[inactive_ids] = True
        self.register_buffer("inactive_output_mask", mask, persistent=False)
        self.train(source.training)

    def forward(self, inputs):
        logits = torch.nn.functional.linear(inputs, self.weight, self.bias)
        return logits.masked_fill(self.inactive_output_mask, float("-inf"))


class MaskedRNNTJoint(RNNTJoint):
    """NVIDIA RNNTJoint with reserved SentencePiece slots masked before softmax."""

    def __init__(self, jointnet, num_classes, vocabulary=None, mask_unused_tokens=True, **kwargs):
        super().__init__(jointnet=jointnet, num_classes=num_classes, vocabulary=vocabulary, **kwargs)
        if not mask_unused_tokens:
            return
        if vocabulary is None or len(vocabulary) != num_classes:
            raise ValueError("Masking requires one vocabulary entry per physical text ID")
        inactive = [i for i, piece in enumerate(vocabulary) if piece == f"<unused_nemotron_{i}>"]
        if not inactive:
            return
        head = self.joint_net[-1]
        if type(head) is not torch.nn.Linear or head.out_features != num_classes + 1:
            raise ValueError("Inactive slots require the standard RNNT blank-last Linear head")
        self.joint_net[-1] = _MaskedLinear(head, inactive)


def register():
    """Allow only this component through NeMo's existing target-validation rules."""
    from nemo.core.classes import common

    prefixes = getattr(common, "ALLOWED_TARGET_PREFIXES", None)
    if not isinstance(prefixes, list):
        raise RuntimeError("Unsupported NeMo target registry; use the documented NeMo revision")
    target = "untok.nemo.MaskedRNNTJoint"
    if target not in prefixes:
        prefixes.append(target)


if __name__ == "__main__":
    import runpy
    import sys
    from pathlib import Path
    import nemo

    scripts = {
        "train": "examples/asr/asr_transducer/speech_to_text_rnnt_bpe_prompt.py",
        "evaluate": "examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py",
    }
    if len(sys.argv) < 2 or sys.argv[1] not in scripts:
        raise SystemExit("Usage: python -m untok.nemo {train,evaluate} [native NVIDIA arguments]")
    script = Path(nemo.__file__).resolve().parents[1] / scripts[sys.argv.pop(1)]
    register()
    runpy.run_path(str(script), run_name="__main__")
