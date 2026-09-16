"""NeMo restoration using the validated native SentencePiece bundle for both directions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile


class _AcousticVocabularyView:
    """NeMo setup needs every physical row; public vocabulary stays active-only."""

    def __init__(self, adapter):
        self._adapter = adapter

    def get_vocab(self):
        return self._adapter.get_acoustic_vocab()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_adapter"), name)


def native_bundle_config(directory):
    """Validate a bundle before registering its complete file set with NeMo."""
    from .bundles import load_tokenizer_bundle

    directory = Path(directory).resolve()
    load_tokenizer_bundle(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = sorted({"manifest.json", *manifest["files"]})
    if any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError("Native bundle artifacts must have contained filenames")
    return {
        "type": "untok_native_unigram",
        "bundle_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "bundle_filenames": {f"file_{i}": name for i, name in enumerate(names)},
        "bundle_files": {f"file_{i}": str(directory / name) for i, name in enumerate(names)},
    }


def _setup_native_tokenizer(model, tokenizer_cfg):
    from .bundles import load_tokenizer_bundle

    if tokenizer_cfg.get("type") != "untok_native_unigram":
        raise ValueError("NativeNemotronRNNTModel requires a native Unigram bundle")
    filenames = dict(tokenizer_cfg.get("bundle_filenames", {}))
    paths = dict(tokenizer_cfg.get("bundle_files", {}))
    if (not filenames or set(paths) != set(filenames)
            or any(not isinstance(name, str) for name in filenames.values())
            or len(set(filenames.values())) != len(filenames)
            or "manifest.json" not in filenames.values()):
        raise ValueError("Incomplete native bundle artifact configuration")
    for key, name in filenames.items():
        if (not re.fullmatch(r"file_[0-9]+", key) or not isinstance(name, str)
                or Path(name).name != name or name in {".", ".."}):
            raise ValueError("Invalid native bundle artifact name")
    registered = {}
    for key in sorted(paths):
        registered[filenames[key]] = model.register_artifact(f"tokenizer.bundle_files.{key}", paths[key])
    # NeMo renames archived files. Reassemble their logical names only while the
    # bundle verifier loads their bytes; registered originals remain persistent.
    with tempfile.TemporaryDirectory(prefix="untok-native-bundle-") as temporary:
        directory = Path(temporary)
        for name, path in registered.items():
            (directory / name).write_bytes(Path(path).read_bytes())
        digest = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        if digest != tokenizer_cfg.get("bundle_manifest_sha256"):
            raise ValueError("Registered native bundle manifest hash changed")
        tokenizer = load_tokenizer_bundle(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if set(registered) != {"manifest.json", *manifest["files"]}:
            raise ValueError("Registered native artifact set differs from its bundle manifest")
    model.tokenizer_cfg = tokenizer_cfg
    model.tokenizer_dir = str(Path(registered["manifest.json"]).parent)
    # NeMo calls its SentencePiece category "bpe", including Unigram models.
    # The actual encoder and decoder here both use the pinned Unigram proto.
    model.tokenizer_type = "bpe"
    if callable(getattr(tokenizer, "get_acoustic_vocab", None)):
        tokenizer.tokenizer = _AcousticVocabularyView(tokenizer)
    model.tokenizer = tokenizer
    model.native_bundle_manifest_sha256 = digest
    model.native_tokenizer_sha256 = hashlib.sha256(tokenizer.model_bytes).hexdigest()


_NATIVE_NEMO_CLASS = None
_MASKED_OUTPUT_LINEAR_CLASS = None


def _masked_output_linear_class():
    """Load Torch only for acoustic models, keeping text-only installs light."""
    global _MASKED_OUTPUT_LINEAR_CLASS
    if _MASKED_OUTPUT_LINEAR_CLASS is None:
        import torch

        class MaskedNativeOutputLinear(torch.nn.Linear):
            """Mask dormant outputs before any joint/loss softmax, in all modes."""

            def __init__(self, source, inactive_ids):
                # Reuse Parameters rather than initialize or copy a replacement.
                torch.nn.Module.__init__(self)
                self.in_features, self.out_features = source.in_features, source.out_features
                self.register_parameter("weight", source.weight)
                self.register_parameter("bias", source.bias)
                mask = torch.zeros(source.out_features, dtype=torch.bool, device=source.weight.device)
                mask[list(inactive_ids)] = True
                self.register_buffer("inactive_output_mask", mask, persistent=False)
                self.train(source.training)

            def forward(self, inputs):
                logits = torch.nn.functional.linear(inputs, self.weight, self.bias)
                return logits.masked_fill(self.inactive_output_mask, float("-inf"))

        MaskedNativeOutputLinear.__module__ = __name__
        MaskedNativeOutputLinear.__qualname__ = "MaskedNativeOutputLinear"
        _MASKED_OUTPUT_LINEAR_CLASS = MaskedNativeOutputLinear
        globals()["MaskedNativeOutputLinear"] = MaskedNativeOutputLinear
    return _MASKED_OUTPUT_LINEAR_CLASS


def _inactive_output_ids(model):
    inactive = tuple(getattr(model.tokenizer, "inactive_native_ids", ()))
    blank = int(model.tokenizer.blank_id)
    if (any(type(index) is not int or not 0 <= index < blank for index in inactive)
            or len(set(inactive)) != len(inactive)):
        raise ValueError("Inactive native IDs must be unique text rows, never blank")
    return tuple(sorted(inactive))


def install_native_output_mask(model):
    """Install the verified bundle's mask without changing any learned tensor.

    NativeNemotronRNNTModel additionally requires the exact standard RNNTJoint:
    sampled/alternative joint implementations can bypass the final Linear.
    """
    import torch

    inactive = _inactive_output_ids(model)
    if not inactive:
        return verify_native_output_mask(model)
    joint = getattr(model, "joint", None)
    if (getattr(joint, "_num_extra_outputs", 0) != 0 or hasattr(joint, "sampled_joint")
            or not isinstance(getattr(joint, "joint_net", None), torch.nn.Sequential)):
        raise ValueError("Inactive-output masking requires the standard RNNT joint head")
    head = joint.joint_net[-1]
    masked_class = _masked_output_linear_class()
    if not isinstance(head, masked_class):
        if type(head) is not torch.nn.Linear or head.out_features != model.tokenizer.blank_id + 1:
            raise ValueError("Inactive-output masking requires a standard blank-last Linear head")
        joint.joint_net[-1] = masked_class(head, inactive)
    return verify_native_output_mask(model)


def verify_native_output_mask(model):
    """Check the actual output module, its derived mask and forward behavior."""
    import torch

    inactive = _inactive_output_ids(model)
    blank = int(model.tokenizer.blank_id)
    report = {"required": bool(inactive), "inactive_native_ids": list(inactive),
              "inactive_output_rows": len(inactive), "physical_text_rows": blank,
              "active_text_rows": blank - len(inactive), "passed": True,
              "policy": "negative_infinity_before_log_softmax" if inactive else "no_inactive_outputs",
              "scope": "Final RNNT output layer in training and inference; reconstructed from verified tokenizer artifacts"}
    if not inactive:
        return report
    head = model.joint.joint_net[-1]
    if not isinstance(head, _masked_output_linear_class()) or head.out_features != blank + 1:
        raise ValueError("Required native inactive-output mask is missing")
    expected = torch.zeros(blank + 1, dtype=torch.bool, device=head.weight.device)
    expected[list(inactive)] = True
    if head.inactive_output_mask.dtype != torch.bool or not torch.equal(head.inactive_output_mask, expected):
        raise ValueError("Native inactive-output mask differs from the verified tokenizer")
    with torch.no_grad():
        probe = head.weight.new_zeros((1, head.in_features))
        raw = torch.nn.functional.linear(probe, head.weight, head.bias)
        actual = head(probe)
        if (not torch.isneginf(actual[..., expected]).all()
                or not torch.isfinite(actual[..., ~expected]).all()
                or not torch.equal(actual[..., ~expected], raw[..., ~expected])):
            raise ValueError("Native output mask failed to suppress only the inactive logits")
    return report


def transcribe_native_file(model, audio, *, target_lang):
    """Transcribe one file through the verified native tensor/prompt path.

    Keep the model in evaluation mode and return the real NeMo hypotheses and
    observed prompt evidence. The pinned
    NeMo file-list loader can choose a unified prompt, so use tensor audio and
    verify the actual requested one-hot conditioning input instead.
    """
    from .inference import _sha256, _transcribe_with_verified_prompt

    if not getattr(model, "native_tokenizer_sha256", None):
        raise ValueError("Restore a native untok checkpoint before using native inference")
    path = Path(audio)
    model.eval()
    try:
        return _transcribe_with_verified_prompt(model, path, _sha256(path), target_lang)
    finally:
        # NeMo's transcription teardown calls submodule.unfreeze(), which can
        # re-enable training mode after restoring the parent model's mode.
        model.eval()


def _register_native_target(model_class):
    from importlib import import_module

    common = import_module("nemo.core.classes.common")
    original = getattr(common, "_is_target_allowed", None)
    serialization = getattr(common, "Serialization", None)
    if not callable(original) or not isinstance(serialization, type):
        raise RuntimeError("Unsupported NeMo target-validation interface")
    if not isinstance(model_class, type) or not issubclass(model_class, serialization):
        raise ValueError("The native model must be a NeMo Serialization subclass")
    if getattr(original, "_untok_native_registered_class", None) is model_class:
        return
    target_path = "untok.native_runtime.NativeNemotronRNNTModel"

    def allow_native_target(target):
        if target == target_path:
            return common.hydra.utils.get_class(target) is model_class
        return original(target)

    allow_native_target._untok_native_registered_class = model_class
    common._is_target_allowed = allow_native_target


def get_native_nemo_model_class():
    """Register only the exact native model class for save/reload with NeMo."""
    global _NATIVE_NEMO_CLASS
    if _NATIVE_NEMO_CLASS is None:
        from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

        class NativeNemotronRNNTModel(EncDecRNNTBPEModelWithPrompt):
            def __init__(self, cfg, trainer=None):
                # NeMo inspects this named argument before forwarding Trainer
                # through from_config_dict()/restore_from().
                super().__init__(cfg=cfg, trainer=trainer)
                if _inactive_output_ids(self):
                    from nemo.collections.asr.modules.rnnt import RNNTJoint

                    if type(self.joint) is not RNNTJoint:
                        raise ValueError("Inactive native rows require the exact standard NeMo RNNTJoint")
                install_native_output_mask(self)

            def _setup_tokenizer(self, tokenizer_cfg):
                _setup_native_tokenizer(self, tokenizer_cfg)

        NativeNemotronRNNTModel.__module__ = __name__
        NativeNemotronRNNTModel.__qualname__ = "NativeNemotronRNNTModel"
        _NATIVE_NEMO_CLASS = NativeNemotronRNNTModel
        globals()["NativeNemotronRNNTModel"] = _NATIVE_NEMO_CLASS
    _register_native_target(_NATIVE_NEMO_CLASS)
    return _NATIVE_NEMO_CLASS


def __getattr__(name):
    if name == "NativeNemotronRNNTModel":
        return get_native_nemo_model_class()
    if name == "MaskedNativeOutputLinear":
        return _masked_output_linear_class()
    raise AttributeError(name)
