"""NeMo's SentencePiece API over a verified, unchanged Untok model.

Text-only bundle loading does not import NeMo. Acoustic restoration uses the
installed NVIDIA implementation for tokenization, sampling and boosting APIs.
"""
from __future__ import annotations

from pathlib import Path
from types import MethodType


NEMO_TOKENIZER_INTERFACE_VERSION = 1
_NEMO_TOKENIZER_CLASS = None


def _processor_get_vocab(processor):
    """Return every physical text row at its actual native ID."""
    return {processor.id_to_piece(index): index for index in range(processor.get_piece_size())}


def get_nemo_tokenizer_class():
    """Load NVIDIA's tokenizer only when a NeMo model needs it."""
    global _NEMO_TOKENIZER_CLASS
    if _NEMO_TOKENIZER_CLASS is None:
        from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

        class NativeSentencePieceTokenizer(SentencePieceTokenizer):
            """Native NeMo behavior plus validated Untok row and artifact metadata.

            ``vocab`` and ``get_vocab()`` include inactive reserved rows because
            NeMo indexes the physical acoustic head. ``get_active_vocab()`` is
            explicitly separate. The model's joint mask suppresses inactive
            outputs; this facade never renumbers or edits the SentencePiece model.
            """

            def __init__(self, adapter, model_path):
                if Path(model_path).read_bytes() != adapter.model_bytes:
                    raise ValueError("NeMo tokenizer bytes differ from the verified bundle")
                super().__init__(model_path=str(model_path), legacy=False)
                self._untok_adapter = adapter
                self.backend = self.tokenizer
                if self.tokenizer.serialized_model_proto() != adapter.model_bytes:
                    raise ValueError("NeMo changed the verified SentencePiece model")
                self._attach_processor_metadata()

            def _attach_processor_metadata(self):
                # Native ASR setup adds these convenience attributes to the
                # SentencePiece processor. Use real IDs rather than list offsets.
                self.tokenizer.get_vocab = MethodType(_processor_get_vocab, self.tokenizer)
                self.tokenizer.vocab_size = self.vocab_size
                self.tokenizer.all_special_tokens = self.special_token_to_id

            def __setstate__(self, state):
                self.__dict__.update(state)
                # SentencePiece serializes its proto but drops Python attributes.
                # Reinstall the same native setup helpers in data-loader workers.
                self._attach_processor_metadata()

            def get_vocab(self):
                return self.tokenizer.get_vocab()

            def get_active_vocab(self):
                return self._untok_adapter.get_vocab()

            def __getattr__(self, name):
                # Native methods/properties resolve before bundle metadata. Do
                # not redirect native padding or unknown-token semantics.
                adapter = self.__dict__.get("_untok_adapter")
                if adapter is None:
                    raise AttributeError(name)
                return getattr(adapter, name)

        NativeSentencePieceTokenizer.__module__ = __name__
        NativeSentencePieceTokenizer.__qualname__ = "NativeSentencePieceTokenizer"
        _NEMO_TOKENIZER_CLASS = NativeSentencePieceTokenizer
        globals()["NativeSentencePieceTokenizer"] = NativeSentencePieceTokenizer
    return _NEMO_TOKENIZER_CLASS


def create_nemo_tokenizer(adapter, model_path):
    """Create the native facade while the verified model file is available."""
    return get_nemo_tokenizer_class()(adapter, model_path)


def __getattr__(name):
    if name == "NativeSentencePieceTokenizer":
        return get_nemo_tokenizer_class()
    raise AttributeError(name)
