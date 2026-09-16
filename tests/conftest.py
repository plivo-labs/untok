import pytest


@pytest.fixture
def native_sentencepiece():
    """The acoustic interface is tested against the installed NVIDIA class."""
    module = pytest.importorskip("nemo.collections.common.tokenizers.sentencepiece_tokenizer")
    return module.SentencePieceTokenizer
