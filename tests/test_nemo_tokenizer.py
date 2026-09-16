"""Differential tests against NVIDIA's actual SentencePiece implementation."""
import copy
import pickle
import subprocess
import sys

import pytest

from untok.bundles import load_tokenizer_bundle
from untok.nemo_tokenizer import create_nemo_tokenizer


@pytest.mark.parametrize("profile", ["original", "latin", "latin-indic", "full"])
def test_native_facade_preserves_model_and_matches_nvidia(profile, tmp_path, native_sentencepiece):
    from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

    adapter = load_tokenizer_bundle(profile)
    model_file = tmp_path / "tokenizer.model"
    model_file.write_bytes(adapter.model_bytes)
    expected = native_sentencepiece(str(model_file), legacy=False)
    facade = create_nemo_tokenizer(adapter, model_file)

    assert isinstance(facade, native_sentencepiece)
    assert isinstance(facade, TokenizerSpec)
    assert facade.model_bytes == facade.backend.serialized_model_proto() == adapter.model_bytes
    assert facade.source_native_to_target_native == adapter.source_native_to_target_native
    assert facade.blank_id == adapter.blank_id
    assert facade.pad_id == expected.pad_id == -1
    assert facade.vocab == expected.vocab
    assert len(facade.vocab) == facade.vocab_size == adapter.blank_id
    assert facade.get_vocab() == dict(enumerated_vocab(expected.vocab))
    assert facade.tokenizer.get_vocab() == facade.get_vocab()
    assert facade.get_active_vocab() == adapter.get_vocab()
    assert len(facade.get_active_vocab()) == adapter.active_vocab_size
    assert set(facade.get_vocab().values()) == set(range(facade.vocab_size))

    for text in ("", "which", "  a  word ", "नमस्ते भारत", "हिन्दी मराठी", "அவள்", "അവന്‍", "Ａ ﬁ", "a\u200cb", "🙂unknown"):
        ids = expected.text_to_ids(text)
        assert facade.text_to_ids(text) == ids
        assert TokenizerWrapper(facade)(text, "hi") == ids
        assert facade.ids_to_text(ids) == expected.ids_to_text(ids)
        assert facade.text_to_tokens(text) == expected.text_to_tokens(text)
        assert facade.ids_to_tokens(ids) == expected.ids_to_tokens(ids)
        assert facade.tokenizer.encode(text) == expected.tokenizer.encode(text)
        assert facade.tokenizer.decode(ids) == expected.tokenizer.decode(ids)
    for tokens in ("▁which", "<en-US>", "<hi-IN>", "missing-nonexistent-piece", ["▁which", "a"], ["a", "<en-US>", "a"]):
        assert facade.tokens_to_ids(tokens) == expected.tokens_to_ids(tokens)
        assert facade.tokens_to_ids(tokens, tokens_to_skip=["a"]) == expected.tokens_to_ids(tokens, tokens_to_skip=["a"])
    assert facade.token_to_id("missing-nonexistent-piece") == expected.unk_id
    assert facade.tokens_to_ids("▁which") == [expected.token_to_id("▁which")]
    assert facade.tokens_to_ids("▁which") == [2971]
    assert facade.tokens_to_ids("<en-US>") == [2947]

    # No path is retained: worker serialization and renamed NeMo artifacts must
    # continue to work after the reconstruction directory has disappeared.
    model_file.unlink()
    for restored in (copy.deepcopy(facade), pickle.loads(pickle.dumps(facade))):
        assert restored.tokenizer.get_vocab() == facade.get_vocab()
        assert restored.get_active_vocab() == facade.get_active_vocab()
        assert restored.text_to_ids("which हिन्दी") == facade.text_to_ids("which हिन्दी")
        assert restored.model_bytes == adapter.model_bytes


def enumerated_vocab(vocab):
    return ((token, index) for index, token in enumerate(vocab))


def test_native_sampling_and_boosting_are_inherited(tmp_path, native_sentencepiece):
    adapter = load_tokenizer_bundle("latin-indic")
    path = tmp_path / "tokenizer.model"
    path.write_bytes(adapter.model_bytes)
    facade = create_nemo_tokenizer(adapter, path)
    expected = native_sentencepiece(str(path), legacy=False)
    # These are NVIDIA's methods, not a second implementation in Untok.
    for name in ("text_to_ids", "tokens_to_ids", "token_to_id", "text_to_ids_var_bpe"):
        assert getattr(type(facade), name) is getattr(native_sentencepiece, name)
    for text in ("which words should be recognized", "भारत में हिन्दी बोली जाती है"):
        samples = [facade.text_to_ids(text, sample_alpha=0.1) for _ in range(8)]
        assert all(facade.ids_to_text(ids) == expected.ids_to_text(expected.text_to_ids(text)) for ids in samples)
        assert all(set(ids).isdisjoint(adapter.inactive_native_ids) for ids in samples)
        for case_insensitive in (False, True):
            assert facade.text_to_ids_var_bpe(text, case_insensitive) == expected.text_to_ids_var_bpe(text, case_insensitive)


def test_facade_refuses_different_model_bytes(tmp_path, native_sentencepiece):
    path = tmp_path / "tokenizer.model"
    path.write_bytes(load_tokenizer_bundle("full").model_bytes)
    with pytest.raises(ValueError, match="differ from the verified bundle"):
        create_nemo_tokenizer(load_tokenizer_bundle("original"), path)


def test_text_only_import_does_not_load_nemo_or_torch():
    result = subprocess.run([sys.executable, "-c", "import sys; import untok.nemo_tokenizer; from untok.bundles import load_tokenizer; load_tokenizer('original'); assert 'nemo' not in sys.modules; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

@pytest.mark.parametrize('profile', ['original', 'latin', 'latin-indic', 'full'])
def test_every_physical_piece_uses_native_string_and_list_lookup(profile, tmp_path, native_sentencepiece):
    adapter = load_tokenizer_bundle(profile)
    path = tmp_path / 'tokenizer.model'
    path.write_bytes(adapter.model_bytes)
    expected = native_sentencepiece(str(path), legacy=False)
    facade = create_nemo_tokenizer(adapter, path)
    for piece in expected.vocab:
        assert facade.tokens_to_ids(piece) == expected.tokens_to_ids(piece)
        assert facade.tokens_to_ids([piece]) == expected.tokens_to_ids([piece])
        assert facade.token_to_id(piece) == expected.token_to_id(piece)
