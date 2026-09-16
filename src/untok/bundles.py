"""Load the four published SentencePiece models without rebuilding them."""
from __future__ import annotations

import hashlib
from importlib import resources
import json
import operator
from pathlib import Path

import sentencepiece as spm

PROFILES = ("original", "latin", "latin-indic", "full")
# Published e2f8acd manifests pin every model and metadata file independently.
_MANIFEST_SHA256 = {
    "original": "4839890ba1106ab6e9648806c8282893e31f116bc05f8c13eb0e222e5e06095f",
    "latin": "8302b9537bae83ad9e2d86437fae560da1581694c22ab9414251ab3b41c156e6",
    "latin-indic": "a9e4a89aae55f3f6369af6c64882e2c1dfe6ea7d1e08d5200b3c9063821a7a81",
    "full": "02ede11284b37eb6b0da1b14f44232ba1f5e358a5ff84c82001730959dd69f29",
}


class Tokenizer:
    """Native text IDs; RNNT blank is outside the SentencePiece vocabulary."""

    def __init__(self, directory):
        raw_manifest = directory.joinpath("manifest.json").read_bytes()
        manifest = json.loads(raw_manifest)
        profile = manifest.get("profile")
        if hashlib.sha256(raw_manifest).hexdigest() != _MANIFEST_SHA256.get(profile):
            raise ValueError("Bundle manifest differs from the published current profile")
        files = {}
        for name, expected in manifest["files"].items():
            raw = directory.joinpath(name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Bundle file hash mismatch: {name}")
            files[name] = raw
        self.model_bytes = files["tokenizer.model"]
        self.base_model_bytes = files["base-tokenizer.model"]
        self.tokenizer_sha256 = manifest["tokenizer_sha256"]
        self.profile = profile
        self.backend = spm.SentencePieceProcessor(model_proto=self.model_bytes)
        self.tokenizer = self
        self.vocab_size = self.backend.get_piece_size()
        self.blank_id = manifest["native_blank_id"]
        self.pad_id = self.blank_id
        mapping = json.loads(files["native-row-map.json"])
        self.source_native_to_target_native = tuple(mapping["source_native_to_target_native"])
        self.inactive_native_ids = tuple(mapping["inactive_native_ids"])
        self._inactive = frozenset(self.inactive_native_ids)
        self.active_native_ids = tuple(i for i in range(self.vocab_size) if i not in self._inactive)
        self.active_vocab_size = len(self.active_native_ids)
        if (self.blank_id != self.vocab_size or self.active_vocab_size != manifest["active_vocabulary_size"]
                or self.source_native_to_target_native != (*range(13087), self.blank_id)
                or mapping["target_blank_id"] != self.blank_id):
            raise ValueError("Native token layout differs from the published profile")
        self.unk_id, self.bos_id, self.eos_id = self.backend.unk_id(), self.backend.bos_id(), self.backend.eos_id()
        self.vocab = self.get_vocab()

    def get_vocab(self):
        return {self.backend.id_to_piece(i): i for i in self.active_native_ids}

    def get_acoustic_vocab(self):
        return {self.backend.id_to_piece(i): i for i in range(self.vocab_size)}

    def text_to_ids(self, text, sample_alpha=None):
        options = {} if sample_alpha is None else {"enable_sampling": True, "alpha": sample_alpha, "nbest_size": -1}
        return self.backend.encode(text, out_type=int, **options)

    def __call__(self, text):
        return self.text_to_ids(text)

    def _text_ids(self, ids):
        result = []
        for value in ids:
            try:
                if isinstance(value, bool):
                    raise TypeError("Boolean token ID")
                value = operator.index(value)
            except TypeError as error:
                raise ValueError("Token IDs must be integers") from error
            if not 0 <= value <= self.blank_id:
                raise ValueError("Native token ID out of range")
            if value in self._inactive:
                raise ValueError("Inactive reserved token ID is not a text label")
            if value != self.blank_id:
                result.append(value)
        return result

    def ids_to_text(self, ids):
        return self.backend.decode(self._text_ids(ids))

    def text_to_tokens(self, text):
        return self.backend.encode(text, out_type=str)

    def ids_to_tokens(self, ids):
        return [self.backend.id_to_piece(i) for i in self._text_ids(ids)]

    def token_to_id(self, token):
        if token not in self.vocab:
            raise ValueError(f"Unknown token string: {token!r}")
        return self.vocab[token]

    def id_to_token(self, index):
        checked = self._text_ids([index])
        return self.backend.id_to_piece(checked[0]) if checked else "<blank>"

    def tokens_to_ids(self, tokens, tokens_to_skip=()):
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.token_to_id(token) for token in tokens if token not in tokens_to_skip]

    def tokens_to_text(self, tokens):
        return self.ids_to_text(self.tokens_to_ids(tokens))


def load_tokenizer_bundle(directory):
    """Choose a packaged profile or a directory containing that exact profile."""
    if isinstance(directory, str) and directory in PROFILES:
        directory = resources.files("untok").joinpath("data", directory)
    else:
        directory = Path(directory)
    return Tokenizer(directory)


load_tokenizer = load_tokenizer_bundle
