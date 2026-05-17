"""
Text cleaning, tokenization, and supervised sequence generation.

The model predicts token t+1 from the previous ``SEQUENCE_LENGTH`` tokens. Labels
are sparse integer token ids, which keeps memory use low compared with one-hot
targets over a 10k-word vocabulary.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

try:
    from .config import DEFAULT_CONFIG, PreprocessingConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG, PreprocessingConfig


LOGGER = logging.getLogger(__name__)
PREPROCESSING_CONFIG = DEFAULT_CONFIG.preprocessing
MAX_VOCAB_SIZE = PREPROCESSING_CONFIG.max_vocab_size
SEQUENCE_LENGTH = PREPROCESSING_CONFIG.sequence_length
TOKENIZER_PATH = DEFAULT_CONFIG.data.tokenizer_path


def clean_text(text: str, *, lowercase: bool = True) -> str:
    """
    Normalize Project Gutenberg prose for word-level language modeling.
    Preserves essential punctuation and adds <EOS> tokens.
    """
    if lowercase:
        text = text.lower()

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": " ",
        "\u2013": " ",
    }

    for src, tgt in replacements.items():
        text = text.replace(src, tgt)

    # remove chapter headings only (safely)
    text = re.sub(r"\n\s*chapter\s+[ivxlcdm]+\s*\n", " ", text)

    # preserve punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\'\"\;\:]", " ", text)

    # optional EOS token
    text = text.replace(".", " <EOS> ")

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_tokenizer(
    clean_corpus: str,
    max_vocab: int = MAX_VOCAB_SIZE,
    oov_token: str = PREPROCESSING_CONFIG.oov_token,
) -> Tokenizer:
    """
    Fit a Keras tokenizer with a capped vocabulary.

    ``max_vocab`` means up to that many word ids in addition to padding id 0.
    Keras keeps tokens with ids strictly less than ``num_words``, so we pass
    ``max_vocab + 1`` to retain ids 1..max_vocab.
    """
    tokenizer = Tokenizer(num_words=max_vocab + 1, oov_token=oov_token, filters="")
    tokenizer.fit_on_texts([clean_corpus])
    LOGGER.info("Unique corpus tokens before cap: %s", f"{len(tokenizer.word_index):,}")
    LOGGER.info(
        "Model vocabulary size including padding: %s",
        f"{vocabulary_size(tokenizer, max_vocab):,}",
    )
    return tokenizer


def vocabulary_size(tokenizer: Tokenizer, max_vocab: int = MAX_VOCAB_SIZE) -> int:
    """Return the Dense/Embedding vocabulary size including padding id 0."""
    return min(max_vocab, len(tokenizer.word_index)) + 1


def save_tokenizer(tokenizer: Tokenizer, path: str | Path = TOKENIZER_PATH) -> Path:
    """Persist the fitted tokenizer for inference and evaluation scripts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        pickle.dump(tokenizer, file_obj)
    LOGGER.info("Tokenizer saved to %s", path)
    return path


def load_tokenizer(path: str | Path = TOKENIZER_PATH) -> Tokenizer:
    """Load a previously saved Keras tokenizer."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {path}. Run training first.")
    with path.open("rb") as file_obj:
        tokenizer = pickle.load(file_obj)
    LOGGER.info("Tokenizer loaded from %s", path)
    return tokenizer


def texts_to_token_ids(clean_corpus: str, tokenizer: Tokenizer) -> np.ndarray:
    """Convert cleaned text to a 1D array of token ids."""
    token_ids = tokenizer.texts_to_sequences([clean_corpus])[0]
    return np.asarray(token_ids, dtype=np.int32)


def generate_sequences(
    clean_corpus: str,
    tokenizer: Tokenizer,
    seq_len: int = SEQUENCE_LENGTH,
    *,
    stride: int = PREPROCESSING_CONFIG.stride,
    skip_oov_targets: bool = PREPROCESSING_CONFIG.skip_oov_targets,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised next-word examples with a sliding context window.

    For position ``i``:
        X[i] = tokens[i : i + seq_len]
        y[i] = tokens[i + seq_len]
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    token_ids = texts_to_token_ids(clean_corpus, tokenizer)
    if len(token_ids) <= seq_len:
        raise ValueError(
            f"Corpus has only {len(token_ids)} tokens, which is too short for "
            f"sequence_length={seq_len}."
        )

    max_samples = (len(token_ids) - seq_len + stride - 1) // stride
    X = np.empty((max_samples, seq_len), dtype=np.int32)
    y = np.empty((max_samples,), dtype=np.int32)

    cursor = 0
    for start in range(0, len(token_ids) - seq_len, stride):
        target = int(token_ids[start + seq_len])
        if skip_oov_targets and target <= 1:
            continue
        X[cursor] = token_ids[start : start + seq_len]
        y[cursor] = target
        cursor += 1

    X = X[:cursor]
    y = y[:cursor]
    LOGGER.info("Generated %s supervised sequences", f"{len(X):,}")
    LOGGER.info("X shape=%s, y shape=%s", X.shape, y.shape)
    return X, y


def summarize_vocabulary(
    tokenizer: Tokenizer,
    n: int = 20,
) -> list[tuple[str, int]]:
    """Return the n most frequent tokens and counts."""
    word_counts = tokenizer.word_counts
    sorted_items = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    return [(word, int(count)) for word, count in sorted_items[:n]]


def decode_tokens(token_ids: Iterable[int], tokenizer: Tokenizer) -> list[str]:
    """Map integer ids back to token strings for inspection."""
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    return [index_to_word.get(int(token_id), "<PAD>") for token_id in token_ids]


def preprocess(
    raw_text: str,
    max_vocab: int = MAX_VOCAB_SIZE,
    seq_len: int = SEQUENCE_LENGTH,
    save_tok: bool = True,
    *,
    config: PreprocessingConfig = PREPROCESSING_CONFIG,
) -> tuple[np.ndarray, np.ndarray, Tokenizer, int]:
    """
    Complete preprocessing pipeline: clean, tokenize, save tokenizer, sequences.

    Returns:
        X: integer context arrays of shape ``(samples, seq_len)``.
        y: sparse integer targets of shape ``(samples,)``.
        tokenizer: fitted tokenizer.
        vocab_size: output vocabulary size including padding.
    """
    clean_corpus = clean_text(raw_text, lowercase=config.lowercase)
    tokenizer = build_tokenizer(clean_corpus, max_vocab=max_vocab, oov_token=config.oov_token)
    vocab_size = vocabulary_size(tokenizer, max_vocab=max_vocab)
    if save_tok:
        save_tokenizer(tokenizer)

    DEFAULT_CONFIG.data.cleaned_corpus_path.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONFIG.data.cleaned_corpus_path.write_text(clean_corpus, encoding="utf-8")
    X, y = generate_sequences(
        clean_corpus,
        tokenizer,
        seq_len=seq_len,
        stride=config.stride,
        skip_oov_targets=config.skip_oov_targets,
    )
    return X, y, tokenizer, vocab_size


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        from .data_loader import download_corpus, load_corpus
    except ImportError:
        from data_loader import download_corpus, load_corpus

    download_corpus()
    raw = load_corpus()
    X_data, y_data, tok, vocab = preprocess(raw)
    print(f"Ready for training: X={X_data.shape}, y={y_data.shape}, vocab={vocab}")
