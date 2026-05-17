"""
Inference utilities for next-word prediction and text generation.

Public API required by the assignment:

    generate_text(seed_text, next_words, temperature=1.0, top_k=5)

The function also accepts optional ``model`` and ``tokenizer`` arguments so the
notebook and scripts can avoid reloading artifacts for every call.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    from .config import DEFAULT_CONFIG
    from .model import build_attention_extractor, load_trained_model
    from .preprocessor import SEQUENCE_LENGTH, clean_text, decode_tokens, load_tokenizer
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG
    from model import build_attention_extractor, load_trained_model
    from preprocessor import SEQUENCE_LENGTH, clean_text, decode_tokens, load_tokenizer


LOGGER = logging.getLogger(__name__)
DecodingStrategy = Literal["greedy", "top_k", "nucleus", "multinomial"]

OUTPUTS_DIR = DEFAULT_CONFIG.outputs.output_dir
BEST_MODEL_PATH = DEFAULT_CONFIG.outputs.best_model_path
SEED_PHRASES = [
    "It was a dark and stormy",
    "Holmes looked at the",
    "The game is",
    "Watson I need your",
    "She entered the room and",
]


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_generation_artifacts(
    model_path: str | Path = BEST_MODEL_PATH,
):
    """Load the trained Keras model and tokenizer."""
    model = load_trained_model(model_path)
    tokenizer = load_tokenizer()
    return model, tokenizer


def _index_to_word(tokenizer) -> dict[int, str]:
    return {index: word for word, index in tokenizer.word_index.items()}


def _normalize_distribution(probs: np.ndarray) -> np.ndarray:
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0:
        probs = np.ones_like(probs, dtype=np.float64)
        probs[0] = 0.0
        total = float(np.sum(probs))
    return probs / total


def _prepare_probabilities(
    raw_probs: np.ndarray,
    *,
    tokenizer,
    generated_token_ids: list[int],
    temperature: float,
    repetition_penalty: float,
) -> np.ndarray:
    """Apply decoding-time probability adjustments."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probs = np.asarray(raw_probs, dtype=np.float64).copy()
    probs[0] = 0.0
    oov_id = tokenizer.word_index.get(DEFAULT_CONFIG.preprocessing.oov_token)
    if oov_id is not None and oov_id < len(probs):
        probs[oov_id] = 0.0

    if repetition_penalty > 1.0:
        for token_id in generated_token_ids[-8:]:
            if 0 <= token_id < len(probs):
                probs[token_id] /= repetition_penalty

    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= np.max(logits)
    adjusted = np.exp(logits)
    adjusted[0] = 0.0
    if oov_id is not None and oov_id < len(adjusted):
        adjusted[oov_id] = 0.0
    return _normalize_distribution(adjusted)


def _top_candidates(
    probs: np.ndarray,
    tokenizer,
    *,
    top_k: int,
) -> list[dict[str, float]]:
    index_to_word = _index_to_word(tokenizer)
    candidate_indices = np.argsort(probs)[::-1][:top_k]
    return [
        {
            "word": index_to_word.get(int(index), "<UNK>"),
            "probability": float(probs[index]),
        }
        for index in candidate_indices
    ]


def _sample_next_id(
    probs: np.ndarray,
    *,
    strategy: DecodingStrategy,
    top_k: int,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    """Choose the next token id according to a decoding strategy."""
    if strategy == "greedy":
        return int(np.argmax(probs))

    if strategy == "top_k":
        k = max(1, min(top_k, len(probs)))
        indices = np.argsort(probs)[::-1][:k]
        local_probs = _normalize_distribution(probs[indices])
        return int(rng.choice(indices, p=local_probs))

    if strategy == "nucleus":
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        keep_count = int(np.searchsorted(cumulative, top_p, side="left") + 1)
        keep_indices = sorted_indices[: max(1, keep_count)]
        local_probs = _normalize_distribution(probs[keep_indices])
        return int(rng.choice(keep_indices, p=local_probs))

    if strategy == "multinomial":
        return int(rng.choice(np.arange(len(probs)), p=probs))

    raise ValueError(f"Unknown decoding strategy: {strategy}")


def _context_array(text: str, tokenizer, seq_len: int) -> np.ndarray:
    cleaned = clean_text(text)
    token_ids = tokenizer.texts_to_sequences([cleaned])[0][-seq_len:]
    return pad_sequences([token_ids], maxlen=seq_len, padding="pre")


def predict_top_k(
    seed_text: str,
    model: tf.keras.Model,
    tokenizer,
    *,
    seq_len: int = SEQUENCE_LENGTH,
    temperature: float = 1.0,
    top_k: int = 5,
    repetition_penalty: float = 1.0,
) -> list[dict[str, float]]:
    """Return top-k next-word predictions and probabilities for one context."""
    context = _context_array(seed_text, tokenizer, seq_len)
    raw_probs = model.predict(context, verbose=0)[0]
    adjusted = _prepare_probabilities(
        raw_probs,
        tokenizer=tokenizer,
        generated_token_ids=[],
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    return _top_candidates(adjusted, tokenizer, top_k=top_k)


def generate_text(
    seed_text: str,
    next_words: int | None = None,
    *,
    model: tf.keras.Model | None = None,
    tokenizer: Any | None = None,
    seq_len: int = SEQUENCE_LENGTH,
    temperature: float = 1.0,
    top_k: int = 5,
    strategy: DecodingStrategy = "top_k",
    top_p: float = 0.90,
    repetition_penalty: float = 1.15,
    random_seed: int | None = None,
    max_phrase_len: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Generate text one token at a time.

    Args:
        seed_text: Starting phrase.
        next_words: Number of new words to generate.
        model: Optional trained model. Loaded automatically when omitted.
        tokenizer: Optional fitted tokenizer. Loaded automatically when omitted.
        seq_len: Context window length used during training.
        temperature: Softmax temperature. Lower is safer; higher is more varied.
        top_k: Number of predictions reported and used by top-k sampling.
        strategy: ``greedy``, ``top_k``, ``nucleus``, or ``multinomial``.
        top_p: Probability mass used by nucleus sampling.
        repetition_penalty: Penalty applied to recently generated token ids.
        random_seed: Optional deterministic generation seed.
        max_phrase_len: Backward-compatible alias for ``next_words``.

    Returns:
        ``(generated_text, step_details)`` where each step contains the chosen
        word plus top-k probability candidates.
    """
    if next_words is None:
        next_words = max_phrase_len if max_phrase_len is not None else 30
    if next_words < 0:
        raise ValueError("next_words must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not seed_text.strip():
        raise ValueError("seed_text must contain at least one word")

    if model is None or tokenizer is None:
        model, tokenizer = load_generation_artifacts()

    rng = np.random.default_rng(random_seed)
    current_text = clean_text(seed_text)
    generated_token_ids: list[int] = []
    details: list[dict[str, Any]] = []
    index_to_word = _index_to_word(tokenizer)

    for step in range(1, next_words + 1):
        context = _context_array(current_text, tokenizer, seq_len)
        raw_probs = model.predict(context, verbose=0)[0]
        adjusted_probs = _prepare_probabilities(
            raw_probs,
            tokenizer=tokenizer,
            generated_token_ids=generated_token_ids,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        top_candidates = _top_candidates(adjusted_probs, tokenizer, top_k=top_k)
        chosen_id = _sample_next_id(
            adjusted_probs,
            strategy=strategy,
            top_k=top_k,
            top_p=top_p,
            rng=rng,
        )
        chosen_word = index_to_word.get(chosen_id, top_candidates[0]["word"])
        if chosen_word in {"<UNK>", DEFAULT_CONFIG.preprocessing.oov_token}:
            chosen_word = top_candidates[0]["word"]
            chosen_id = tokenizer.word_index.get(chosen_word, chosen_id)

        generated_token_ids.append(int(chosen_id))
        context_words = current_text.split()[-seq_len:]
        details.append(
            {
                "step": step,
                "context": " ".join(context_words),
                "chosen": chosen_word,
                "chosen_probability": float(adjusted_probs[chosen_id]),
                "top5": top_candidates[:5],
                "top_k": top_candidates,
                "strategy": strategy,
                "temperature": temperature,
            }
        )
        current_text = f"{current_text} {chosen_word}".strip()

    return current_text, details


def attention_for_seed(
    seed_text: str,
    model: tf.keras.Model,
    tokenizer,
    *,
    seq_len: int = SEQUENCE_LENGTH,
) -> tuple[list[str], np.ndarray]:
    """Return context tokens and attention weights for visualization."""
    cleaned = clean_text(seed_text)
    token_ids = tokenizer.texts_to_sequences([cleaned])[0][-seq_len:]
    context = pad_sequences([token_ids], maxlen=seq_len, padding="pre")
    extractor = build_attention_extractor(model)
    weights = extractor.predict(context, verbose=0)[0]
    non_padding_start = seq_len - len(token_ids)
    tokens = decode_tokens(token_ids, tokenizer)
    return tokens, np.asarray(weights[non_padding_start:], dtype=float)


def show_top5_table(words_and_probabilities: list[dict[str, Any]]) -> None:
    """Print a readable top-5 probability table for generation steps."""
    print("\n" + "=" * 72)
    print("Step-by-step Top-5 Prediction Breakdown")
    print("=" * 72)
    for entry in words_and_probabilities:
        print(f"\nStep {entry['step']:>3} | Context: \"{entry['context']}\"")
        print(
            f"         | Chosen : \"{entry['chosen']}\" "
            f"({entry['chosen_probability'] * 100:.2f}%)"
        )
        for rank, candidate in enumerate(entry["top5"], start=1):
            marker = " <- chosen" if candidate["word"] == entry["chosen"] else ""
            print(
                f"         | Rank {rank}: {candidate['word']:<18} "
                f"{candidate['probability'] * 100:>7.2f}%{marker}"
            )
    print("\n" + "=" * 72)


def run_demo(
    *,
    model_path: str | Path = BEST_MODEL_PATH,
    next_words: int = 35,
    temperature: float = 0.8,
    top_k: int = 5,
    strategy: DecodingStrategy = "top_k",
) -> None:
    """Generate examples for the seed phrases used in the notebook/report."""
    model, tokenizer = load_generation_artifacts(model_path)
    for index, seed in enumerate(SEED_PHRASES, start=1):
        output_text, breakdown = generate_text(
            seed,
            next_words=next_words,
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            top_k=top_k,
            strategy=strategy,
            random_seed=DEFAULT_CONFIG.training.seed + index,
        )
        print(f"\nExample {index}: {seed!r}")
        print(output_text)
        if index == 1:
            show_top5_table(breakdown)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with the trained model.")
    parser.add_argument("--seed", default=None, help="Seed phrase. Runs demo when omitted.")
    parser.add_argument("--next-words", type=int, default=35)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--strategy",
        choices=["greedy", "top_k", "nucleus", "multinomial"],
        default="top_k",
    )
    parser.add_argument("--model-path", default=str(BEST_MODEL_PATH))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    if args.seed:
        loaded_model, loaded_tokenizer = load_generation_artifacts(args.model_path)
        text, step_details = generate_text(
            args.seed,
            next_words=args.next_words,
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            temperature=args.temperature,
            top_k=args.top_k,
            strategy=args.strategy,
        )
        print(text)
        show_top5_table(step_details)
    else:
        run_demo(
            model_path=args.model_path,
            next_words=args.next_words,
            temperature=args.temperature,
            top_k=args.top_k,
            strategy=args.strategy,
        )
