"""Standalone evaluation script for a saved next-word prediction model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

try:
    from .config import DEFAULT_CONFIG
    from .data_loader import download_corpus, load_corpus
    from .model import load_trained_model
    from .preprocessor import clean_text, generate_sequences, load_tokenizer
    from .train import evaluate_model, split_dataset
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG
    from data_loader import download_corpus, load_corpus
    from model import load_trained_model
    from preprocessor import clean_text, generate_sequences, load_tokenizer
    from train import evaluate_model, split_dataset


LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def evaluate_saved_model(
    model_path: str | Path = DEFAULT_CONFIG.outputs.best_model_path,
    metrics_path: str | Path = DEFAULT_CONFIG.outputs.output_dir / "evaluation_metrics.json",
) -> dict[str, float]:
    """Load artifacts, rebuild the deterministic test split, and evaluate."""
    DEFAULT_CONFIG.ensure_directories()
    model = load_trained_model(model_path)
    download_corpus()
    raw_text = load_corpus()
    tokenizer = load_tokenizer()
    clean_corpus = clean_text(raw_text)
    X, y = generate_sequences(
        clean_corpus,
        tokenizer,
        seq_len=DEFAULT_CONFIG.preprocessing.sequence_length,
        stride=DEFAULT_CONFIG.preprocessing.stride,
        skip_oov_targets=DEFAULT_CONFIG.preprocessing.skip_oov_targets,
    )
    if np.max(y) >= model.output_shape[-1]:
        raise ValueError(
            "Tokenizer target ids exceed model output dimension. Retrain the model "
            "with the current tokenizer/configuration."
        )

    splits = split_dataset(
        X,
        y,
        validation_fraction=DEFAULT_CONFIG.training.validation_fraction,
        test_fraction=DEFAULT_CONFIG.training.test_fraction,
        seed=DEFAULT_CONFIG.training.seed,
    )
    metrics = evaluate_model(
        model,
        *splits["test"],
        batch_size=DEFAULT_CONFIG.training.batch_size,
    )

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Evaluation metrics saved to %s", metrics_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved next-word model.")
    parser.add_argument("--model-path", default=str(DEFAULT_CONFIG.outputs.best_model_path))
    parser.add_argument(
        "--metrics-path",
        default=str(DEFAULT_CONFIG.outputs.output_dir / "evaluation_metrics.json"),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    result = evaluate_saved_model(args.model_path, args.metrics_path)
    print(json.dumps(result, indent=2))
