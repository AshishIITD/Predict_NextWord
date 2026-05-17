"""
Training pipeline for the LSTM + attention next-word predictor.

Artifacts written to ``outputs/``:
    - best_model.keras
    - final_model.keras
    - training_history.png
    - training_history.json
    - training_history.csv
    - metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

try:
    from .config import DEFAULT_CONFIG, AppConfig, TrainingConfig
    from .data_loader import download_corpus, load_corpus
    from .model import build_model, print_model_summary
    from .preprocessor import preprocess
    from .visualization import plot_training_history
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG, AppConfig, TrainingConfig
    from data_loader import download_corpus, load_corpus
    from model import build_model, print_model_summary
    from preprocessor import preprocess
    from visualization import plot_training_history


LOGGER = logging.getLogger(__name__)

ROOT_DIR = DEFAULT_CONFIG.outputs.output_dir.parent
OUTPUTS_DIR = DEFAULT_CONFIG.outputs.output_dir
BEST_MODEL_PATH = DEFAULT_CONFIG.outputs.best_model_path
FINAL_MODEL_PATH = DEFAULT_CONFIG.outputs.final_model_path
HISTORY_PLOT = DEFAULT_CONFIG.outputs.history_plot_path


def configure_logging(level: int = logging.INFO) -> None:
    """Configure console logging for command-line runs."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Create deterministic train/validation/test splits."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    test_size = int(len(X) * test_fraction)
    val_size = int(len(X) * validation_fraction)
    train_end = len(X) - val_size - test_size
    val_end = len(X) - test_size

    return {
        "train": (X[:train_end], y[:train_end]),
        "validation": (X[train_end:val_end], y[train_end:val_end]),
        "test": (X[val_end:], y[val_end:]),
    }


def make_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Build an efficient TensorFlow dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(X), 20_000), seed=seed)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_callbacks(
    training_config: TrainingConfig = DEFAULT_CONFIG.training,
    app_config: AppConfig = DEFAULT_CONFIG,
) -> list[tf.keras.callbacks.Callback]:
    """Return checkpointing, early stopping, LR scheduling, and CSV logging."""
    output_config = app_config.outputs
    output_config.output_dir.mkdir(parents=True, exist_ok=True)
    return [
        ModelCheckpoint(
            filepath=str(output_config.best_model_path),
            monitor=training_config.monitor_metric,
            mode=training_config.monitor_mode,
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor=training_config.monitor_metric,
            mode=training_config.monitor_mode,
            patience=training_config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=training_config.reduce_lr_factor,
            patience=training_config.reduce_lr_patience,
            min_lr=training_config.min_learning_rate,
            verbose=1,
        ),
        CSVLogger(str(output_config.history_csv_path)),
    ]


def compute_perplexity_from_loss(loss: float) -> float:
    """Perplexity is exp(cross-entropy loss)."""
    return float(np.exp(np.clip(loss, a_min=None, a_max=50)))


def evaluate_model(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = DEFAULT_CONFIG.training.batch_size,
) -> dict[str, float]:
    """Evaluate loss, top-1 accuracy, top-5 accuracy, and perplexity."""
    values = model.evaluate(X, y, batch_size=batch_size, verbose=0)
    metric_names = model.metrics_names
    metrics = {name: float(value) for name, value in zip(metric_names, values)}
    metrics["perplexity"] = compute_perplexity_from_loss(metrics["loss"])
    return metrics


def save_history_json(history: tf.keras.callbacks.History, path: str | Path) -> Path:
    """Persist epoch-wise metrics as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: [float(value) for value in values] for key, values in history.history.items()
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return path


def save_metrics_json(metrics: dict[str, Any], path: str | Path) -> Path:
    """Persist final train/validation/test metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def save_metrics_csv(metrics: dict[str, dict[str, float]], path: str | Path) -> Path:
    """Optional tabular metric summary for quick inspection."""
    path = Path(path)
    rows = []
    for split, split_metrics in metrics.items():
        for metric_name, value in split_metrics.items():
            rows.append({"split": split, "metric": metric_name, "value": value})
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["split", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def train(
    *,
    app_config: AppConfig = DEFAULT_CONFIG,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> tuple[tf.keras.Model, Any, int, tf.keras.callbacks.History, dict[str, Any]]:
    """Run the full training pipeline and return model, tokenizer, and metrics."""
    app_config.ensure_directories()
    training_config = app_config.training
    if epochs is not None:
        training_config = replace(training_config, epochs=epochs)
    if batch_size is not None:
        training_config = replace(training_config, batch_size=batch_size)

    set_random_seed(training_config.seed)

    LOGGER.info("Loading and preprocessing corpus")
    download_corpus(app_config.data.corpus_path, config=app_config.data)
    raw_text = load_corpus(app_config.data.corpus_path)
    X, y, tokenizer, vocab_size = preprocess(
        raw_text,
        max_vocab=app_config.preprocessing.max_vocab_size,
        seq_len=app_config.preprocessing.sequence_length,
        save_tok=True,
        config=app_config.preprocessing,
    )

    splits = split_dataset(
        X,
        y,
        validation_fraction=training_config.validation_fraction,
        test_fraction=training_config.test_fraction,
        seed=training_config.seed,
    )
    for split_name, (split_X, _) in splits.items():
        LOGGER.info("%s samples: %s", split_name.title(), f"{len(split_X):,}")

    train_dataset = make_dataset(
        *splits["train"],
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=training_config.seed,
    )
    validation_dataset = make_dataset(
        *splits["validation"],
        batch_size=training_config.batch_size,
        shuffle=False,
        seed=training_config.seed,
    )

    LOGGER.info("Building model")
    model = build_model(
        vocab_size=vocab_size,
        seq_len=app_config.preprocessing.sequence_length,
        embed_dim=app_config.model.embedding_dim,
        lstm_units=app_config.model.lstm_units,
        attention_units=app_config.model.attention_units,
        dense_units=app_config.model.dense_units,
        embedding_dropout=app_config.model.embedding_dropout,
        lstm_dropout=app_config.model.lstm_dropout,
        recurrent_dropout=app_config.model.recurrent_dropout,
        dense_dropout_1=app_config.model.dense_dropout_1,
        dense_dropout_2=app_config.model.dense_dropout_2,
        learning_rate=app_config.model.learning_rate,
        clipnorm=app_config.model.clipnorm,
    )
    print_model_summary(model)

    LOGGER.info("Starting training for up to %s epochs", training_config.epochs)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=training_config.epochs,
        callbacks=get_callbacks(training_config, app_config),
        verbose=1,
    )

    LOGGER.info("Evaluating final model")
    final_metrics = {
        split_name: evaluate_model(
            model,
            split_X,
            split_y,
            batch_size=training_config.batch_size,
        )
        for split_name, (split_X, split_y) in splits.items()
    }

    model.save(str(app_config.outputs.final_model_path))
    save_history_json(history, app_config.outputs.history_json_path)
    save_metrics_json(final_metrics, app_config.outputs.metrics_json_path)
    save_metrics_csv(
        final_metrics,
        app_config.outputs.output_dir / "metrics.csv",
    )
    plot_training_history(history, app_config.outputs.history_plot_path)

    LOGGER.info("Final model saved to %s", app_config.outputs.final_model_path)
    LOGGER.info("Best model checkpoint saved to %s", app_config.outputs.best_model_path)
    LOGGER.info("Metrics saved to %s", app_config.outputs.metrics_json_path)
    return model, tokenizer, vocab_size, history, final_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Sherlock Holmes next-word model.")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override configured batch size."
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    train(epochs=args.epochs, batch_size=args.batch_size)
