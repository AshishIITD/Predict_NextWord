"""Plotting helpers for training curves, perplexity, and attention weights."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .config import DEFAULT_CONFIG
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG


def _history_to_dict(history: object) -> Mapping[str, Sequence[float]]:
    """Accept a Keras History object or a plain dictionary."""
    if hasattr(history, "history"):
        return history.history
    if isinstance(history, Mapping):
        return history
    raise TypeError("history must be a Keras History object or mapping")


def plot_training_history(
    history: object,
    save_path: str | Path = DEFAULT_CONFIG.outputs.history_plot_path,
) -> Path:
    """Save loss, accuracy, top-5 accuracy, and perplexity curves."""
    history_dict = _history_to_dict(history)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history_dict.get("loss", [])) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(epochs, history_dict.get("loss", []), label="Train")
    axes[0, 0].plot(epochs, history_dict.get("val_loss", []), label="Validation")
    axes[0, 0].set_title("Cross-Entropy Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history_dict.get("accuracy", []), label="Train")
    axes[0, 1].plot(epochs, history_dict.get("val_accuracy", []), label="Validation")
    axes[0, 1].set_title("Top-1 Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    top5_key = "top5_accuracy" if "top5_accuracy" in history_dict else "top5_acc"
    val_top5_key = "val_top5_accuracy" if "val_top5_accuracy" in history_dict else "val_top5_acc"
    axes[1, 0].plot(epochs, history_dict.get(top5_key, []), label="Train")
    axes[1, 0].plot(epochs, history_dict.get(val_top5_key, []), label="Validation")
    axes[1, 0].axhline(0.80, linestyle="--", color="green",  label="Train target 80%")
    axes[1, 0].axhline(0.75, linestyle="--", color="orange", label="Val target 75%")
    axes[1, 0].set_title("Top-5 Accuracy  ← assignment target")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    train_perplexity = np.exp(np.asarray(history_dict.get("loss", []), dtype=float))
    val_perplexity = np.exp(np.asarray(history_dict.get("val_loss", []), dtype=float))
    axes[1, 1].plot(epochs, train_perplexity, label="Train")
    axes[1, 1].plot(epochs, val_perplexity, label="Validation")
    axes[1, 1].axhline(250, linestyle="--", color="purple", label="Perplexity target")
    axes[1, 1].set_title("Perplexity")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("exp(loss)")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_attention_heatmap(
    tokens: Sequence[str],
    weights: Sequence[float],
    save_path: str | Path = DEFAULT_CONFIG.outputs.attention_heatmap_path,
    title: str = "Attention Weights for Next-Word Prediction",
) -> Path:
    """Save a compact heatmap showing attention mass over context tokens."""
    if len(tokens) != len(weights):
        raise ValueError("tokens and weights must have the same length")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    weights_array = np.asarray(weights, dtype=float).reshape(1, -1)
    fig_width = max(8, min(16, len(tokens) * 0.85))
    fig, ax = plt.subplots(figsize=(fig_width, 2.5))
    image = ax.imshow(weights_array, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=35, ha="right")
    ax.set_yticks([])
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Attention weight")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
