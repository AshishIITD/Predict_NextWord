"""
Central configuration for the next-word prediction project.

Keeping paths and hyperparameters in one place makes the scripts, notebook, and
README describe the same experiment. Values are intentionally conservative so
the project runs on a laptop, while still exposing the full architecture needed
for the assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataConfig:
    """Dataset download and storage settings."""

    data_dir: Path = PROJECT_ROOT / "data"
    corpus_filename: str = "sherlock_holmes.txt"
    tokenizer_filename: str = "tokenizer.pkl"
    cleaned_filename: str = "sherlock_holmes_clean.txt"
    min_corpus_chars: int = 100_000
    request_timeout_seconds: int = 60
    retries_per_url: int = 3
    corpus_urls: Tuple[str, ...] = (
        "https://www.gutenberg.org/files/1661/1661-0.txt",
        "https://gutenberg.pglaf.org/files/1661/1661-0.txt",
        "https://gutenberg.readingroo.ms/1/6/6/1/1661/1661-0.txt",
        "http://www.gutenberg.org/files/1661/1661-0.txt",
    )

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / self.corpus_filename

    @property
    def tokenizer_path(self) -> Path:
        return self.data_dir / self.tokenizer_filename

    @property
    def cleaned_corpus_path(self) -> Path:
        return self.data_dir / self.cleaned_filename


@dataclass(frozen=True)
class PreprocessingConfig:
    """Text cleaning, tokenization, and sequence generation settings."""

    max_vocab_size: int = 2000         # reduced to 2k to fight data-sparsity
    sequence_length: int = 15            # back to 15 as per advanced proposal
    stride: int = 1
    oov_token: str = "<OOV>"
    skip_oov_targets: bool = True
    lowercase: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Neural architecture and optimizer settings."""

    embedding_dim: int = 128
    lstm_units: int = 256
    attention_units: int = 128
    dense_units: int = 512              # proposal uses Dense(512) -> Dense(256)
    
    # Specific dropouts from advanced proposal
    embedding_dropout: float = 0.10
    lstm_dropout: float = 0.30
    recurrent_dropout: float = 0.20
    dense_dropout_1: float = 0.40
    dense_dropout_2: float = 0.30
    
    learning_rate: float = 1e-3
    clipnorm: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop, split, and callback settings."""

    seed: int = 42
    epochs: int = 60
    batch_size: int = 256
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    early_stopping_patience: int = 20   # give more time — val_top5 still climbing
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    monitor_metric: str = "val_accuracy"  # monitor Top-1 to allow longer training
    monitor_mode: str = "max"                  # higher top5_accuracy is better


@dataclass(frozen=True)
class OutputConfig:
    """Output artifact paths."""

    output_dir: Path = PROJECT_ROOT / "outputs"
    best_model_filename: str = "best_model.keras"
    final_model_filename: str = "final_model.keras"
    history_plot_filename: str = "training_history.png"
    history_json_filename: str = "training_history.json"
    history_csv_filename: str = "training_history.csv"
    metrics_json_filename: str = "metrics.json"
    attention_heatmap_filename: str = "attention_heatmap.png"

    @property
    def best_model_path(self) -> Path:
        return self.output_dir / self.best_model_filename

    @property
    def final_model_path(self) -> Path:
        return self.output_dir / self.final_model_filename

    @property
    def history_plot_path(self) -> Path:
        return self.output_dir / self.history_plot_filename

    @property
    def history_json_path(self) -> Path:
        return self.output_dir / self.history_json_filename

    @property
    def history_csv_path(self) -> Path:
        return self.output_dir / self.history_csv_filename

    @property
    def metrics_json_path(self) -> Path:
        return self.output_dir / self.metrics_json_filename

    @property
    def attention_heatmap_path(self) -> Path:
        return self.output_dir / self.attention_heatmap_filename


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    def ensure_directories(self) -> None:
        """Create project directories used by the pipeline."""
        self.data.data_dir.mkdir(parents=True, exist_ok=True)
        self.outputs.output_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = AppConfig()
