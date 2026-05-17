"""
LSTM language model with Bahdanau-style additive attention.

Architecture:
    Embedding(128)
    -> Bidirectional LSTM(256, return_sequences=True)
    -> Additive attention over all hidden states
    -> Dense(256, relu)
    -> Dropout(0.3)
    -> Softmax over vocabulary
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

try:
    from .config import DEFAULT_CONFIG, ModelConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG, ModelConfig


LOGGER = logging.getLogger(__name__)
MODEL_CONFIG = DEFAULT_CONFIG.model


@tf.keras.utils.register_keras_serializable(package="next_word_prediction")
class BahdanauAttention(layers.Layer):
    """
    Additive attention over a sequence of BiLSTM hidden states.

    For hidden states H = [h_1, ..., h_T], the layer computes

        q = mean(H)
        e_t = v^T tanh(W_1 h_t + W_2 q)
        alpha_t = softmax(e_t)
        c = sum_t alpha_t h_t

    where alpha is interpretable as the attention weight assigned to each input
    token and c is the context vector passed to the classifier. The mean hidden
    state acts as an encoder-side query because next-word prediction has no
    separate decoder state.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.supports_masking = True
        self.W1 = layers.Dense(self.units)
        self.W2 = layers.Dense(self.units)
        self.V = layers.Dense(1)

    def build(self, input_shape):
        """Build child Dense layers for reliable Keras 2/3 deserialization."""
        input_shape = tf.TensorShape(input_shape).as_list()
        self.W1.build(input_shape)
        self.W2.build([input_shape[0], 1, input_shape[-1]])
        self.V.build([input_shape[0], input_shape[1], self.units])
        super().build(input_shape)

    def call(self, hidden_states: tf.Tensor, mask: tf.Tensor | None = None):
        """Return ``(context_vector, attention_weights)``."""
        if mask is not None:
            mask_float = tf.cast(tf.expand_dims(mask, axis=-1), hidden_states.dtype)
            denominator = tf.maximum(tf.reduce_sum(mask_float, axis=1, keepdims=True), 1.0)
            query = tf.reduce_sum(hidden_states * mask_float, axis=1, keepdims=True) / denominator
        else:
            query = tf.reduce_mean(hidden_states, axis=1, keepdims=True)

        scores = self.V(tf.nn.tanh(self.W1(hidden_states) + self.W2(query)))

        if mask is not None:
            mask = tf.cast(mask, dtype=scores.dtype)
            scores += (1.0 - tf.expand_dims(mask, axis=-1)) * tf.cast(-1e9, scores.dtype)

        attention_weights = tf.nn.softmax(scores, axis=1)
        context_vector = tf.reduce_sum(attention_weights * hidden_states, axis=1)
        return context_vector, attention_weights

    def compute_mask(self, inputs, mask=None):  # noqa: D401 - Keras API hook
        """Do not propagate masks beyond the pooled context vector."""
        return (None, None)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_model(
    vocab_size: int,
    seq_len: int = DEFAULT_CONFIG.preprocessing.sequence_length,
    embed_dim: int = MODEL_CONFIG.embedding_dim,
    lstm_units: int = MODEL_CONFIG.lstm_units,
    attention_units: int = MODEL_CONFIG.attention_units,
    dense_units: int = MODEL_CONFIG.dense_units,
    embedding_dropout: float = MODEL_CONFIG.embedding_dropout,
    lstm_dropout: float = MODEL_CONFIG.lstm_dropout,
    recurrent_dropout: float = MODEL_CONFIG.recurrent_dropout,
    dense_dropout_1: float = MODEL_CONFIG.dense_dropout_1,
    dense_dropout_2: float = MODEL_CONFIG.dense_dropout_2,
    learning_rate: float = MODEL_CONFIG.learning_rate,
    clipnorm: float = MODEL_CONFIG.clipnorm,
) -> Model:
    """
    Build and compile the next-word prediction model with advanced features.
    
    Architecture (from proposal):
        Embedding(vocab_size, 128) -> Dropout(0.10)
        BiLSTM(256) -> LayerNorm
        LSTM(256) -> LayerNorm (returns state)
        BahdanauAttention(128)
        Concatenate[context, last_h_T]
        Dropout(0.40) -> Dense(512, GELU) -> Dropout(0.30) -> Dense(256, GELU)
        Dense(vocab_size, Softmax)
    """
    if vocab_size <= 2:
        raise ValueError("vocab_size must include at least padding, OOV, and one word")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    inputs = layers.Input(shape=(seq_len,), name="context_tokens")
    
    # Embedding(10001, 128) -> Dropout(0.10)
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=False,  # Proposal says removed mask_zero
        name="embedding",
    )(inputs)
    x = layers.Dropout(embedding_dropout, name="embedding_dropout")(x)

    # BiLSTM(256) -> LayerNorm
    lstm_1 = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=lstm_dropout,
            recurrent_dropout=recurrent_dropout,
            name="lstm_1",
        ),
        name="bidirectional_lstm_1",
    )(x)
    lstm_1 = layers.LayerNormalization(name="layernorm_1")(lstm_1)

    # LSTM(256) -> LayerNorm (returns sequences AND state)
    lstm_2_layer = layers.LSTM(
        lstm_units,
        return_sequences=True,
        return_state=True,
        dropout=lstm_dropout,
        recurrent_dropout=recurrent_dropout,
        name="lstm_2",
    )
    lstm_2_output, last_h, last_c = lstm_2_layer(lstm_1)
    lstm_2_output = layers.LayerNormalization(name="layernorm_2")(lstm_2_output)

    # BahdanauAttention(128)
    context_vector, attention_weights = BahdanauAttention(
        units=attention_units,
        name="bahdanau_attention",
    )(lstm_2_output)

    # Concatenate[context, last_h_T]
    concat = layers.Concatenate(name="residual_concat")([context_vector, last_h])

    # Dropout(0.40) -> Dense(512, GELU) -> Dropout(0.30) -> Dense(256, GELU)
    x = layers.Dropout(dense_dropout_1, name="dropout_1")(concat)
    x = layers.Dense(dense_units, activation="gelu", name="dense_512")(x)
    x = layers.Dropout(dense_dropout_2, name="dropout_2")(x)
    x = layers.Dense(256, activation="gelu", name="dense_256")(x)

    # Dense(vocab_size, Softmax)
    outputs = layers.Dense(vocab_size, activation="softmax", name="next_word")(x)

    model = Model(inputs=inputs, outputs=outputs, name="lstm_attention_next_word")
    
    # Compile (removed label smoothing as it is not supported in this TF version)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5,
                name="top5_accuracy",
            ),
        ],
    )
    return model


def build_attention_extractor(model: Model) -> Model:
    """Create a model that returns attention weights for a trained classifier."""
    attention_layer = model.get_layer("bahdanau_attention")
    attention_output = attention_layer.output
    if isinstance(attention_output, (list, tuple)):
        weights = tf.squeeze(attention_output[1], axis=-1)
    else:
        raise ValueError("The attention layer does not expose attention weights.")
    return Model(inputs=model.input, outputs=weights, name="attention_extractor")


def load_trained_model(path: str | Path = DEFAULT_CONFIG.outputs.best_model_path) -> Model:
    """Load a saved Keras model with the custom attention layer registered."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Run training first.")
    try:
        return tf.keras.models.load_model(
            path,
            custom_objects={"BahdanauAttention": BahdanauAttention},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model artifact at {path}. This usually means the file was "
            "saved with a different Keras/TensorFlow major version than the active "
            "environment. Re-run `uv run python -m src.train` to regenerate compatible "
            "model artifacts."
        ) from exc


def print_model_summary(model: Model) -> None:
    """Print model architecture and trainable parameter count."""
    model.summary(line_length=100)
    trainable_params = int(
        np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    )
    LOGGER.info("Total parameters: %s", f"{model.count_params():,}")
    LOGGER.info("Trainable parameters: %s", f"{trainable_params:,}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    demo_model = build_model(vocab_size=10_001, seq_len=15)
    print_model_summary(demo_model)
    dummy_input = np.random.randint(1, 10_000, size=(4, 15), dtype=np.int32)
    output = demo_model(dummy_input)
    print(f"Output shape: {output.shape}")
