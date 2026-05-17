"""
Model builder using pre-trained GloVe embeddings.
Isolated from the main model.py file to preserve the original implementation.
"""

from __future__ import annotations

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

try:
    from .config import DEFAULT_CONFIG
    from .model import BahdanauAttention
except ImportError:  # pragma: no cover
    from config import DEFAULT_CONFIG
    from model import BahdanauAttention

LOGGER = logging.getLogger(__name__)

def load_glove_embeddings(tokenizer, glove_path, vocab_size, embed_dim=100):
    """Load GloVe embeddings and map them to our vocabulary."""
    LOGGER.info(f"Loading GloVe embeddings from {glove_path}...")
    embeddings = np.zeros((vocab_size, embed_dim))
    
    try:
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in tokenizer.word_index:
                    idx = tokenizer.word_index[word]
                    if idx < vocab_size:
                        embeddings[idx] = np.array(values[1:], dtype="float32")
        LOGGER.info("GloVe embeddings loaded successfully.")
    except FileNotFoundError:
        LOGGER.warning(f"GloVe file not found at {glove_path}. Using random embeddings!")
        
    return embeddings

def build_glove_model(vocab_size, seq_len, embedding_matrix, embed_dim=100, learning_rate=0.001, clipnorm=5.0):
    """Build the model using pre-trained GloVe embeddings."""
    inputs = layers.Input(shape=(seq_len,), name="input_tokens")
    
    # Embedding layer with pre-trained weights
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        weights=[embedding_matrix],
        trainable=True,          # fine-tune during training
        mask_zero=False,         # Match current model
        name="embedding",
    )(inputs)
    
    # Architecture copied from model.py to maintain consistency
    lstm_units = 256
    attention_units = 128
    dense_units = 512
    lstm_dropout = 0.2
    recurrent_dropout = 0.2
    embedding_dropout = 0.2
    dense_dropout_1 = 0.4
    dense_dropout_2 = 0.3
    
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

    model = Model(inputs=inputs, outputs=outputs, name="lstm_attention_next_word_glove")
    
    # Compile
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
