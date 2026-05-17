"""
src/test.py
-----------
Loads the best saved model and runs a full evaluation suite:

  1. Quantitative metrics  — Top-1 accuracy, Top-5 accuracy, Perplexity
     on the held-out validation set (re-created from the same corpus + seed).
  2. Text generation tests — 5 required seed phrases, each producing ≥ 30 words.
  3. Step-by-step Top-5 breakdown — printed for Example 1 (assignment requirement).
  4. Pass / Fail summary table — shows which targets were met.

Usage:
    python src/test.py

The script is self-contained: it downloads the corpus if needed, rebuilds the
preprocessed data (to re-create the val split), loads the saved model, and
prints a full report.  No training is required.
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from data_loader  import download_corpus, load_corpus
from preprocessor import preprocess, SEQUENCE_LENGTH, load_tokenizer, clean_text, \
                          generate_sequences, vocabulary_size
from model        import BahdanauAttention
from generate     import generate_text, show_top5_table
from config       import DEFAULT_CONFIG

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT_DIR        = os.path.join(os.path.dirname(__file__), "..")
OUTPUTS_DIR     = os.path.join(ROOT_DIR, "outputs")
BEST_MODEL_PATH = os.path.join(OUTPUTS_DIR, "best_model.keras")
TOKENIZER_PATH  = os.path.join(ROOT_DIR, "data", "tokenizer.pkl")

# ──────────────────────────────────────────────────────────────────────────────
# Targets (from assignment) — all targets refer to TOP-5 accuracy
TARGET_TRAIN_TOP5 = 0.80   # Train Top-5 accuracy > 80 %
TARGET_VAL_TOP5   = 0.75   # Val/Test Top-5 accuracy > 75 %
TARGET_PERPLEXITY = 250.0  # Val perplexity < 250

SEED_PHRASES = [
    "It was a dark and stormy",
    "Holmes looked at the",
    "The game is",
    "Watson I need your",
    "She entered the room and",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_model_safe(path: str) -> tf.keras.Model:
    """Loads a Keras model that contains the custom BahdanauAttention layer."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Saved model not found at: {path}\n"
            "Run `python src/train.py` first to train and save the model."
        )
    model = tf.keras.models.load_model(
        path,
        custom_objects={"BahdanauAttention": BahdanauAttention},
    )
    print(f"[INFO] Model loaded from {path}")
    return model


def rebuild_val_split(random_seed: int = DEFAULT_CONFIG.training.seed):
    """
    Re-creates the validation split from the corpus using the same procedure
    as train.py (same random seed → same val indices every run).
    Returns X_val, y_val, tokenizer, vocab_size.
    """
    download_corpus()
    raw_text = load_corpus()

    # Use the saved tokenizer so vocab mapping is identical to training
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    max_vocab  = len(tokenizer.word_index)  # matches what training used

    clean  = clean_text(raw_text)
    X, y   = generate_sequences(clean, tokenizer, SEQUENCE_LENGTH)
    vocab_size = vocabulary_size(tokenizer, max_vocab)

    # Same shuffle + split as train.py
    rng     = np.random.default_rng(random_seed)
    indices = rng.permutation(len(X))
    X, y   = X[indices], y[indices]

    test_size  = int(len(X) * DEFAULT_CONFIG.training.test_fraction)
    val_size   = int(len(X) * DEFAULT_CONFIG.training.validation_fraction)
    train_end  = len(X) - val_size - test_size
    val_end    = len(X) - test_size

    X_train, y_train = X[:train_end],  y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]

    print(f"[INFO] Val split: {len(X_val):,} samples  |  "
          f"Train split: {len(X_train):,} samples")
    return X_train, y_train, X_val, y_val, tokenizer, vocab_size


def compute_metrics(model, X, y, split_name: str = "val", batch_size: int = 512):
    """
    Evaluates the model and returns a dict of metric name → value.
    Uses sparse_categorical_crossentropy to match integer y labels.
    """
    model.compile(
        loss      = "sparse_categorical_crossentropy",
        optimizer = tf.keras.optimizers.Adam(),
        metrics   = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
    )
    results = model.evaluate(X, y, batch_size=batch_size, verbose=0)
    metrics = {
        "loss": results[0],
        "accuracy": results[1] if len(results) > 1 else 0,
        "top5_acc": results[2] if len(results) > 2 else 0,
    }
    metrics["perplexity"] = float(np.exp(np.clip(metrics["loss"], None, 50)))

    print(f"\n  [{split_name}]  "
          f"Loss={metrics['loss']:.4f}  "
          f"Top-1 Acc={metrics['accuracy']*100:.2f}%  "
          f"Top-5 Acc={metrics['top5_acc']*100:.2f}%  "
          f"Perplexity={metrics['perplexity']:.2f}")
    return metrics


def print_pass_fail(label: str, value: float, target: float,
                    higher_is_better: bool = True) -> bool:
    """Prints a coloured PASS/FAIL line and returns whether it passed."""
    if higher_is_better:
        passed = value >= target
        rel    = f"{value*100:.2f}% (target ≥ {target*100:.0f}%)"
    else:
        passed = value <= target
        rel    = f"{value:.2f} (target ≤ {target:.0f})"

    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {label:<35} {rel}")
    return passed


# ──────────────────────────────────────────────────────────────────────────────
# Main test routine
# ──────────────────────────────────────────────────────────────────────────────

def run_tests():
    print("\n" + "═" * 65)
    print("  NEXT-WORD PREDICTION — TEST SUITE")
    print("═" * 65)

    # ── 1. Load model & data ──────────────────────────────────────────────────
    print("\n[SECTION 1] Loading model and data ...")
    model = load_model_safe(BEST_MODEL_PATH)
    X_train, y_train, X_val, y_val, tokenizer, vocab_size = rebuild_val_split()

    # ── 2. Quantitative metrics ───────────────────────────────────────────────
    print("\n[SECTION 2] Computing metrics ...")
    train_metrics = compute_metrics(model, X_train, y_train, split_name="train")
    val_metrics   = compute_metrics(model, X_val,   y_val,   split_name="val")

    # ── 3. Text generation ────────────────────────────────────────────────────
    print("\n[SECTION 3] Text generation (temperature=0.8, max_phrase_len=35) ...")
    all_results = []
    min_words_ok = True

    for idx, seed in enumerate(SEED_PHRASES, 1):
        output, w_and_p = generate_text(
            seed_text      = seed,
            max_phrase_len = 35,
            model          = model,
            tokenizer      = tokenizer,
            temperature    = 0.8,
        )
        generated_words = len(output.split()) - len(seed.split())
        all_results.append((seed, output, w_and_p, generated_words))

        status = "✅" if generated_words >= 30 else "❌"
        print(f"\n  {status} Example {idx}  (generated {generated_words} words)")
        print(f"     Seed : \"{seed}\"")
        print(f"     Out  : {output[:120]}{'...' if len(output) > 120 else ''}")

        if generated_words < 30:
            min_words_ok = False

    # ── 4. Top-5 breakdown for Example 1 (assignment requirement) ─────────────
    print("\n[SECTION 4] Step-by-step Top-5 breakdown — Example 1")
    seed, output, w_and_p, _ = all_results[0]
    print(f"  Seed      : \"{seed}\"")
    print(f"  Generated : {output}")
    show_top5_table(w_and_p)

    # ── 5. Pass / Fail summary ────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  PASS / FAIL SUMMARY")
    print("═" * 65)

    results = []
    results.append(print_pass_fail(
        "Training Top-5 Accuracy",
        train_metrics.get("top5_acc", 0),
        TARGET_TRAIN_TOP5,
        higher_is_better=True,
    ))
    results.append(print_pass_fail(
        "Val Top-5 Accuracy",
        val_metrics.get("top5_acc", 0),
        TARGET_VAL_TOP5,
        higher_is_better=True,
    ))
    results.append(print_pass_fail(
        "Val Perplexity",
        val_metrics["perplexity"],
        TARGET_PERPLEXITY,
        higher_is_better=False,
    ))
    results.append(
        True if min_words_ok else False
    )
    gen_status = "✅ PASS" if min_words_ok else "❌ FAIL"
    print(f"  {gen_status}  {'All examples ≥ 30 generated words':<35}")

    print("\n" + "─" * 65)
    passed = sum(results)
    total  = len(results)
    print(f"  Result: {passed}/{total} checks passed")

    if passed == total:
        print("  🎉 All assignment targets met!")
    else:
        print("  ⚠️  Some targets not met — check training or hyperparameters.")
    print("═" * 65 + "\n")

    return {
        "train_metrics" : train_metrics,
        "val_metrics"   : val_metrics,
        "all_results"   : all_results,
        "passed"        : passed,
        "total"         : total,
    }


if __name__ == "__main__":
    run_tests()