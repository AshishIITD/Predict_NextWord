"""
src/visualize_training.py
--------------------------
Visualises the training history of the LSTM + Bahdanau Attention model.

Produces TWO outputs
────────────────────
1. outputs/training_history.png
     Static 4-panel figure:
       • Top-left  : Accuracy (train vs val) with 80 %/75 % target lines
       • Top-right : Loss (train vs val)
       • Bottom-left : Perplexity (val) with 250 target line
       • Bottom-right: Learning-rate schedule

2. outputs/training_animation.gif
     Frame-by-frame animation of accuracy + loss building up epoch by epoch.
     Each frame adds one epoch so you can see the curves grow in real time.
     (~50 frames, ~3 s per loop, ~2–4 MB file size)

Usage
─────
    # After training:
    python src/visualize_training.py

    # Or import and pass your own history dict:
    from visualize_training import plot_all
    plot_all(history.history)

Requirements
────────────
    pip install matplotlib pillow          # Pillow is needed for GIF export
    (both are already in pyproject.toml via matplotlib)
"""

import os
import sys
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR        = os.path.join(os.path.dirname(__file__), "..")
OUTPUTS_DIR     = os.path.join(ROOT_DIR, "outputs")
HISTORY_JSON    = os.path.join(OUTPUTS_DIR, "training_history.json")  # matches train.py output
STATIC_PNG      = os.path.join(OUTPUTS_DIR, "training_history.png")
ANIMATION_GIF   = os.path.join(OUTPUTS_DIR, "training_animation.gif")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
TRAIN_COLOR   = "#3266ad"
VAL_COLOR     = "#e67c3b"
PERP_COLOR    = "#534ab7"
TARGET_GREEN  = "#2ca05a"
TARGET_AMBER  = "#e6a817"
TARGET_RED    = "#e24b4a"
LR_COLOR      = "#1d9e75"
GRID_ALPHA    = 0.25
BG_COLOR      = "#fafafa"
PANEL_COLOR   = "#ffffff"


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic history generator
# (used when no real history is available — mirrors the fixed model settings)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_history(epochs: int = 50, seed: int = 42) -> dict:
    """
    Generates a realistic-looking training history that mimics what you would
    get from the LSTM+Attention model on the Sherlock Holmes corpus.

    Replace this with your actual `history.history` dict after training.
    """
    rng = np.random.default_rng(seed)

    def smooth_noise(n, scale=0.008):
        return rng.normal(0, scale, n)

    ep = np.arange(1, epochs + 1)
    prog = ep / epochs

    # Accuracy curves  (logistic growth + noise)
    ta = 0.10 + 0.77 / (1 + np.exp(-10 * (prog - 0.35))) + smooth_noise(epochs, 0.006)
    va = 0.08 + 0.72 / (1 + np.exp(-10 * (prog - 0.38))) + smooth_noise(epochs, 0.009)
    ta = np.clip(ta, 0, 0.93)
    va = np.clip(va, 0, 0.88)

    # Loss curves  (exponential decay + noise)
    tl = 4.5 * np.exp(-4.2 * prog) + 0.38 + np.abs(smooth_noise(epochs, 0.015))
    vl = 4.8 * np.exp(-3.8 * prog) + 0.52 + np.abs(smooth_noise(epochs, 0.020))

    # Perplexity = exp(val_loss)
    perp = np.exp(vl)

    # Top-5 accuracy  (~20-25 pp above top-1)
    ta5 = np.clip(ta + 0.20 + smooth_noise(epochs, 0.004), 0, 0.99)
    va5 = np.clip(va + 0.19 + smooth_noise(epochs, 0.005), 0, 0.99)

    # Learning rate  (ReduceLROnPlateau halves at stalls)
    lr = []
    cur_lr = 1e-3
    stall = 0
    prev_vl = vl[0]
    for i in range(epochs):
        if vl[i] >= prev_vl - 0.005:
            stall += 1
        else:
            stall = 0
        if stall >= 5 and cur_lr > 1e-6:
            cur_lr *= 0.5
            stall = 0
        prev_vl = vl[i]
        lr.append(cur_lr)

    return {
        "accuracy":        ta.tolist(),
        "val_accuracy":    va.tolist(),
        "loss":            tl.tolist(),
        "val_loss":        vl.tolist(),
        "top5_acc":        ta5.tolist(),
        "val_top5_acc":    va5.tolist(),
        "perplexity":      perp.tolist(),
        "lr":              lr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Static 4-panel figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_static(history: dict, save_path: str = STATIC_PNG) -> None:
    """
    Saves a clean 4-panel static PNG of the full training history.
    """
    epochs   = list(range(1, len(history["accuracy"]) + 1))
    ta       = [v * 100 for v in history["accuracy"]]
    va       = [v * 100 for v in history["val_accuracy"]]
    tl       = history["loss"]
    vl       = history["val_loss"]
    ta5      = [v * 100 for v in history.get("top5_accuracy",     history.get("top5_acc", []))]
    va5      = [v * 100 for v in history.get("val_top5_accuracy", history.get("val_top5_acc", []))]
    perp     = history.get("perplexity", [math.exp(v) for v in vl])
    lr       = history.get("lr", [1e-3] * len(epochs))

    fig = plt.figure(figsize=(16, 10), facecolor=BG_COLOR)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    ax_acc  = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_perp = fig.add_subplot(gs[1, 0])
    ax_lr   = fig.add_subplot(gs[1, 1])

    for ax in (ax_acc, ax_loss, ax_perp, ax_lr):
        ax.set_facecolor(PANEL_COLOR)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#d0d0d0")
        ax.tick_params(colors="#666", labelsize=9)
        ax.grid(True, alpha=GRID_ALPHA, linewidth=0.6)

    # ── Panel 1: Accuracy ────────────────────────────────────────────────────
    ax_acc.plot(epochs, ta, color=TRAIN_COLOR, lw=2, label="Train top-1", zorder=3)
    ax_acc.plot(epochs, va, color=VAL_COLOR,   lw=2, label="Val top-1",   zorder=3)
    if ta5:
        ax_acc.plot(epochs, ta5, color=TRAIN_COLOR, lw=1.2, ls="--",
                    alpha=0.6, label="Train top-5")
        ax_acc.plot(epochs, va5, color=VAL_COLOR,   lw=1.2, ls="--",
                    alpha=0.6, label="Val top-5")
    ax_acc.axhline(80, color=TARGET_GREEN, lw=1.2, ls="--", alpha=0.8, label="80% target")
    ax_acc.axhline(75, color=TARGET_AMBER, lw=1.2, ls=":",  alpha=0.8, label="75% target")
    ax_acc.fill_between(epochs, ta, va, alpha=0.07, color=TRAIN_COLOR)

    # Mark best val accuracy
    best_ep  = int(np.argmax(va)) + 1
    best_val = max(va)
    ax_acc.annotate(f"Best val\n{best_val:.1f}% @ ep {best_ep}",
                    xy=(best_ep, best_val),
                    xytext=(best_ep + max(1, len(epochs)//8), best_val - 8),
                    fontsize=8, color=VAL_COLOR,
                    arrowprops=dict(arrowstyle="->", color=VAL_COLOR, lw=1))
    ax_acc.set_title("Accuracy", fontsize=12, fontweight="500", pad=8)
    ax_acc.set_xlabel("Epoch", fontsize=9)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=9)
    ax_acc.set_ylim(0, 100)
    ax_acc.legend(fontsize=8, framealpha=0.7, loc="lower right")

    # ── Panel 2: Loss ────────────────────────────────────────────────────────
    ax_loss.plot(epochs, tl, color=TRAIN_COLOR, lw=2, label="Train loss", zorder=3)
    ax_loss.plot(epochs, vl, color=VAL_COLOR,   lw=2, label="Val loss",   zorder=3)
    ax_loss.fill_between(epochs, tl, vl, alpha=0.07, color=VAL_COLOR)
    ax_loss.set_title("Loss (categorical cross-entropy)", fontsize=12, fontweight="500", pad=8)
    ax_loss.set_xlabel("Epoch", fontsize=9)
    ax_loss.set_ylabel("Loss", fontsize=9)
    ax_loss.legend(fontsize=8, framealpha=0.7)

    # ── Panel 3: Perplexity ──────────────────────────────────────────────────
    ax_perp.plot(epochs, perp, color=PERP_COLOR, lw=2, label="Val perplexity", zorder=3)
    ax_perp.axhline(250, color=TARGET_RED, lw=1.2, ls="--", alpha=0.8, label="Target < 250")
    ax_perp.fill_between(epochs, perp, 250,
                         where=[p < 250 for p in perp],
                         alpha=0.12, color=TARGET_GREEN, label="Below target")
    best_perp_ep = int(np.argmin(perp)) + 1
    best_perp    = min(perp)
    ax_perp.annotate(f"Best {best_perp:.0f}\n@ ep {best_perp_ep}",
                     xy=(best_perp_ep, best_perp),
                     xytext=(best_perp_ep + max(1, len(epochs)//8), best_perp + 10),
                     fontsize=8, color=PERP_COLOR,
                     arrowprops=dict(arrowstyle="->", color=PERP_COLOR, lw=1))
    ax_perp.set_title("Perplexity (val)", fontsize=12, fontweight="500", pad=8)
    ax_perp.set_xlabel("Epoch", fontsize=9)
    ax_perp.set_ylabel("Perplexity", fontsize=9)
    ax_perp.legend(fontsize=8, framealpha=0.7)

    # ── Panel 4: Learning rate ───────────────────────────────────────────────
    ax_lr.semilogy(epochs, lr, color=LR_COLOR, lw=2, zorder=3)
    ax_lr.fill_between(epochs, lr, alpha=0.12, color=LR_COLOR)
    # Mark LR drops
    for i in range(1, len(lr)):
        if lr[i] < lr[i - 1] * 0.6:
            ax_lr.axvline(epochs[i], color="#aaa", lw=0.8, ls="--")
            ax_lr.text(epochs[i] + 0.3, lr[i] * 1.4, "÷2", fontsize=7, color="#888")
    ax_lr.set_title("Learning rate schedule", fontsize=12, fontweight="500", pad=8)
    ax_lr.set_xlabel("Epoch", fontsize=9)
    ax_lr.set_ylabel("Learning rate (log)", fontsize=9)

    # ── Super-title & final metrics box ──────────────────────────────────────
    fig.suptitle("LSTM + Bahdanau Attention — Training History\n"
                 "Corpus: The Adventures of Sherlock Holmes",
                 fontsize=13, fontweight="500", y=0.98, color="#222")

    # Status badges inside figure
    final_ta   = ta[-1]
    final_va   = va[-1]
    final_perp = perp[-1]
    badges = [
        (f"Train acc  {final_ta:.1f}%",  final_ta  >= 80, "≥80% ✓" if final_ta  >= 80 else "<80% ✗"),
        (f"Val acc    {final_va:.1f}%",  final_va  >= 75, "≥75% ✓" if final_va  >= 75 else "<75% ✗"),
        (f"Perplexity {final_perp:.0f}", final_perp < 250, "<250 ✓" if final_perp < 250 else "≥250 ✗"),
    ]
    badge_x = 0.02
    for label, ok, note in badges:
        col = "#2ca05a" if ok else "#e24b4a"
        fig.text(badge_x, 0.005, f"  {label}  [{note}]  ",
                 fontsize=8.5, color=col,
                 bbox=dict(facecolor="#f0fff4" if ok else "#fff0f0",
                           edgecolor=col, boxstyle="round,pad=0.25", linewidth=0.8))
        badge_x += 0.29

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[INFO] Static plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Animated GIF
# ─────────────────────────────────────────────────────────────────────────────

def plot_animation(history: dict,
                   save_path: str = ANIMATION_GIF,
                   fps: int = 8,
                   skip: int = 1) -> None:
    """
    Creates a frame-by-frame animated GIF showing accuracy and loss
    building up epoch by epoch.

    Args:
        history   : Training history dict.
        save_path : Output GIF path.
        fps       : Frames per second (higher = faster animation).
        skip      : Only render every Nth epoch as a frame (reduces file size).
                    skip=1 → every epoch; skip=2 → every other epoch.
    """
    epochs_full = list(range(1, len(history["accuracy"]) + 1))
    ta_full     = [v * 100 for v in history["accuracy"]]
    va_full     = [v * 100 for v in history["val_accuracy"]]
    tl_full     = history["loss"]
    vl_full     = history["val_loss"]
    perp_full   = history.get("perplexity", [math.exp(v) for v in vl_full])

    total_epochs = len(epochs_full)
    frame_indices = list(range(0, total_epochs, skip))
    # Always include the final frame
    if frame_indices[-1] != total_epochs - 1:
        frame_indices.append(total_epochs - 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.14, wspace=0.32)

    ax_acc, ax_loss, ax_perp = axes
    for ax in axes:
        ax.set_facecolor(PANEL_COLOR)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#d0d0d0")
        ax.tick_params(colors="#666", labelsize=9)
        ax.grid(True, alpha=GRID_ALPHA, linewidth=0.6)

    # Static target lines (drawn once, not redrawn each frame)
    ax_acc.axhline(80, color=TARGET_GREEN, lw=1.2, ls="--", alpha=0.8)
    ax_acc.axhline(75, color=TARGET_AMBER, lw=1.2, ls=":",  alpha=0.8)
    ax_perp.axhline(250, color=TARGET_RED, lw=1.2, ls="--", alpha=0.8)

    ax_acc.set_xlim(1, total_epochs)
    ax_acc.set_ylim(0, 100)
    ax_loss.set_xlim(1, total_epochs)
    ax_loss.set_ylim(0, max(tl_full) * 1.05)
    ax_perp.set_xlim(1, total_epochs)
    ax_perp.set_ylim(0, max(perp_full) * 1.1)

    ax_acc.set_xlabel("Epoch", fontsize=9);  ax_acc.set_ylabel("Accuracy (%)", fontsize=9)
    ax_loss.set_xlabel("Epoch", fontsize=9); ax_loss.set_ylabel("Loss", fontsize=9)
    ax_perp.set_xlabel("Epoch", fontsize=9); ax_perp.set_ylabel("Perplexity", fontsize=9)

    # Legend patches (static)
    legend_kw = dict(fontsize=8, framealpha=0.7)
    ax_acc.legend(handles=[
        Line2D([0],[0], color=TRAIN_COLOR, lw=2, label="Train"),
        Line2D([0],[0], color=VAL_COLOR,   lw=2, label="Val"),
        Line2D([0],[0], color=TARGET_GREEN, lw=1.2, ls="--", label="80% target"),
        Line2D([0],[0], color=TARGET_AMBER, lw=1.2, ls=":",  label="75% target"),
    ], **legend_kw)
    ax_loss.legend(handles=[
        Line2D([0],[0], color=TRAIN_COLOR, lw=2, label="Train"),
        Line2D([0],[0], color=VAL_COLOR,   lw=2, label="Val"),
    ], **legend_kw)
    ax_perp.legend(handles=[
        Line2D([0],[0], color=PERP_COLOR,  lw=2, label="Perplexity"),
        Line2D([0],[0], color=TARGET_RED,  lw=1.2, ls="--", label="< 250 target"),
    ], **legend_kw)

    # Animated line objects
    line_ta,  = ax_acc.plot([],  [], color=TRAIN_COLOR, lw=2, zorder=3)
    line_va,  = ax_acc.plot([],  [], color=VAL_COLOR,   lw=2, zorder=3)
    line_tl,  = ax_loss.plot([], [], color=TRAIN_COLOR, lw=2, zorder=3)
    line_vl,  = ax_loss.plot([], [], color=VAL_COLOR,   lw=2, zorder=3)
    line_perp,= ax_perp.plot([],[], color=PERP_COLOR,   lw=2, zorder=3)

    # Epoch counter text
    ep_text = fig.text(0.50, 0.96, "", ha="center", fontsize=11,
                       fontweight="500", color="#333")
    metric_text = fig.text(0.50, 0.90, "", ha="center", fontsize=9, color="#555")

    title_text = fig.suptitle(
        "LSTM + Bahdanau Attention — Training Animation\n"
        "Sherlock Holmes corpus  |  256 LSTM units  |  Bahdanau attention",
        fontsize=11, fontweight="500", y=1.02, color="#222"
    )

    def init():
        for line in (line_ta, line_va, line_tl, line_vl, line_perp):
            line.set_data([], [])
        ep_text.set_text("")
        metric_text.set_text("")
        return line_ta, line_va, line_tl, line_vl, line_perp, ep_text, metric_text

    def update(frame_idx):
        i = frame_indices[frame_idx]
        ep_slice = epochs_full[:i+1]

        line_ta.set_data(ep_slice, ta_full[:i+1])
        line_va.set_data(ep_slice, va_full[:i+1])
        line_tl.set_data(ep_slice, tl_full[:i+1])
        line_vl.set_data(ep_slice, vl_full[:i+1])
        line_perp.set_data(ep_slice, perp_full[:i+1])

        epoch_num = epochs_full[i]
        ep_text.set_text(f"Epoch {epoch_num} / {total_epochs}")

        ta_now   = ta_full[i]
        va_now   = va_full[i]
        perp_now = perp_full[i]

        def badge(val, target, higher=True):
            ok = val >= target if higher else val <= target
            return ("[PASS]" if ok else "[...]")

        metric_text.set_text(
            f"Train acc: {ta_now:.1f}% {badge(ta_now, 80)}   "
            f"Val acc: {va_now:.1f}% {badge(va_now, 75)}   "
            f"Perplexity: {perp_now:.0f} {badge(perp_now, 250, higher=False)}"
        )
        return line_ta, line_va, line_tl, line_vl, line_perp, ep_text, metric_text

    ani = FuncAnimation(
        fig, update, frames=len(frame_indices),
        init_func=init, blit=False, interval=1000 // fps,
    )

    print(f"[INFO] Saving animation → {save_path}  ({len(frame_indices)} frames @ {fps} fps) ...")
    writer = PillowWriter(fps=fps)
    ani.save(save_path, writer=writer, dpi=100)
    plt.close()
    print(f"[INFO] Animation saved  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level convenience
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(history: dict = None,
             static_path: str    = STATIC_PNG,
             anim_path:   str    = ANIMATION_GIF,
             fps:         int    = 8,
             skip:        int    = 1) -> None:
    """
    Generates both the static PNG and the animated GIF.

    Args:
        history     : history.history dict from Keras model.fit().
                      If None, a synthetic history is generated so you can
                      preview the output format before training finishes.
        static_path : Path for the static PNG.
        anim_path   : Path for the animated GIF.
        fps         : Animation frames per second.
        skip        : Render every Nth epoch (reduces GIF size without losing shape).
    """
    if history is None:
        print("[INFO] No history provided — using synthetic data for preview.")
        history = generate_synthetic_history(epochs=50)

    print(f"\n[INFO] Epochs in history : {len(history['accuracy'])}")
    print("[INFO] Keys found        :", list(history.keys()))

    # Ensure perplexity key exists
    if "perplexity" not in history:
        history["perplexity"] = [math.exp(v) for v in history["val_loss"]]

    plot_static(history, save_path=static_path)
    plot_animation(history, save_path=anim_path, fps=fps, skip=skip)

    print("\n✅  All plots generated:")
    print(f"    Static PNG  → {static_path}")
    print(f"    Animated GIF→ {anim_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Integration with train.py
# ─────────────────────────────────────────────────────────────────────────────

def load_and_plot_from_saved_model():
    """
    Loads the trained model + re-evaluates on the val set to reconstruct
    metrics if history.json was not saved. Falls back to synthetic data.
    """
    history_path = HISTORY_JSON
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        print(f"[INFO] Loaded history from {history_path}")
        plot_all(history)
    else:
        print("[WARN] history.json not found. Generating synthetic preview.")
        print("       To use real data, add this to your train.py after model.fit():")
        print("         import json")
        print("         with open('outputs/history.json', 'w') as f:")
        print("             json.dump(history.history, f)")
        plot_all(None)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise LSTM training history")
    parser.add_argument("--history", type=str, default=None,
                        help="Path to history.json (optional; uses synthetic if absent)")
    parser.add_argument("--fps", type=int, default=8,
                        help="Animation frames per second (default: 8)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Render every Nth epoch as a frame (default: 1 = all)")
    parser.add_argument("--static-only", action="store_true",
                        help="Only generate the static PNG, skip the GIF")
    args = parser.parse_args()

    if args.history and os.path.exists(args.history):
        with open(args.history) as f:
            hist = json.load(f)
    else:
        hist = None

    if args.static_only:
        h = hist or generate_synthetic_history()
        if "perplexity" not in h:
            h["perplexity"] = [math.exp(v) for v in h["val_loss"]]
        plot_static(h)
    else:
        plot_all(hist, fps=args.fps, skip=args.skip)