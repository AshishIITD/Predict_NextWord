"""
src/visualize_test.py
---------------------
Visualises the TEST / EVALUATION results of the trained LSTM + Bahdanau
Attention model.

Produces THREE outputs
──────────────────────
1. outputs/test_results.png
     Static 6-panel figure:
       • Panel 1 : Train vs Val vs Test accuracy bar chart (Top-1 & Top-5)
       • Panel 2 : Confusion heat-map of top-10 predicted vs true words
       • Panel 3 : Perplexity gauge (needle chart) vs target
       • Panel 4 : Per-seed-phrase word probability strip chart
       • Panel 5 : Attention weight heatmap (context → next word)
       • Panel 6 : Pass/Fail target summary table

2. outputs/test_animation.gif
     Animated reveal of each evaluation panel, one by one, with a
     live metric counter ticking up.

3. outputs/test_report.png
     Clean single-page "report card" with all key numbers.

Usage
─────
    python src/visualize_test.py

    # With real model results:
    from visualize_test import plot_all_test
    plot_all_test(test_metrics, generation_results, attention_weights)

Requirements
────────────
    pip install matplotlib pillow numpy
"""

import os
import sys
import math
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, Wedge, Arc
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(__file__))

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR       = os.path.join(os.path.dirname(__file__), "..")
OUTPUTS_DIR    = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

STATIC_PNG     = os.path.join(OUTPUTS_DIR, "test_results.png")
ANIMATION_GIF  = os.path.join(OUTPUTS_DIR, "test_animation.gif")
REPORT_PNG     = os.path.join(OUTPUTS_DIR, "test_report.png")

# ── palette ───────────────────────────────────────────────────────────────────
TRAIN_C   = "#3266ad"
VAL_C     = "#e67c3b"
TEST_C    = "#1d9e75"
PASS_C    = "#2ca05a"
FAIL_C    = "#e24b4a"
PERP_C    = "#534ab7"
WARN_C    = "#e6a817"
BG        = "#f8f8f6"
PANEL_BG  = "#ffffff"
GRID_A    = 0.22


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic test data generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_test_data(seed: int = 7) -> dict:
    """
    Produces realistic-looking test evaluation data.
    Replace with your real test_metrics dict from test.py after training.
    """
    rng = np.random.default_rng(seed)

    # ── scalar metrics ────────────────────────────────────────────────────────
    train_acc   = 0.843
    val_acc     = 0.791
    test_acc    = 0.776
    train_top5  = 0.961
    val_top5    = 0.942
    test_top5   = 0.931
    val_perp    = 187.4
    test_perp   = 203.8

    # ── per-seed generation results ───────────────────────────────────────────
    seeds = [
        "It was a dark and stormy",
        "Holmes looked at the",
        "The game is",
        "Watson I need your",
        "She entered the room and",
    ]
    seed_probs = []
    for _ in seeds:
        # Average probability of chosen word at each step
        probs = 0.18 + 0.22 * rng.random(35) + rng.normal(0, 0.04, 35)
        probs = np.clip(probs, 0.04, 0.72)
        seed_probs.append(probs.tolist())

    # ── top-10 word frequency confusion (true vs predicted) ───────────────────
    top_words = ["the", "and", "a", "of", "to", "i", "he", "was", "in", "that"]
    confusion = np.zeros((10, 10))
    for i in range(10):
        row = rng.dirichlet(np.ones(10) * 2.5)
        row[i] *= 5      # boost diagonal (correct predictions)
        row /= row.sum()
        confusion[i] = row * rng.integers(80, 200)

    # ── attention weights for one generation step ─────────────────────────────
    context_words  = ["it", "was", "a", "dark", "and", "stormy",
                      "night", "when", "holmes", "arrived", "at", "the", "door", "of", "the"]
    attn_steps     = ["night→when", "when→holmes", "holmes→arrived",
                      "arrived→at", "at→the", "the→door"]
    attn_weights   = []
    for _ in attn_steps:
        w = rng.dirichlet(np.ones(len(context_words)) * 0.6)
        attn_weights.append(w.tolist())

    # ── loss by epoch on test set (simulated) ─────────────────────────────────
    test_losses = (0.52 + 0.05 * rng.random(50)).tolist()

    return {
        "train_acc":   train_acc,
        "val_acc":     val_acc,
        "test_acc":    test_acc,
        "train_top5":  train_top5,
        "val_top5":    val_top5,
        "test_top5":   test_top5,
        "val_perp":    val_perp,
        "test_perp":   test_perp,
        "seeds":       seeds,
        "seed_probs":  seed_probs,
        "top_words":   top_words,
        "confusion":   confusion.tolist(),
        "context_words":  context_words,
        "attn_steps":     attn_steps,
        "attn_weights":   attn_weights,
        "test_losses":    test_losses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Individual panel drawing functions
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d0d0d0")
    ax.tick_params(colors="#555", labelsize=8.5)
    ax.grid(True, alpha=GRID_A, linewidth=0.6)


def draw_accuracy_bars(ax, data):
    """Panel 1: grouped bar chart — Train / Val / Test accuracy."""
    labels   = ["Top-1 Accuracy", "Top-5 Accuracy"]
    train_v  = [data["train_acc"] * 100, data["train_top5"] * 100]
    val_v    = [data["val_acc"]   * 100, data["val_top5"]   * 100]
    test_v   = [data["test_acc"]  * 100, data["test_top5"]  * 100]

    x   = np.arange(len(labels))
    w   = 0.24
    bar_kw = dict(edgecolor="white", linewidth=0.8, zorder=3)

    b1 = ax.bar(x - w, train_v, w, color=TRAIN_C, label="Train", **bar_kw)
    b2 = ax.bar(x,     val_v,   w, color=VAL_C,   label="Val",   **bar_kw)
    b3 = ax.bar(x + w, test_v,  w, color=TEST_C,  label="Test",  **bar_kw)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=7.5, color="#333")

    ax.axhline(80, color=PASS_C,  lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.axhline(75, color=WARN_C,  lw=1.2, ls=":",  alpha=0.7, zorder=2)
    ax.text(len(labels) - 0.05, 80.8, "80% target", fontsize=7,
            color=PASS_C, ha="right")
    ax.text(len(labels) - 0.05, 75.8, "75% target", fontsize=7,
            color=WARN_C, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_title("Accuracy — Train / Val / Test", fontsize=11,
                 fontweight="500", pad=8)
    ax.legend(fontsize=8, framealpha=0.7, loc="lower right")
    _style_ax(ax)
    ax.grid(False, axis="x")


def draw_confusion(ax, data):
    """Panel 2: word prediction confusion heatmap."""
    confusion = np.array(data["confusion"])
    words     = data["top_words"]

    # Normalise rows to percentages
    row_sums  = confusion.sum(axis=1, keepdims=True)
    norm      = confusion / np.maximum(row_sums, 1) * 100

    im = ax.imshow(norm, cmap="Blues", aspect="auto", vmin=0, vmax=60)
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(words, fontsize=8)
    ax.set_xlabel("Predicted word", fontsize=9)
    ax.set_ylabel("True word", fontsize=9)
    ax.set_title("Prediction confusion — top-10 words (%)", fontsize=11,
                 fontweight="500", pad=8)

    for i in range(len(words)):
        for j in range(len(words)):
            val = norm[i, j]
            col = "white" if val > 35 else "#333"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=6.5, color=col)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="% of true-word predictions")
    # Remove internal grid from heatmap
    ax.grid(False)
    ax.set_facecolor(PANEL_BG)
    ax.spines[:].set_color("#d0d0d0")
    ax.tick_params(colors="#555", labelsize=8)


def draw_perplexity_gauge(ax, data):
    """Panel 3: semicircle gauge for val and test perplexity."""
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Perplexity gauge", fontsize=11, fontweight="500", pad=8)

    TARGET = 250
    MAX_P  = 500

    # Background arc zones
    zone_colors = ["#eaf3de", "#faeeda", "#fcebeb"]   # green / amber / red
    zone_limits = [0, 0.4, 0.7, 1.0]

    for i, (c, s, e) in enumerate(zip(zone_colors,
                                       zone_limits[:-1], zone_limits[1:])):
        theta1 = 180 - s * 180
        theta2 = 180 - e * 180
        wedge  = Wedge((0, 0), 1.0, theta2, theta1,
                       width=0.3, facecolor=c, edgecolor="white", lw=1.5)
        ax.add_patch(wedge)

    # Zone labels
    for label, pos in [("Safe", 0.15), ("Caution", 0.53), ("Over target", 0.85)]:
        angle = math.radians(180 - pos * 180)
        r = 0.85
        ax.text(r * math.cos(angle), r * math.sin(angle), label,
                ha="center", va="center", fontsize=6.5, color="#555",
                rotation=-(pos * 180 - 90))

    def _needle(ax, value, max_val, color, label, r=0.95, lw=3):
        frac  = min(value / max_val, 1.0)
        angle = math.radians(180 - frac * 180)
        ax.annotate("",
                    xy=(r * math.cos(angle), r * math.sin(angle)),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=12))
        ax.text(r * math.cos(angle) * 1.1,
                r * math.sin(angle) * 1.1 + 0.04,
                f"{label}\n{value:.0f}", ha="center", va="bottom",
                fontsize=8, color=color, fontweight="500")

    _needle(ax, data["val_perp"],  MAX_P, VAL_C,  "Val",  r=0.88, lw=3)
    _needle(ax, data["test_perp"], MAX_P, TEST_C, "Test", r=0.70, lw=2.5)

    # Target tick
    frac  = TARGET / MAX_P
    angle = math.radians(180 - frac * 180)
    ax.plot([0.68 * math.cos(angle), 1.02 * math.cos(angle)],
            [0.68 * math.sin(angle), 1.02 * math.sin(angle)],
            color=FAIL_C, lw=2, zorder=5)
    ax.text(1.08 * math.cos(angle), 1.08 * math.sin(angle),
            "250\ntarget", ha="center", va="center",
            fontsize=7, color=FAIL_C)

    # Centre hub
    hub = plt.Circle((0, 0), 0.06, color="#444", zorder=6)
    ax.add_patch(hub)

    # Status
    both_ok = data["val_perp"] < TARGET and data["test_perp"] < TARGET
    ax.text(0, -0.22, "PASS" if both_ok else "FAIL",
            ha="center", fontsize=13, fontweight="500",
            color=PASS_C if both_ok else FAIL_C)
    ax.text(0, -0.35, f"Val {data['val_perp']:.0f}  |  Test {data['test_perp']:.0f}",
            ha="center", fontsize=8, color="#555")

    # Scale ticks
    for v in [0, 100, 200, 300, 400, 500]:
        f = v / MAX_P
        a = math.radians(180 - f * 180)
        ax.plot([1.0 * math.cos(a), 1.08 * math.cos(a)],
                [1.0 * math.sin(a), 1.08 * math.sin(a)],
                color="#aaa", lw=1)
        ax.text(1.18 * math.cos(a), 1.18 * math.sin(a), str(v),
                ha="center", va="center", fontsize=6.5, color="#888")


def draw_seed_probs(ax, data):
    """Panel 4: per-seed probability strip (violin-style summary)."""
    seeds  = [s.split()[-1] + "…" for s in data["seeds"]]
    probs  = data["seed_probs"]

    positions = list(range(len(seeds)))
    parts = ax.violinplot([np.array(p) * 100 for p in probs],
                          positions=positions,
                          showmedians=True, showextrema=True,
                          widths=0.6)
    for pc in parts["bodies"]:
        pc.set_facecolor(TEST_C)
        pc.set_alpha(0.4)
    parts["cmedians"].set_color(TEST_C)
    parts["cmins"].set_color("#aaa")
    parts["cmaxes"].set_color("#aaa")
    parts["cbars"].set_color("#aaa")

    # Overlay mean dots
    for i, p in enumerate(probs):
        ax.scatter(i, np.mean(p) * 100, color=TEST_C, s=40, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(seeds, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Chosen word probability (%)", fontsize=9)
    ax.set_title("Per-seed generation confidence", fontsize=11,
                 fontweight="500", pad=8)
    ax.set_ylim(0, 80)
    _style_ax(ax)
    ax.grid(False, axis="x")


def draw_attention_heatmap(ax, data):
    """Panel 5: attention weights heatmap across generation steps."""
    weights = np.array(data["attn_weights"])   # (steps, context_len)
    steps   = data["attn_steps"]
    ctx     = data["context_words"]

    im = ax.imshow(weights, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.25)
    ax.set_xticks(range(len(ctx)))
    ax.set_yticks(range(len(steps)))
    ax.set_xticklabels(ctx, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(steps, fontsize=8)
    ax.set_xlabel("Context word", fontsize=9)
    ax.set_ylabel("Prediction step", fontsize=9)
    ax.set_title("Bahdanau attention weights", fontsize=11,
                 fontweight="500", pad=8)

    for i in range(len(steps)):
        for j in range(len(ctx)):
            v = weights[i, j]
            col = "white" if v > 0.15 else "#333"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color=col)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention weight")
    ax.grid(False)
    ax.set_facecolor(PANEL_BG)
    ax.spines[:].set_color("#d0d0d0")
    ax.tick_params(colors="#555", labelsize=8)


def draw_pass_fail_table(ax, data):
    """Panel 6: pass/fail summary table."""
    ax.axis("off")
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Assignment target summary", fontsize=11,
                 fontweight="500", pad=8)

    rows = [
        ("Train Accuracy (Top-1)", f"{data['train_acc']*100:.1f}%",
         "> 80%", data["train_acc"] >= 0.80),
        ("Val Accuracy (Top-1)",   f"{data['val_acc']*100:.1f}%",
         "> 75%", data["val_acc"]   >= 0.75),
        ("Test Accuracy (Top-1)",  f"{data['test_acc']*100:.1f}%",
         "> 75%", data["test_acc"]  >= 0.75),
        ("Train Accuracy (Top-5)", f"{data['train_top5']*100:.1f}%",
         "> 90%", data["train_top5"]>= 0.90),
        ("Val Accuracy (Top-5)",   f"{data['val_top5']*100:.1f}%",
         "> 90%", data["val_top5"]  >= 0.90),
        ("Val Perplexity",         f"{data['val_perp']:.0f}",
         "< 250", data["val_perp"]  <  250),
        ("Test Perplexity",        f"{data['test_perp']:.0f}",
         "< 250", data["test_perp"] <  250),
    ]

    col_labels = ["Metric", "Achieved", "Target", "Status"]
    col_widths = [0.44, 0.18, 0.16, 0.14]
    x_starts   = [0.01, 0.45, 0.63, 0.81]
    y_start    = 0.88
    row_h      = 0.115

    # Header
    for col, cx in zip(col_labels, x_starts):
        ax.text(cx, y_start, col, transform=ax.transAxes,
                fontsize=8.5, fontweight="500", color="#333",
                va="top")

    ax.plot([0.01, 0.99], [y_start - 0.01, y_start - 0.01],
            color="#ccc", lw=0.8, transform=ax.transAxes, clip_on=False)

    for i, (metric, achieved, target, ok) in enumerate(rows):
        y = y_start - (i + 1) * row_h
        bg_col = "#f0fff4" if ok else "#fff5f5"
        rect = mpatches.FancyBboxPatch(
            (0.005, y - 0.01), 0.99, row_h - 0.01,
            boxstyle="round,pad=0.005",
            facecolor=bg_col, edgecolor="none",
            transform=ax.transAxes, zorder=0
        )
        ax.add_patch(rect)

        vals = [metric, achieved, target, "PASS" if ok else "FAIL"]
        cols = ["#333",   "#333",   "#777",
                PASS_C if ok else FAIL_C]
        wts  = ["normal", "500",    "normal",
                "500"]

        for val, cx, col, wt in zip(vals, x_starts, cols, wts):
            ax.text(cx, y + row_h * 0.45, val,
                    transform=ax.transAxes,
                    fontsize=8.5, color=col, fontweight=wt, va="center")

    # Summary footer
    passed = sum(1 for *_, ok in rows if ok)
    total  = len(rows)
    footer_col = PASS_C if passed == total else (WARN_C if passed >= total - 1 else FAIL_C)
    ax.text(0.5, 0.01,
            f"{passed}/{total} targets met",
            transform=ax.transAxes,
            ha="center", fontsize=10, fontweight="500", color=footer_col)


# ─────────────────────────────────────────────────────────────────────────────
# Static 6-panel figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_static(data: dict, save_path: str = STATIC_PNG) -> None:
    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.42, wspace=0.35,
                            left=0.06, right=0.97,
                            top=0.91, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    draw_accuracy_bars(ax1, data)
    draw_confusion(ax2, data)
    draw_perplexity_gauge(ax3, data)
    draw_seed_probs(ax4, data)
    draw_attention_heatmap(ax5, data)
    draw_pass_fail_table(ax6, data)

    fig.suptitle(
        "LSTM + Bahdanau Attention — Test Evaluation Report\n"
        "Corpus: The Adventures of Sherlock Holmes  |  Vocab: 10 001  |  Seq len: 15",
        fontsize=13, fontweight="500", y=0.975, color="#222"
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[INFO] Static test plot   → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Report card (single-page)
# ─────────────────────────────────────────────────────────────────────────────

def plot_report_card(data: dict, save_path: str = REPORT_PNG) -> None:
    fig = plt.figure(figsize=(10, 6), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_facecolor(BG)

    # Title banner
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.01, 0.88), 0.98, 0.10,
        boxstyle="round,pad=0.01",
        facecolor=TRAIN_C, edgecolor="none",
        transform=ax.transAxes))
    ax.text(0.5, 0.935, "LSTM + Bahdanau Attention — Test Report Card",
            ha="center", va="center", fontsize=14, fontweight="500",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.895, "Adventures of Sherlock Holmes  |  256 LSTM units  |  seq_len=15",
            ha="center", va="center", fontsize=9, color="#c8d8f0",
            transform=ax.transAxes)

    # Big metric cards (top row)
    big_metrics = [
        ("Train Acc", f"{data['train_acc']*100:.1f}%",  data["train_acc"] >= 0.80, ">80%"),
        ("Val Acc",   f"{data['val_acc']*100:.1f}%",    data["val_acc"]   >= 0.75, ">75%"),
        ("Test Acc",  f"{data['test_acc']*100:.1f}%",   data["test_acc"]  >= 0.75, ">75%"),
        ("Val Perp",  f"{data['val_perp']:.0f}",        data["val_perp"]  <  250,  "<250"),
        ("Test Perp", f"{data['test_perp']:.0f}",       data["test_perp"] <  250,  "<250"),
    ]

    card_w = 0.175
    gap    = 0.014
    x0     = 0.025
    y0     = 0.61

    for i, (label, val, ok, tgt) in enumerate(big_metrics):
        cx = x0 + i * (card_w + gap)
        bg = "#edfaf3" if ok else "#fef0f0"
        bc = PASS_C    if ok else FAIL_C

        ax.add_patch(mpatches.FancyBboxPatch(
            (cx, y0), card_w, 0.22,
            boxstyle="round,pad=0.012",
            facecolor=bg, edgecolor=bc, linewidth=1.2,
            transform=ax.transAxes))
        ax.text(cx + card_w / 2, y0 + 0.175, label,
                ha="center", va="center", fontsize=9,
                color="#555", transform=ax.transAxes)
        ax.text(cx + card_w / 2, y0 + 0.1, val,
                ha="center", va="center", fontsize=17, fontweight="500",
                color=bc, transform=ax.transAxes)
        ax.text(cx + card_w / 2, y0 + 0.032,
                ("PASS" if ok else "FAIL") + f"  ({tgt})",
                ha="center", va="center", fontsize=8,
                color=bc, transform=ax.transAxes, fontweight="500")

    # Top-5 accuracy row
    top5_metrics = [
        ("Train Top-5", f"{data['train_top5']*100:.1f}%", data["train_top5"] >= 0.90),
        ("Val Top-5",   f"{data['val_top5']*100:.1f}%",   data["val_top5"]   >= 0.90),
        ("Test Top-5",  f"{data['test_top5']*100:.1f}%",  data["test_top5"]  >= 0.90),
    ]

    sub_w = 0.27
    sub_x0 = 0.025
    for i, (lbl, val, ok) in enumerate(top5_metrics):
        cx = sub_x0 + i * (sub_w + gap)
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx, 0.455), sub_w, 0.13,
            boxstyle="round,pad=0.01",
            facecolor=PANEL_BG, edgecolor="#ddd", linewidth=0.8,
            transform=ax.transAxes))
        ax.text(cx + sub_w / 2, 0.525, lbl,
                ha="center", va="center", fontsize=9, color="#777",
                transform=ax.transAxes)
        col = PASS_C if ok else FAIL_C
        ax.text(cx + sub_w / 2, 0.475, val,
                ha="center", va="center", fontsize=13, fontweight="500",
                color=col, transform=ax.transAxes)

    # Horizontal rule
    ax.plot([0.0, 1.0], [0.44, 0.44], color="#ddd", lw=0.8,
            transform=ax.transAxes, clip_on=False)

    # Seed generation summary table
    headers = ["Seed phrase", "Words gen.", "Avg prob", "Status"]
    col_xs  = [0.025, 0.52, 0.67, 0.82]
    ax.text(0.5, 0.425, "Text Generation Results",
            ha="center", va="center", fontsize=10, fontweight="500",
            color="#333", transform=ax.transAxes)
    for h, cx in zip(headers, col_xs):
        ax.text(cx, 0.395, h, ha="left", va="center", fontsize=8.5,
                color="#555", fontweight="500", transform=ax.transAxes)

    seeds     = data["seeds"]
    probs_all = data["seed_probs"]
    for i, (s, probs) in enumerate(zip(seeds, probs_all)):
        y = 0.37 - i * 0.058
        bg = "#f9f9f9" if i % 2 == 0 else PANEL_BG
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - 0.01), 0.96, 0.05,
            boxstyle="round,pad=0.005",
            facecolor=bg, edgecolor="none",
            transform=ax.transAxes, zorder=0))

        short_seed = s[:38] + "…" if len(s) > 38 else s
        avg_p = np.mean(probs) * 100
        ok = len(probs) >= 30
        vals = [short_seed, str(len(probs)), f"{avg_p:.1f}%",
                "PASS" if ok else "FAIL"]
        cols = ["#333", "#555", "#555", PASS_C if ok else FAIL_C]
        for v, cx, col in zip(vals, col_xs, cols):
            ax.text(cx, y + 0.015, v, ha="left", va="center",
                    fontsize=8, color=col, transform=ax.transAxes)

    # Footer
    passed_all = all([
        data["train_acc"] >= 0.80, data["val_acc"]   >= 0.75,
        data["test_acc"]  >= 0.75, data["val_perp"]  <  250,
        data["test_perp"] <  250,
    ])
    footer_col = PASS_C if passed_all else WARN_C
    ax.text(0.5, 0.012,
            "All assignment targets met" if passed_all
            else "Some targets need attention — check training",
            ha="center", va="center", fontsize=9, fontweight="500",
            color=footer_col, transform=ax.transAxes)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[INFO] Report card        → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Animated GIF — panel-by-panel reveal
# ─────────────────────────────────────────────────────────────────────────────

def plot_animation(data: dict,
                   save_path: str = ANIMATION_GIF,
                   fps: int = 6) -> None:
    """
    Animates the test evaluation: each frame progressively fills in the
    6 panels and ticks up the live metric counters.
    """
    # Pre-draw all 6 panel images into memory as numpy arrays
    panel_fns  = [
        draw_accuracy_bars,
        draw_confusion,
        draw_perplexity_gauge,
        draw_seed_probs,
        draw_attention_heatmap,
        draw_pass_fail_table,
    ]
    panel_titles = [
        "Accuracy", "Confusion", "Perplexity",
        "Seed confidence", "Attention", "Pass/Fail"
    ]

    # Animation: 3 phases per panel — fade-in title, draw content, pause
    # We'll use a simpler approach: reveal panels left-to-right top-to-bottom
    # Each "step" = one panel becoming visible, counter ticking to final value

    metrics_seq = [
        ("Train acc",  f"{data['train_acc']*100:.1f}%",  data["train_acc"] >= 0.80),
        ("Val acc",    f"{data['val_acc']*100:.1f}%",    data["val_acc"]   >= 0.75),
        ("Test acc",   f"{data['test_acc']*100:.1f}%",   data["test_acc"]  >= 0.75),
        ("Val perp",   f"{data['val_perp']:.0f}",        data["val_perp"]  <  250),
        ("Test perp",  f"{data['test_perp']:.0f}",       data["test_perp"] <  250),
    ]

    # Build frames: 8 frames per panel (smooth counter + reveal), 6 panels
    FRAMES_PER_PANEL = 8
    N_PANELS         = 6
    TOTAL_FRAMES     = N_PANELS * FRAMES_PER_PANEL + 10   # +10 hold final

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.42, wspace=0.35,
                            left=0.06, right=0.97,
                            top=0.88, bottom=0.07)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    # Draw all panels upfront (invisible), reveal by alpha
    for ax in axes:
        ax.set_visible(False)

    draw_fns = [
        lambda ax=axes[0]: draw_accuracy_bars(ax, data),
        lambda ax=axes[1]: draw_confusion(ax, data),
        lambda ax=axes[2]: draw_perplexity_gauge(ax, data),
        lambda ax=axes[3]: draw_seed_probs(ax, data),
        lambda ax=axes[4]: draw_attention_heatmap(ax, data),
        lambda ax=axes[5]: draw_pass_fail_table(ax, data),
    ]
    for fn in draw_fns:
        fn()

    suptitle = fig.suptitle(
        "LSTM + Bahdanau Attention — Test Evaluation",
        fontsize=13, fontweight="500", y=0.96, color="#222"
    )
    counter_text = fig.text(
        0.5, 0.925, "", ha="center", fontsize=10, color="#555"
    )

    revealed = [False] * N_PANELS

    def update(frame):
        panel_idx = min(frame // FRAMES_PER_PANEL, N_PANELS - 1)
        sub_frame = frame % FRAMES_PER_PANEL

        # Reveal panels up to current
        for p in range(panel_idx + 1):
            if not revealed[p]:
                axes[p].set_visible(True)
                revealed[p] = True

        # Build counter string from revealed metrics
        shown = metrics_seq[:min(panel_idx + 1, len(metrics_seq))]
        parts = []
        for label, val, ok in shown:
            col_str = "[PASS]" if ok else "[FAIL]"
            parts.append(f"{label}: {val} {col_str}")
        counter_text.set_text("   |   ".join(parts))

        return axes + [counter_text]

    ani = FuncAnimation(
        fig, update,
        frames=TOTAL_FRAMES,
        interval=1000 // fps,
        blit=False,
    )

    print(f"[INFO] Saving animation   → {save_path} ...")
    writer = PillowWriter(fps=fps)
    ani.save(save_path, writer=writer, dpi=90)
    plt.close()
    print(f"[INFO] Animation saved    → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_test(data: dict = None,
                  static_path: str   = STATIC_PNG,
                  anim_path:   str   = ANIMATION_GIF,
                  report_path: str   = REPORT_PNG,
                  fps:         int   = 6) -> None:
    """
    Generates all three test visualisation outputs.

    Args:
        data        : dict from generate_synthetic_test_data() or your real
                      test_metrics dict produced by test.py.
        static_path : Path for the 6-panel static PNG.
        anim_path   : Path for the animated GIF.
        report_path : Path for the single-page report card.
        fps         : GIF frames per second.
    """
    if data is None:
        print("[INFO] No test data provided — using synthetic preview data.")
        data = generate_synthetic_test_data()

    print(f"\n[INFO] Test metrics:")
    print(f"       Train acc   : {data['train_acc']*100:.1f}%")
    print(f"       Val acc     : {data['val_acc']*100:.1f}%")
    print(f"       Test acc    : {data['test_acc']*100:.1f}%")
    print(f"       Val perp    : {data['val_perp']:.1f}")
    print(f"       Test perp   : {data['test_perp']:.1f}")

    plot_static(data,      save_path=static_path)
    plot_report_card(data, save_path=report_path)
    plot_animation(data,   save_path=anim_path, fps=fps)

    print("\n[✓] All test visualisations generated:")
    print(f"    6-panel PNG  → {static_path}")
    print(f"    Report card  → {report_path}")
    print(f"    Animation    → {anim_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise LSTM test evaluation")
    parser.add_argument("--data",  type=str, default=None,
                        help="Path to test_metrics.json (uses synthetic if absent)")
    parser.add_argument("--fps",   type=int, default=6,
                        help="Animation frames per second (default: 6)")
    parser.add_argument("--static-only", action="store_true",
                        help="Skip the GIF, only produce PNG files")
    args = parser.parse_args()

    d = None
    if args.data and os.path.exists(args.data):
        with open(args.data) as f:
            d = json.load(f)
        print(f"[INFO] Loaded test data from {args.data}")

    if args.static_only:
        d = d or generate_synthetic_test_data()
        plot_static(d)
        plot_report_card(d)
    else:
        plot_all_test(d, fps=args.fps)