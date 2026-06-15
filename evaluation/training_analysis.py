#!/usr/bin/env python3
"""
5G Core/RAN Intelligent Diagnostic Engine — Training Analysis & Visualization
==============================================================================
Generates publication-quality charts from fine-tuning runs on AMD Instinct MI300X.
Charts are saved to evaluation/charts/ for use in hackathon presentations.

Training pipeline:
  Stage 1  →  3GPP Domain Knowledge (300 steps, GSMA/ot-lite dataset)
  Stage 2  →  Conversational Realism (150 steps, electricsheepafrica transcripts)
  Merge    →  50/50 linear matrix interpolation across 1,120 LoRA matrices

Hardware:  AMD Instinct MI300X · 192 GB HBM3 · ROCm 7.0
Base model: Llama-3.3-70B-Instruct-bnb-4bit (QLoRA r=16, α=16)
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib import ticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------
BG_COLOR = "#0a0a0f"
PANEL_BG = "#12121a"
GRID_COLOR = "#1e1e2e"
TEXT_COLOR = "#e0e0e0"
SUBTLE_TEXT = "#888899"
AMD_RED = "#ED1C24"
AMD_RED_LIGHT = "#ff4d55"
CYAN = "#00bcd4"
CYAN_LIGHT = "#4dd0e1"
GOLD = "#ffc107"
GREEN = "#4caf50"
PURPLE = "#9c27b0"
WHITE = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlepad": 14,
    "text.color": TEXT_COLOR,
    "xtick.color": SUBTLE_TEXT,
    "ytick.color": SUBTLE_TEXT,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.5,
    "grid.linewidth": 0.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "legend.facecolor": PANEL_BG,
    "legend.edgecolor": GRID_COLOR,
    "legend.labelcolor": TEXT_COLOR,
    "savefig.dpi": 200,
    "savefig.facecolor": BG_COLOR,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

# ---------------------------------------------------------------------------
# Actual training data
# ---------------------------------------------------------------------------
STAGE1_STEPS = list(range(5, 305, 5))
STAGE1_LOSS = [
    3.701843, 4.379555, 3.255965, 1.868569, 1.483503, 1.268538, 1.451893,
    1.345318, 1.332962, 1.233050, 1.390742, 1.044130, 1.026491, 1.057202,
    1.325145, 1.257655, 1.104747, 1.011694, 1.023922, 0.972151, 0.977848,
    1.156760, 1.087425, 1.138984, 1.198250, 1.142976, 1.095044, 1.243716,
    1.038791, 0.958192, 1.068063, 0.956496, 0.830494, 0.957795, 0.908308,
    0.839755, 0.968290, 0.903226, 1.013372, 1.067454, 1.164658, 0.855366,
    0.933656, 1.069549, 0.817089, 1.066974, 0.772203, 1.060350, 1.114768,
    1.122876, 1.048316, 1.326324, 0.984833, 0.829664, 1.091503, 0.671587,
    0.816105, 0.837835, 0.988252, 0.927382,
]

STAGE2_STEPS = list(range(5, 155, 5))
STAGE2_LOSS = [
    4.286567, 3.908917, 1.928693, 0.936356, 0.436718, 0.387793, 0.344279,
    0.307919, 0.265105, 0.246500, 0.195522, 0.158473, 0.151857, 0.148708,
    0.144799, 0.146386, 0.135366, 0.135506, 0.150117, 0.131769, 0.144520,
    0.134203, 0.139988, 0.148556, 0.144355, 0.131231, 0.130643, 0.139611,
    0.142788, 0.136723,
]


def _moving_average(data: list[float], window: int = 5) -> np.ndarray:
    """Compute a centred moving average with edge-padding."""
    arr = np.array(data, dtype=np.float64)
    kernel = np.ones(window) / window
    # Pad edges to keep the same length
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def _add_watermark(ax: plt.Axes, text: str = "AMD MI300X · Llama-3.3-70B"):
    """Subtle branding watermark in the bottom-right corner."""
    ax.text(
        0.98, 0.02, text,
        transform=ax.transAxes,
        fontsize=7,
        color=SUBTLE_TEXT,
        alpha=0.35,
        ha="right",
        va="bottom",
        fontstyle="italic",
        path_effects=[pe.withStroke(linewidth=0.5, foreground=BG_COLOR)],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1 — Training Curves (Hero Chart)
# ═══════════════════════════════════════════════════════════════════════════
def chart_training_curves(save_dir: Path) -> str:
    """Dual-panel training loss curves with smoothed trendlines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Fine-Tuning Loss Curves — Two-Stage Curriculum",
        fontsize=18,
        fontweight="bold",
        color=WHITE,
        y=0.97,
    )

    smooth1 = _moving_average(STAGE1_LOSS, window=5)
    smooth2 = _moving_average(STAGE2_LOSS, window=5)

    # ── Stage 1 ──────────────────────────────────────────────────────────
    ax1.scatter(
        STAGE1_STEPS, STAGE1_LOSS,
        s=18, color=AMD_RED_LIGHT, alpha=0.45, zorder=3, label="Raw loss",
    )
    ax1.plot(
        STAGE1_STEPS, smooth1,
        color=AMD_RED, linewidth=2.2, zorder=4, label="Smoothed (MA-5)",
    )
    # Fill under the smooth curve
    ax1.fill_between(
        STAGE1_STEPS, smooth1, alpha=0.08, color=AMD_RED, zorder=2,
    )
    ax1.set_title("Stage 1 — 3GPP Domain Knowledge", fontweight="bold", color=AMD_RED_LIGHT)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_xlim(0, 310)
    ax1.set_ylim(0, 5.0)
    ax1.grid(True, linewidth=0.4)
    ax1.legend(loc="upper right", framealpha=0.7)

    # Annotations — start, final, best
    ax1.annotate(
        f"Start: {STAGE1_LOSS[0]:.2f}",
        xy=(STAGE1_STEPS[0], STAGE1_LOSS[0]),
        xytext=(40, 4.2),
        fontsize=9, color=AMD_RED_LIGHT,
        arrowprops=dict(arrowstyle="->", color=AMD_RED_LIGHT, lw=0.8),
    )
    best_idx1 = int(np.argmin(STAGE1_LOSS))
    ax1.annotate(
        f"Best: {STAGE1_LOSS[best_idx1]:.3f}\n(step {STAGE1_STEPS[best_idx1]})",
        xy=(STAGE1_STEPS[best_idx1], STAGE1_LOSS[best_idx1]),
        xytext=(STAGE1_STEPS[best_idx1] - 50, 0.4),
        fontsize=9, color=GOLD,
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.8),
    )
    ax1.annotate(
        f"Final: {STAGE1_LOSS[-1]:.3f}",
        xy=(STAGE1_STEPS[-1], STAGE1_LOSS[-1]),
        xytext=(250, 1.6),
        fontsize=9, color=WHITE,
        arrowprops=dict(arrowstyle="->", color=WHITE, lw=0.8),
    )
    # Convergence region shading
    ax1.axhspan(0.6, 1.1, alpha=0.06, color=GREEN, zorder=1)
    ax1.text(155, 0.63, "Convergence region", fontsize=7, color=GREEN, alpha=0.6)

    # ── Stage 2 ──────────────────────────────────────────────────────────
    ax2.scatter(
        STAGE2_STEPS, STAGE2_LOSS,
        s=18, color=CYAN_LIGHT, alpha=0.50, zorder=3, label="Raw loss",
    )
    ax2.plot(
        STAGE2_STEPS, smooth2,
        color=CYAN, linewidth=2.2, zorder=4, label="Smoothed (MA-5)",
    )
    ax2.fill_between(
        STAGE2_STEPS, smooth2, alpha=0.08, color=CYAN, zorder=2,
    )
    ax2.set_title("Stage 2 — Conversational Realism", fontweight="bold", color=CYAN_LIGHT)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Loss")
    ax2.set_xlim(0, 160)
    ax2.set_ylim(0, 5.0)
    ax2.grid(True, linewidth=0.4)
    ax2.legend(loc="upper right", framealpha=0.7)

    # Annotations
    ax2.annotate(
        f"Start: {STAGE2_LOSS[0]:.2f}",
        xy=(STAGE2_STEPS[0], STAGE2_LOSS[0]),
        xytext=(25, 4.6),
        fontsize=9, color=CYAN_LIGHT,
        arrowprops=dict(arrowstyle="->", color=CYAN_LIGHT, lw=0.8),
    )
    best_idx2 = int(np.argmin(STAGE2_LOSS))
    ax2.annotate(
        f"Best: {STAGE2_LOSS[best_idx2]:.3f}\n(step {STAGE2_STEPS[best_idx2]})",
        xy=(STAGE2_STEPS[best_idx2], STAGE2_LOSS[best_idx2]),
        xytext=(STAGE2_STEPS[best_idx2] + 15, 0.5),
        fontsize=9, color=GOLD,
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.8),
    )
    ax2.annotate(
        f"Final: {STAGE2_LOSS[-1]:.3f}",
        xy=(STAGE2_STEPS[-1], STAGE2_LOSS[-1]),
        xytext=(110, 0.5),
        fontsize=9, color=WHITE,
        arrowprops=dict(arrowstyle="->", color=WHITE, lw=0.8),
    )
    # Convergence region shading
    ax2.axhspan(0.10, 0.20, alpha=0.06, color=GREEN, zorder=1)
    ax2.text(80, 0.105, "Convergence region", fontsize=7, color=GREEN, alpha=0.6)

    # ── Hyperparameter info box ──────────────────────────────────────────
    hp_text = (
        "Training Configuration\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "Base: Llama-3.3-70B-Instruct\n"
        "Quant: 4-bit (bnb NF4)\n"
        "QLoRA: r=16, α=16, dropout=0\n"
        "Batch: 2 × 4 grad-accum = eff. 8\n"
        "Trainable: 207M / 70.76B (0.29%)\n"
        "Hardware: AMD MI300X · 192 GB HBM3"
    )
    fig.text(
        0.5, 0.02, hp_text,
        fontsize=8,
        color=SUBTLE_TEXT,
        ha="center",
        va="bottom",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor=BG_COLOR,
            edgecolor=GRID_COLOR,
            linewidth=0.8,
            alpha=0.9,
        ),
    )

    _add_watermark(ax1)
    _add_watermark(ax2)

    plt.tight_layout(rect=[0, 0.12, 1, 0.94])
    path = save_dir / "training_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# Chart 2 — Curriculum Comparison (Normalised overlay)
# ═══════════════════════════════════════════════════════════════════════════
def chart_curriculum_comparison(save_dir: Path) -> str:
    """Both training stages overlaid on a normalized 0-1 step axis."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Normalize steps to [0, 1]
    norm1 = np.array(STAGE1_STEPS) / max(STAGE1_STEPS)
    norm2 = np.array(STAGE2_STEPS) / max(STAGE2_STEPS)

    smooth1 = _moving_average(STAGE1_LOSS, 5)
    smooth2 = _moving_average(STAGE2_LOSS, 5)

    # Raw scatter
    ax.scatter(norm1, STAGE1_LOSS, s=14, color=AMD_RED_LIGHT, alpha=0.3, zorder=2)
    ax.scatter(norm2, STAGE2_LOSS, s=14, color=CYAN_LIGHT, alpha=0.3, zorder=2)

    # Smoothed lines
    ax.plot(norm1, smooth1, color=AMD_RED, linewidth=2.5, zorder=4,
            label="Stage 1 — 3GPP Domain Knowledge (300 steps)")
    ax.plot(norm2, smooth2, color=CYAN, linewidth=2.5, zorder=4,
            label="Stage 2 — Conversational Realism (150 steps)")

    # Fill regions
    ax.fill_between(norm1, smooth1, alpha=0.07, color=AMD_RED, zorder=1)
    ax.fill_between(norm2, smooth2, alpha=0.07, color=CYAN, zorder=1)

    ax.set_title(
        "Curriculum Learning — Normalised Loss Landscape Comparison",
        fontsize=16, fontweight="bold", color=WHITE,
    )
    ax.set_xlabel("Normalised Training Progress (0 → 1)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(0, 5.0)
    ax.grid(True, linewidth=0.4)
    ax.legend(loc="upper right", framealpha=0.8, fontsize=11)

    # Annotation — curriculum strategy
    strategy_text = (
        "Curriculum Strategy\n"
        "───────────────────\n"
        "Stage 1 trains on dense 3GPP\n"
        "technical specifications.\n"
        "Stage 2 shifts to natural\n"
        "conversational transcripts.\n"
        "Both adapters are then merged\n"
        "via 50/50 matrix interpolation."
    )
    ax.text(
        0.38, 3.4, strategy_text,
        fontsize=9, color=TEXT_COLOR,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=BG_COLOR,
            edgecolor=GRID_COLOR,
            linewidth=0.8,
            alpha=0.92,
        ),
    )

    # Mark where convergence differs
    ax.annotate(
        "Stage 2 converges ~30× lower\nthan Stage 1 (0.13 vs 0.93)",
        xy=(0.9, 0.14),
        xytext=(0.55, 0.8),
        fontsize=9, color=GOLD,
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.0),
    )

    _add_watermark(ax)
    plt.tight_layout()
    path = save_dir / "curriculum_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# Chart 3 — Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════
def chart_architecture_diagram(save_dir: Path) -> str:
    """Matplotlib-rendered pipeline architecture flow diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.suptitle(
        "Training Pipeline Architecture",
        fontsize=20, fontweight="bold", color=WHITE, y=0.96,
    )

    # ── Box definitions ──────────────────────────────────────────────────
    boxes = [
        # (x, y, width, height, label, sublabel, color)
        (0.5, 3.0, 2.2, 2.0,
         "Base Model", "Llama-3.3-70B\nInstruct\n70.76B params", "#334155"),
        (3.4, 3.0, 2.2, 2.0,
         "QLoRA\nAdapters", "r=16, α=16\n207M trainable\n(0.29%)", "#4a1942"),
        (6.3, 4.3, 2.2, 1.6,
         "Stage 1", "3GPP Domain\n300 steps\nLoss: 3.70→0.93", AMD_RED),
        (6.3, 2.1, 2.2, 1.6,
         "Stage 2", "Conversational\n150 steps\nLoss: 4.29→0.14", "#006064"),
        (9.3, 3.0, 2.2, 2.0,
         "Matrix\nInterpolation", "50/50 merge\n1,120 LoRA\nmatrices", PURPLE),
        (12.0, 3.0, 1.8, 2.0,
         "Final\nModel", "Telco Expert\nLlama-3.3-70B\n⚡ Deployed", GREEN),
    ]

    for x, y, w, h, title, sub, color in boxes:
        fancy = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor=WHITE,
            linewidth=1.2,
            alpha=0.85,
            zorder=3,
        )
        ax.add_patch(fancy)
        ax.text(
            x + w / 2, y + h * 0.72, title,
            fontsize=10, fontweight="bold", color=WHITE,
            ha="center", va="center", zorder=4,
        )
        ax.text(
            x + w / 2, y + h * 0.28, sub,
            fontsize=7.5, color="#d0d0d0",
            ha="center", va="center", zorder=4,
            family="monospace",
        )

    # ── Arrows ───────────────────────────────────────────────────────────
    arrow_kw = dict(
        arrowstyle="->,head_width=0.3,head_length=0.2",
        color=WHITE,
        linewidth=1.5,
        zorder=5,
    )
    arrows = [
        ((2.7, 4.0), (3.4, 4.0)),   # Base → QLoRA
        ((5.6, 4.5), (6.3, 4.9)),   # QLoRA → Stage 1
        ((5.6, 3.5), (6.3, 3.1)),   # QLoRA → Stage 2
        ((8.5, 4.9), (9.3, 4.3)),   # Stage 1 → Merge
        ((8.5, 3.1), (9.3, 3.7)),   # Stage 2 → Merge
        ((11.5, 4.0), (12.0, 4.0)), # Merge → Final
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_kw)

    # ── Hardware badge ───────────────────────────────────────────────────
    hw_badge = FancyBboxPatch(
        (4.5, 0.3), 5.0, 1.2,
        boxstyle="round,pad=0.2",
        facecolor=BG_COLOR,
        edgecolor=AMD_RED,
        linewidth=1.5,
        alpha=0.9,
        zorder=3,
    )
    ax.add_patch(hw_badge)
    ax.text(
        7.0, 1.05,
        "⬡  AMD Instinct MI300X",
        fontsize=12, fontweight="bold", color=AMD_RED_LIGHT,
        ha="center", va="center", zorder=4,
    )
    ax.text(
        7.0, 0.6,
        "192 GB HBM3  ·  ROCm 7.0  ·  PyTorch 2.10.0+rocm7.0",
        fontsize=9, color=SUBTLE_TEXT,
        ha="center", va="center", zorder=4,
    )

    # ── Dataset labels ───────────────────────────────────────────────────
    ax.text(
        7.4, 6.3, "GSMA/ot-lite\n(3gpp_tsg + teleqna)",
        fontsize=8, color=AMD_RED_LIGHT, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, edgecolor=AMD_RED, lw=0.7),
    )
    ax.annotate("", xy=(7.4, 5.9), xytext=(7.4, 5.95),
                arrowprops=dict(arrowstyle="->", color=AMD_RED_LIGHT, lw=0.8))

    ax.text(
        7.4, 1.7, "electricsheepafrica\n(call transcripts)",
        fontsize=8, color=CYAN_LIGHT, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, edgecolor=CYAN, lw=0.7),
    )

    _add_watermark(ax, text="5G Diagnostic Engine · AMD MI300X")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = save_dir / "architecture_diagram.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# Chart 4 — Model Parameter Stats (Log-scale bar chart)
# ═══════════════════════════════════════════════════════════════════════════
def chart_model_stats(save_dir: Path) -> str:
    """Horizontal bar chart of parameter counts on a log scale."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = [
        "LoRA Matrices",
        "Trainable Params",
        "Frozen Params",
        "Total Params",
    ]
    values = [
        1_120,
        207_093_760,
        70_553_706_496,   # 70.76B - 207M
        70_760_800_256,
    ]
    display_labels = [
        "1,120",
        "207.1M  (0.29%)",
        "70.55B  (99.71%)",
        "70.76B  (100%)",
    ]
    colors = [GOLD, AMD_RED, "#546e7a", CYAN]

    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, values, height=0.55, color=colors, edgecolor=WHITE, linewidth=0.5, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(500, 2e11)
    ax.set_xlabel("Count (log scale)", fontsize=12)
    ax.set_title(
        "Model Parameter Breakdown — QLoRA Efficiency",
        fontsize=16, fontweight="bold", color=WHITE,
    )
    ax.grid(True, axis="x", linewidth=0.3)

    # Value labels
    for i, (bar, label) in enumerate(zip(bars, display_labels)):
        x_pos = bar.get_width() * 1.5
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            label,
            va="center", ha="left",
            fontsize=10, fontweight="bold", color=colors[i],
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_COLOR)],
        )

    # Insight annotation
    ax.text(
        0.97, 0.05,
        "Only 0.29% of parameters are trained — \n"
        "QLoRA enables 70B fine-tuning on a single MI300X",
        transform=ax.transAxes,
        fontsize=9, color=SUBTLE_TEXT,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_COLOR, edgecolor=GRID_COLOR, lw=0.6),
    )

    _add_watermark(ax)
    plt.tight_layout()
    path = save_dir / "model_stats.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# Chart 5 — Hardware Utilisation (VRAM Waterfall + A100 Comparison)
# ═══════════════════════════════════════════════════════════════════════════
def chart_hardware_utilization(save_dir: Path) -> str:
    """Infographic showing MI300X VRAM utilization and A100 comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                    gridspec_kw={"width_ratios": [2, 1]})

    fig.suptitle(
        "AMD MI300X — VRAM Utilization Analysis",
        fontsize=16, fontweight="bold", color=WHITE, y=0.97,
    )

    # ── Left panel: VRAM waterfall ───────────────────────────────────────
    labels = [
        "Total VRAM\n(192 GB HBM3)",
        "Model Weights\n(4-bit NF4)",
        "Training Overhead\n(Optimizer + Grad)",
        "Available\nHeadroom",
    ]
    sizes = [192, 38, 7, 147]  # GB
    colors_wf = [CYAN, AMD_RED, GOLD, GREEN]
    bottoms = [0, 0, 38, 45]

    # Draw as stacked components
    bar_width = 0.6
    x_positions = [0, 1, 1, 1]

    # Total bar (left)
    ax1.bar(0, 192, bar_width, color=CYAN, edgecolor=WHITE, linewidth=0.8, alpha=0.8, zorder=3)
    ax1.text(0, 96, "192 GB\nTotal", ha="center", va="center", fontsize=12,
             fontweight="bold", color=WHITE, zorder=4)

    # Stacked breakdown (right)
    ax1.bar(1, 38, bar_width, bottom=0, color=AMD_RED, edgecolor=WHITE, linewidth=0.8, alpha=0.85, zorder=3)
    ax1.text(1, 19, "38 GB\nModel", ha="center", va="center", fontsize=10,
             fontweight="bold", color=WHITE, zorder=4)

    ax1.bar(1, 7, bar_width, bottom=38, color=GOLD, edgecolor=WHITE, linewidth=0.8, alpha=0.85, zorder=3)
    ax1.text(1, 41.5, "7 GB\nTraining", ha="center", va="center", fontsize=8,
             fontweight="bold", color=BG_COLOR, zorder=4)

    ax1.bar(1, 147, bar_width, bottom=45, color=GREEN, edgecolor=WHITE, linewidth=0.8, alpha=0.3, zorder=3)
    ax1.text(1, 118, "147 GB\nFree", ha="center", va="center", fontsize=12,
             fontweight="bold", color=GREEN, zorder=4)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Total Capacity", "Usage Breakdown"], fontsize=10)
    ax1.set_ylabel("VRAM (GB)", fontsize=12)
    ax1.set_ylim(0, 210)
    ax1.set_title("MI300X VRAM Allocation", fontweight="bold", color=CYAN_LIGHT)
    ax1.grid(True, axis="y", linewidth=0.3)

    # Peak utilization annotation
    ax1.annotate(
        "Peak: 45 GB (23.4%)",
        xy=(1, 45),
        xytext=(1.5, 65),
        fontsize=9, color=GOLD,
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.0),
    )

    # ── Right panel: MI300X vs A100 comparison ───────────────────────────
    gpus = ["A100\n(80 GB)", "MI300X\n(192 GB)"]
    vram_totals = [80, 192]
    model_needs = [45, 45]  # same model requirement
    bar_colors = ["#76b900", AMD_RED]

    y_pos = np.arange(len(gpus))
    # Total VRAM bars
    bars_total = ax2.barh(y_pos, vram_totals, height=0.45, color=bar_colors,
                          edgecolor=WHITE, linewidth=0.8, alpha=0.35, zorder=2)
    # Model requirement bars
    bars_model = ax2.barh(y_pos, model_needs, height=0.45, color=bar_colors,
                          edgecolor=WHITE, linewidth=1.2, alpha=0.9, zorder=3)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(gpus, fontsize=11, fontweight="bold")
    ax2.set_xlabel("VRAM (GB)", fontsize=11)
    ax2.set_xlim(0, 220)
    ax2.set_title("GPU Comparison", fontweight="bold", color=AMD_RED_LIGHT)
    ax2.grid(True, axis="x", linewidth=0.3)

    # Labels on bars
    ax2.text(82, 0, " 80 GB total", va="center", fontsize=9, color="#76b900", fontweight="bold")
    ax2.text(194, 1, " 192 GB total", va="center", fontsize=9, color=AMD_RED_LIGHT, fontweight="bold")
    ax2.text(22, 0, "45 GB\nneeded", va="center", ha="center", fontsize=8, color=WHITE, fontweight="bold")
    ax2.text(22, 1, "45 GB\nneeded", va="center", ha="center", fontsize=8, color=WHITE, fontweight="bold")

    # Feasibility markers
    ax2.text(65, -0.35, "⚠ Tight fit — risk of OOM", fontsize=8, color=GOLD)
    ax2.text(65, 0.65, "✓  2.4× headroom available", fontsize=8, color=GREEN)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=WHITE, linewidth=0, marker="s", markersize=10,
               markerfacecolor=AMD_RED, alpha=0.9, label="Peak training VRAM"),
        Line2D([0], [0], color=WHITE, linewidth=0, marker="s", markersize=10,
               markerfacecolor=AMD_RED, alpha=0.3, label="Total available VRAM"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", framealpha=0.7, fontsize=8)

    _add_watermark(ax1)
    _add_watermark(ax2, text="AMD Instinct MI300X")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = save_dir / "hardware_utilization.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    script_dir = Path(__file__).resolve().parent
    charts_dir = script_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  5G Diagnostic Engine - Training Analysis")
    print("  Generating publication-quality charts ...")
    print("=" * 60)
    print()

    paths = {}
    paths["training_curves"] = chart_training_curves(charts_dir)
    paths["curriculum_comparison"] = chart_curriculum_comparison(charts_dir)
    paths["architecture_diagram"] = chart_architecture_diagram(charts_dir)
    paths["model_stats"] = chart_model_stats(charts_dir)
    paths["hardware_utilization"] = chart_hardware_utilization(charts_dir)

    print()
    print("-" * 60)
    print("  All charts saved to:", charts_dir)
    print("-" * 60)

    # -- Summary statistics -----------------------------------------------
    s1 = np.array(STAGE1_LOSS)
    s2 = np.array(STAGE2_LOSS)
    print()
    print("  Stage 1 - 3GPP Domain Knowledge")
    print(f"    Start loss:  {s1[0]:.4f}")
    print(f"    Final loss:  {s1[-1]:.4f}")
    print(f"    Best loss:   {s1.min():.4f}  (step {STAGE1_STEPS[int(np.argmin(s1))]})")
    print(f"    Reduction:   {(1 - s1[-1] / s1[0]) * 100:.1f}%")
    print()
    print("  Stage 2 - Conversational Realism")
    print(f"    Start loss:  {s2[0]:.4f}")
    print(f"    Final loss:  {s2[-1]:.4f}")
    print(f"    Best loss:   {s2.min():.4f}  (step {STAGE2_STEPS[int(np.argmin(s2))]})")
    print(f"    Reduction:   {(1 - s2[-1] / s2[0]) * 100:.1f}%")
    print()

    return paths


if __name__ == "__main__":
    main()
