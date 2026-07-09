"""
Publication-quality plotting theme for ICLR dataset presentation.

Provides consistent styling across all figures:
- Colorblind-safe palettes (Okabe-Ito)
- Clean white backgrounds, no top/right spines
- Dual export: PNG@300dpi + PDF vector
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from .paths import RESULTS_DIR

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIGURES_DIR = RESULTS_DIR / "dataset_presentation" / "figures"
TABLES_DIR = RESULTS_DIR / "dataset_presentation" / "tables"
TEXT_DIR = RESULTS_DIR / "dataset_presentation" / "text"

# ---------------------------------------------------------------------------
# Year palette  (Okabe-Ito colorblind-safe)
# ---------------------------------------------------------------------------
YEAR_COLORS = {
    2023: "#E69F00",  # orange
    2024: "#56B4E9",  # sky blue
    2025: "#009E73",  # bluish green
    2026: "#CC79A7",  # reddish purple
}

# ---------------------------------------------------------------------------
# Decision palette
# ---------------------------------------------------------------------------
DECISION_COLORS = {
    "Oral": "#1a6b3c",       # dark green
    "Spotlight": "#2d9e5e",  # medium green
    "Poster": "#6cc88a",     # light green
    "Rejected": "#d9534f",   # red
    "Withdrawn": "#999999",  # gray
    "Pending": "#cccccc",    # light gray
}

DECISION_ORDER = ["Oral", "Spotlight", "Poster", "Rejected", "Withdrawn"]

# ---------------------------------------------------------------------------
# Theme setup
# ---------------------------------------------------------------------------

def setup_publication_theme() -> None:
    """Apply publication-quality rcParams globally."""
    mpl.rcParams.update({
        # Figure
        "figure.figsize": (7, 4.5),
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Grid
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        # Font
        "font.size": 10,
        "font.family": "sans-serif",
        # Tick
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Legend
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        # Save
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.facecolor": "white",
    })


def save_figure(fig: plt.Figure, name: str) -> tuple[Path, Path]:
    """Save figure as both PNG (300 dpi) and PDF to the figures directory.

    Parameters
    ----------
    fig : matplotlib Figure
    name : str
        Filename stem without extension (e.g. "fig_1_dataset_overview").

    Returns
    -------
    (png_path, pdf_path)
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png_path.name}, {pdf_path.name}")
    return png_path, pdf_path
