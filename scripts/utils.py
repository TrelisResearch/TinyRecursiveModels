from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
MPL_DIR = REPO_ROOT / ".matplotlib"
CACHE_DIR = REPO_ROOT / ".cache"


def ensure_cache_dirs() -> None:
    """Ensure matplotlib and font caches are writable."""
    MPL_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))


def apply_paper_style() -> None:
    """Configure Matplotlib to match the grayscale journal styling used in the paper."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.figsize": (5.5, 3.4),
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def unique_legend(ax: plt.Axes, **legend_kwargs) -> None:
    """Deduplicate legend entries while preserving order."""
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique_handles: list[plt.Artist] = []
    unique_labels: list[str] = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    legend_kwargs.setdefault("frameon", False)
    ax.legend(unique_handles, unique_labels, **legend_kwargs)


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure as both PDF and PNG under arc2025_paper/figures."""
    output_dir = REPO_ROOT / "arc2025_paper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png", dpi=300)
