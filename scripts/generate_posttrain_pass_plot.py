from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb


def ensure_cache_dirs() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    mpl_dir = repo_root / ".matplotlib"
    cache_dir = repo_root / ".cache"
    mpl_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def fetch_histories(run_ids: dict[str, str], max_step: int = 6000) -> dict[str, pd.DataFrame]:
    api = wandb.Api()
    histories: dict[str, pd.DataFrame] = {}
    for label, run_id in run_ids.items():
        run = api.run(f"trelis/Arc-eval2-aug-1000-ACT-torch/{run_id}")
        history = run.history(keys=["_step", "ARC/pass@2"], pandas=True).dropna()
        history = history[history["_step"] <= max_step].sort_values("_step")
        history["_step_k"] = history["_step"] / 1000.0
        histories[label] = history
    return histories


def plot_pass_at_2(histories: dict[str, pd.DataFrame]) -> None:
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

    base_colors = {
        "Full fine-tune": "0.1",
        "Embeddings only": "0.3",
        "Embed+full (quarter)": "0.5",
        "Embed+full (half)": "0.7",
        "LoRA": "0.9",
    }
    markers = {
        "Full fine-tune": "o",
        "Embeddings only": "s",
        "Embed+full (quarter)": "^",
        "Embed+full (half)": "D",
        "LoRA": "P",
    }

    fig, ax = plt.subplots()
    for label, history in histories.items():
        ax.plot(
            history["_step_k"],
            history["ARC/pass@2"],
            color=base_colors[label],
            linestyle="-",
            marker=markers[label],
            label=label,
        )

    ax.set_xlabel("Training step (×10³)")
    ax.set_ylabel("Evaluation pass@2")
    ax.set_title("ARC Evaluation pass@2 During Post-training")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    output_dir = Path("arc2025_paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "posttrain_pass_accuracy.pdf")
    fig.savefig(output_dir / "posttrain_pass_accuracy.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_cache_dirs()
    run_ids = {
        "Full fine-tune": "5f806fuh",
        "Embeddings only": "qvemnhoc",
        "Embed+full (quarter)": "rczfrinm",
        "Embed+full (half)": "wttj0clr",
        "LoRA": "g8kks1zz",
    }
    histories = fetch_histories(run_ids)
    plot_pass_at_2(histories)
    print("Saved arc2025_paper/figures/posttrain_pass_accuracy.(pdf|png)")


if __name__ == "__main__":
    main()
