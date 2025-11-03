from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import wandb
import matplotlib.pyplot as plt

# Add parent scripts to path for utils
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from utils import apply_paper_style, ensure_cache_dirs, save_figure, unique_legend


RUN_PATH = "trelis/Arc2-pretrain-final-ACT-torch/9bp7agqh"


def fetch_history() -> pd.DataFrame:
    """Fetch embedding cosine metrics from the pretrain_all_200k run."""
    api = wandb.Api()
    run = api.run(RUN_PATH)
    history = run.history(
        keys=[
            "_step",
            "all.embedding_cosine",
            "train/embedding_cosine",
            "all.embedding_cosine_within_task",
            "train/embedding_cosine_within_task",
        ],
        pandas=True,
    ).dropna()
    history["_step_k"] = history["_step"] / 1000.0
    return history


def plot_embedding_cosine_across(history: pd.DataFrame) -> None:
    """Plot embedding cosine similarity across base tasks."""
    fig, ax = plt.subplots()
    ax.plot(
        history["_step_k"],
        history["all.embedding_cosine"],
        color="0.1",
        linestyle="-",
        marker="o",
        label="Evaluation",
    )
    ax.plot(
        history["_step_k"],
        history["train/embedding_cosine"],
        color="0.6",
        linestyle="--",
        marker="s",
        label="Training",
    )
    ax.set_xlabel("Training step (×10³)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Embedding Cosine Similarity Across Tasks")
    unique_legend(ax, loc="best")
    fig.tight_layout()
    save_figure(fig, "embedding_cosine_across")
    plt.close(fig)


def plot_embedding_cosine_within(history: pd.DataFrame) -> None:
    """Plot embedding cosine similarity within task variants."""
    fig, ax = plt.subplots()
    ax.plot(
        history["_step_k"],
        history["all.embedding_cosine_within_task"],
        color="0.1",
        linestyle="-",
        marker="o",
        label="Evaluation",
    )
    ax.plot(
        history["_step_k"],
        history["train/embedding_cosine_within_task"],
        color="0.6",
        linestyle="--",
        marker="s",
        label="Training",
    )
    ax.set_xlabel("Training step (×10³)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Embedding Cosine Similarity Within Task Variants")
    unique_legend(ax, loc="best")
    fig.tight_layout()
    save_figure(fig, "embedding_cosine_within")
    plt.close(fig)


def main() -> None:
    ensure_cache_dirs()
    apply_paper_style()
    print(f"Fetching data from W&B run: {RUN_PATH}")
    history = fetch_history()
    print(f"Fetched {len(history)} data points")

    print("Generating embedding_cosine_across plot...")
    plot_embedding_cosine_across(history)

    print("Generating embedding_cosine_within plot...")
    plot_embedding_cosine_within(history)

    print("Saved embedding cosine plots to arc2025_paper/figures/embedding_cosine_(across|within).(pdf|png)")


if __name__ == "__main__":
    main()
