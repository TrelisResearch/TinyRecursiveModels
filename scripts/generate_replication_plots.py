from __future__ import annotations

import pandas as pd
import wandb
from utils import apply_paper_style, ensure_cache_dirs, save_figure, unique_legend
import matplotlib.pyplot as plt


RUN_PATH = "trelis/Arc2concept-aug-1000-ACT-torch/942w7khi"


def fetch_history() -> pd.DataFrame:
    api = wandb.Api()
    run = api.run(RUN_PATH)
    history = run.history(
        keys=["_step", "ARC/pass@2", "ARC/pass@1000", "all.exact_accuracy", "train/exact_accuracy"],
        pandas=True,
    ).dropna()
    history["_step_k"] = history["_step"] / 1000.0
    return history


def plot_pass_metrics(history: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.plot(
        history["_step_k"],
        history["ARC/pass@2"],
        color="0.1",
        linestyle="-",
        marker="o",
        label="pass@2",
    )
    ax.plot(
        history["_step_k"],
        history["ARC/pass@1000"],
        color="0.6",
        linestyle="--",
        marker="s",
        label="pass@1000",
    )
    ax.set_xlabel("Training step (×10³)")
    ax.set_ylabel("Accuracy")
    ax.set_title("ARC Evaluation Accuracy")
    unique_legend(ax, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "pass_accuracy")
    plt.close(fig)


def plot_exact_accuracy(history: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.plot(
        history["_step_k"],
        history["all.exact_accuracy"],
        color="0.1",
        linestyle="-",
        marker="o",
        label="Evaluation exact accuracy",
    )
    ax.plot(
        history["_step_k"],
        history["train/exact_accuracy"],
        color="0.6",
        linestyle="--",
        marker="s",
        label="Train exact accuracy",
    )
    ax.set_xlabel("Training step (×10³)")
    ax.set_ylabel("Exact accuracy")
    ax.set_title("Exact Accuracy During Training")
    unique_legend(ax, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "exact_accuracy")
    plt.close(fig)


def main() -> None:
    ensure_cache_dirs()
    apply_paper_style()
    history = fetch_history()
    plot_pass_metrics(history)
    plot_exact_accuracy(history)
    print("Saved replication plots to arc2025_paper/figures/(pass_accuracy|exact_accuracy).(pdf|png)")


if __name__ == "__main__":
    main()
