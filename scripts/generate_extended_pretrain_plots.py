from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

from utils import apply_paper_style, ensure_cache_dirs, save_figure, unique_legend


RUN_IDS = {
    "all-200k": "trelis/Arc2-pretrain-final-ACT-torch/9bp7agqh",
    "hard-1M": "trelis/Arc2-pretrain-final-ACT-torch/ogqol1fh",
}

METRIC_ALIASES = {
    "pass@2": ["ARC/pass@2", "all.pass@2", "pass@2"],
    "pass@1000": ["ARC/pass@1000", "all.pass@1000", "pass@1000"],
    "eval_exact": ["all.exact_accuracy"],
    "train_exact": ["train/exact_accuracy"],
    "eval_within": ["all.embedding_cosine_within_task"],
    "train_within": ["train.embedding_cosine_within_task"],
    "eval_across": ["all.embedding_cosine"],
    "train_across": ["train.embedding_cosine"],
}

PLOT_SPECS = {
    "extended_pass_accuracy": [
        {"key": "pass@2", "label": "pass@2", "color": "0.1", "linestyle": "-"},
        {"key": "pass@1000", "label": "pass@1000", "color": "0.6", "linestyle": "--"},
    ],
    "extended_exact_accuracy": [
        {"key": "eval_exact", "label": "Evaluation exact accuracy", "color": "0.1", "linestyle": "-"},
        {"key": "train_exact", "label": "Train exact accuracy", "color": "0.6", "linestyle": "--"},
    ],
    "embedding_cosine_within": [
        {"key": "eval_within", "label": "Evaluation", "color": "0.1", "linestyle": "-"},
        {"key": "train_within", "label": "Train", "color": "0.6", "linestyle": "--"},
    ],
    "embedding_cosine_across": [
        {"key": "eval_across", "label": "Evaluation", "color": "0.1", "linestyle": "-"},
        {"key": "train_across", "label": "Train", "color": "0.6", "linestyle": "--"},
    ],
}

RUN_STYLES = {"all-200k": "-", "hard-1M": "--"}
RUN_MARKERS = {"all-200k": "o", "hard-1M": "s"}

LEGEND_KW = {
    "extended_pass_accuracy": {"loc": "lower right", "bbox_to_anchor": (0.98, 0.02)},
    "extended_exact_accuracy": {"loc": "center right", "bbox_to_anchor": (0.98, 0.5)},
    "embedding_cosine_within": {"loc": "upper left", "bbox_to_anchor": (0.02, 0.98)},
    "embedding_cosine_across": {"loc": "upper left", "bbox_to_anchor": (0.02, 0.98)},
}

TITLES = {
    "extended_pass_accuracy": "ARC Evaluation Accuracy (Extended Runs)",
    "extended_exact_accuracy": "Exact Accuracy (Extended Runs)",
    "embedding_cosine_within": "Embedding Cosine Within Task Variants",
    "embedding_cosine_across": "Embedding Cosine Across Tasks",
}

Y_LABELS = {
    "extended_pass_accuracy": "Accuracy",
    "extended_exact_accuracy": "Exact accuracy",
    "embedding_cosine_within": "Cosine similarity",
    "embedding_cosine_across": "Cosine similarity",
}


def _first_available_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    return None


def fetch_histories() -> dict[str, pd.DataFrame]:
    api = wandb.Api()
    histories: dict[str, pd.DataFrame] = {}
    required_keys = {"_step", "Step", "all.steps"}
    for aliases in METRIC_ALIASES.values():
        required_keys.update(aliases)

    for run_name, run_path in RUN_IDS.items():
        run = api.run(run_path)
        history = run.history(keys=list(required_keys), pandas=True)
        history = history.dropna(how="all")

        if "_step" not in history.columns:
            fallback = _first_available_column(history, ["Step", "all.steps"])
            if fallback is not None:
                history["_step"] = history[fallback]
            else:
                history["_step"] = history.index.to_series().astype(float)

        history = history.sort_values("_step")
        history["_step_k"] = history["_step"] / 1000.0
        histories[run_name] = history
    return histories


def plot_group(name: str, histories: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots()
    for spec in PLOT_SPECS[name]:
        aliases = METRIC_ALIASES[spec["key"]]
        for run_name, history in histories.items():
            column = _first_available_column(history, aliases)
            if column is None:
                continue
            ax.plot(
                history["_step_k"],
                history[column],
                color=spec["color"],
                linestyle=RUN_STYLES[run_name],
                marker=RUN_MARKERS[run_name],
                label=f"{run_name} {spec['label']}",
            )

    ax.set_xlabel("Training step (×10³)")
    ax.set_ylabel(Y_LABELS[name])
    ax.set_title(TITLES[name])
    unique_legend(ax, **LEGEND_KW[name])
    fig.tight_layout()
    save_figure(fig, name)
    plt.close(fig)


def main() -> None:
    ensure_cache_dirs()
    apply_paper_style()
    histories = fetch_histories()
    for group in PLOT_SPECS:
        plot_group(group, histories)
    print("Saved extended pre-training figures to arc2025_paper/figures/")


if __name__ == "__main__":
    main()
