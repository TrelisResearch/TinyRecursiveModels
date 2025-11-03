# W&B Data Fetching Scripts

This folder contains scripts for fetching data from Weights & Biases and generating plots for the paper.

## Scripts

### `generate_embedding_plots.py`

Fetches embedding cosine similarity metrics from the `pretrain_all_200k` run and generates two plots:

1. **embedding_cosine_across.pdf/png**: Cosine similarity across base tasks
   - Evaluation: `all.embedding_cosine`
   - Training: `train/embedding_cosine`

2. **embedding_cosine_within.pdf/png**: Cosine similarity within task variants
   - Evaluation: `all.embedding_cosine_within_task`
   - Training: `train/embedding_cosine_within_task`

**W&B Project**: `trelis/Arc2-pretrain-final-ACT-torch`
**Run**: `pretrain_all_200k` (ID: 9bp7agqh)

**Usage**: From the repository root, run:
```bash
uv run python arc2025_paper/wandb-scripts/generate_embedding_plots.py
```

### `list_runs.py`

Utility script to list all runs in the `Arc2-pretrain-final-ACT-torch` project with their names and IDs.

**Usage**:
```bash
uv run python arc2025_paper/wandb-scripts/list_runs.py
```

## Output

Plots are saved to `arc2025_paper/figures/` in both PDF and PNG formats (300 DPI).

All plots use a grayscale, journal-style format suitable for black and white printing, matching the style of other plots in the paper.
