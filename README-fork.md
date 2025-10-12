# Additional Documentation

## Runpod One-click Template
Option to startup up a pod with [this template](https://console.runpod.io/deploy?template=1urgylpi1x&ref=jmfkcdio). Note: This is a Trelis template and is an affiliate link.

## Container Setup

For containerized deployments, use the included `container-onstart.sh` script:

```bash
# Set environment variables (optional but recommended)
export WANDB_API_KEY="your-wandb-key"
export GITHUB_PAT="your-github-token"  # if repo is private
export GIT_USER_NAME="Your Name"
export GIT_USER_EMAIL="your@email.com"

# Run the container with the startup script
# The script will:
# - Install system dependencies
# - Clone/update the TrelisResearch/TinyRecursiveModels repo
# - Set up Python 3.10 venv with uv
# - Install PyTorch with CUDA 12.6 support
# - Install all dependencies including flash-attn and adam-atan2
# - Auto-login to wandb if WANDB_API_KEY is set
```

## Training Details for ARC-AGI-2

### How to Run

**1. Prepare the dataset:**
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```

**2. Run training (4 GPUs):**
```bash
run_name="pretrain_att_arc2concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc2concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

**Runtime:** ~3 days on 4x H100 GPUs

### Evaluation Schedule

Evaluation runs periodically based on the `eval_interval` config parameter:
- Training is split into iterations, each running for `eval_interval` epochs (default: 10,000)
- After each training iteration, evaluation runs
- Only starts after `min_eval_interval` iterations have passed (default: 0)
- Checkpoints saved at each evaluation if `checkpoint_every_eval: True`

### Metrics Logged to Weights & Biases

#### Training Metrics (logged every step)
- **`train/count`** - Number of valid examples processed
- **`train/accuracy`** - Token-level accuracy (correct predictions / total tokens)
- **`train/exact_accuracy`** - Sequence-level accuracy (entire grid correct)
- **`train/q_halt_accuracy`** - How well the model predicts when to stop reasoning
- **`train/steps`** - Average number of recursive reasoning steps taken
- **`train/lm_loss`** - Language modeling loss (stablemax cross-entropy on predictions)
- **`train/q_halt_loss`** - Binary cross-entropy for halt prediction
- **`train/lr`** - Current learning rate

#### Evaluation Metrics (logged every eval_interval)
- **`ARC/pass@1`** - Accuracy with best prediction only
- **`ARC/pass@2`** - Accuracy if correct answer in top-2 predictions
- **`ARC/pass@5`** - Accuracy if correct answer in top-5 predictions
- **`ARC/pass@10`** - Accuracy if correct answer in top-10 predictions
- **`ARC/pass@100`** - Accuracy if correct answer in top-100 predictions
- **`ARC/pass@1000`** - Accuracy if correct answer in top-1000 predictions

Plus the same training metrics computed on the eval set.

### How Evaluation Works

1. **Inference**: Model runs recursively until convergence (`all_finish=True`)
2. **Voting**: Predictions from augmented puzzle versions are aggregated
3. **Ranking**: Predictions ranked by halt confidence scores (`q_halt_logits`)
4. **Submission**: Top-K predictions saved in ARC-AGI competition format (`submission.json`)

### Key Configuration (config/cfg_pretrain.yaml)
- `global_batch_size: 768`
- `epochs: 100000`
- `eval_interval: 10000`
- `lr: 1e-4`
- `puzzle_emb_lr: 1e-2`
- `checkpoint_every_eval: True`
- Loss function: `stablemax_cross_entropy` (more stable than softmax for this task)