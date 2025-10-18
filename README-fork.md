# Additional Documentation

## Runpod One-click Template
Option to startup up a pod with [this template](https://console.runpod.io/deploy?template=1urgylpi1x&ref=jmfkcdio)). Note: This is a Trelis template and is an affiliate link.

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

**2. Run training (8 GPUs):**

```bash
run_name="pretrain_att_arc2concept"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
+run_name=${run_name}
```

**Runtime:** ~3 days on 8x H100 GPUs

*Other options:*
To test it on a small batch size, change global_batch_size:

```bash
sed -i 's/^global_batch_size:.*/global_batch_size: 16/' config/cfg_pretrain.yaml
```

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

### Push trained model to HF

Utility script at [./utils/push_to_hf.py](./utils/push_to_hf.py)

## LoRA Fine-Tuning - Manual Dataset
- **Download checkpoint:** Start from the published ARC checkpoint (example below) so the adapters can piggyback on the same architecture:
```bash
uv pip install hf_transfer
hf download Trelis/TRM-ARC-AGI-II \
  all_config.yaml losses.py step_723914 trm.py \
  --local-dir pretrained
```
- **Build your adaptation set:** Re-use the ARC builder to target the tasks you want to adapt on (e.g., just the evaluation puzzles) while keeping their test grids for scoring:
  The LoRA config already targets `data/arc-manual-eval-aug-1000`; rebuild it only if you still need the dataset:
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-manual-eval-aug-1000 \
  --subsets manual_evaluation \
  --test-set-name manual_evaluation \
  --num-aug 1000
```
- **Run LoRA tuning:** Switch to the LoRA config, point at the freshly built data, and load the base checkpoint:
```bash
run_name="lora_manual_Trelis_wgtd"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_pretrain_lora \
  +run_name=${run_name} > lora-manual.log &
```
  This attaches rank-1 adapters (alpha 16), keeps embeddings trainable (`puzzle_emb_lr: 1e-2`), leaves EMA off, and logs submissions every eval pass. No merge step is required for inference—just keep the base checkpoint alongside the LoRA state; merge the low-rank deltas only if you need dense weights.

## LoRA Fine-Tuning - Eval Dataset
- **Download checkpoint:** Start from the published ARC checkpoint (example below) so the adapters can piggyback on the same architecture:
```bash
uv pip install hf_transfer
hf download Trelis/TRM-ARC-AGI-II \
  all_config.yaml losses.py step_723914 trm.py \
  --local-dir pretrained
```
- **Build your adaptation set:** Re-use the ARC builder to target the tasks you want to adapt on (e.g., just the evaluation puzzles) while keeping their test grids for scoring:
  The LoRA config already targets `data/arc-manual-eval-aug-1000`; rebuild it only if you still need the dataset:
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2-aug-1000 \
  --subsets evaluation2 \
  --test-set-name evaluation2 \
  --num-aug 1000
```
- **Run LoRA tuning:** Switch to the LoRA config, point at the freshly built data, and load the base checkpoint:
```bash
run_name="lora_manual_Trelis_eval2"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_pretrain_lora \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  +run_name=${run_name} > lora-manual.log &
```

## LoRA testing for Kaggle
- **Download checkpoint:** Start from the published ARC checkpoint (example below) so the adapters can piggyback on the same architecture:
```bash
uv pip install hf_transfer
hf download Trelis/TRM-ARC-AGI-II \
  all_config.yaml losses.py step_723914 trm.py \
  --local-dir pretrained
```
- **Build your adaptation set:** Re-use the ARC builder to target the tasks you want to adapt on (e.g., just the evaluation puzzles) while keeping their test grids for scoring:
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2-test-aug-1000 \
  --subsets test \
  --test-set-name test \
  --num-aug 1000
```
- **Run training:** 
```bash
run_name="kaggle-lora-test-5ke-1ka"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py --config-name cfg_kaggle_lora load_checkpoint=pretrained/step_723914 +run_name=${run_name} > lora.log &
```

## Continued Pretraining on ARC Evaluation Tasks
> Scores 6.5% pass@2 on manual tasks, with aa2 pre-trained model.
- **Download checkpoint:** Grab the published ARC checkpoint (e.g. `Sanjin2024/TinyRecursiveModels-ARC-AGI-2`) with `git lfs` or `huggingface-cli`. Keep the accompanying `all_config.yaml` handy so you can mirror the architecture, optimizer, and embedding hyperparameters that were used to produce the weights. As follows:
```bash
huggingface-cli download --repo-type model Sanjin2024/TinyRecursiveModels-ARC-AGI-2 --local-dir pretrained
```
- **Match the config:** The checkpoint expects the same architecture (`arch=trm`, hidden size 512, etc.). The dedicated post-train config at `config/cfg_emb_posttrain.yaml` keeps the architecture fixed while shrinking runtime settings—`global_batch_size: 96`, `epochs: 10000`, `lr: 1.25e-5`, `puzzle_emb_lr: 1.25e-3`, warmup 200 steps, eval every 1000 epochs, `freeze_weights: True` so only the puzzle embeddings update, and `ema: False` to avoid maintaining a duplicate embedding table. Adjust those numbers if you need different scaling, but do not change the structural fields unless you know they match the checkpoint.
- **Rebuild evaluation-only data:** Use the ARC builder to generate a dataset that contains only the evaluation puzzles you want to adapt on while holding out their test grids for evaluation. Both `arc-agi_evaluation_challenges.json` and `arc-agi_manual_evaluation_challenges.json` follow the same schema (`{"train": [...], "test": [...]}` with `test` entries containing just `"input"`), so the builder can treat their `test` examples as the evaluation split:
```bash
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-manual-eval-aug-1000 \
  --subsets manual_evaluation \
  --test-set-name manual_evaluation \
  --num-aug 1000
```
  Adjust `--num-aug` downward if you do not want a million-row embedding table.
- **Launch continued training:** Point `pretrain.py` at the new dataset and supply the checkpoint using the post-train config. Example (single node, 4 GPUs):
```bash
run_name="post-train-embs-manual"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_emb_posttrain \
  +run_name=${run_name} \
  load_checkpoint=/workspace/TinyRecursiveModels/pretrained/step_217602 > pt-manual.log &
```
  Override knobs inline (e.g. `lr=...`) if you need to deviate from `cfg_emb_posttrain.yaml`; keep the architecture consistent so the checkpoint loads cleanly. If you change the effective batch size, scale the learning rates accordingly.
