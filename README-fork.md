# Additional Documentation

## Runpod One-click Template
Option to startup up a pod with [this template](https://console.runpod.io/deploy?template=tduftocnct&ref=jmfkcdio)). Note: This is a Trelis template and is an affiliate link.

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

## Training Details

### Original ctd pretraining
```bash
uv pip install hf_transfer
hf download Trelis/TRM-ARC-AGI-II \
  all_config.yaml losses.py step_723914 trm.py \
  --local-dir pretrained
```

```bash
run_name="ctdpretrain_original"
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets concept training2 evaluation2 \
  --test-set-name evaluation2 && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain \
  load_checkpoint=/workspace/TinyRecursiveModels/pretrained/step_723914 \
  +run_name="${run_name}" > ctdpretrain_original.log &
```


### Base Evaluation Training Hard
```bash
run_name="pretrain_base"
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets traininghard evaluation2 \
  --test-set-name evaluation2 && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain \
  +run_name="${run_name}" > pretrain_base.log &
```
or for L4:
```bash
run_name="pretrain_base_l4"
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2ethard-aug-1000 \
  --subsets traininghard evaluation2 \
  --test-set-name evaluation2 && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain_l4 \
  +run_name="${run_name}" > pretrain_base_l4.log &
```

### Base AA1 Concept Manual
```bash
run_name="pretrain_base"
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arccm-aug-1000 \
  --subsets concept manual_evaluation \
  --test-set-name concept \
  --test-set-name2 manual_evaluation && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain \
  +run_name="${run_name}" > pretrain_base.log &
```

### Post train evalconcept on AA1 Concept Manual
```bash
uv pip install hf_transfer
hf download Trelis/TRM-ARC-AGI-II \
  all_config.yaml losses.py step_723914 trm.py \
  --local-dir pretrained
```

```bash
run_name="posttrain_base"
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arccm-aug-1000 \
  --subsets concept manual_evaluation \
  --test-set-name concept \
  --test-set-name2 manual_evaluation && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain \
  load_checkpoint=/workspace/TinyRecursiveModels/pretrained/step_723914 \
  +run_name="${run_name}" > posttrain_base.log &
```

And to only train embeds
```bash
run_name="posttrain_base_embs"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain \
  load_checkpoint=/workspace/TinyRecursiveModels/pretrained/step_723914 \
  freeze_weights=True \
  +run_name="${run_name}" > posttrain_base_embs.log &
```
And to only train embeds for first half:
```bash
run_name="posttrain_base_embs_start"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain \
  load_checkpoint=/workspace/TinyRecursiveModels/pretrained/step_723914 \
  freeze_weights=True \
  freeze_weights_epochs=4000 \
  +run_name="${run_name}" > posttrain_base_embs_start.log &
```

### Push trained model to HF

Utility script at [./utils/push_to_hf.py](./utils/push_to_hf.py)

## Eval dataset on aa1 model
- **Download checkpoint:** Start from the published ARC checkpoint (example below) so the adapters can piggyback on the same architecture:
```bash
uv pip install hf_transfer
hf download Sanjin2024/TinyRecursiveModels-ARC-AGI-1 \
  all_config.yaml losses.py step_155718 trm.py \
  --local-dir pretrained
```
- **Build your adaptation set:** Re-use the ARC builder to target the tasks you want to adapt on (e.g., just the evaluation puzzles) while keeping their test grids for scoring:
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2-aug-1000 \
  --subsets evaluation2 \
  --test-set-name evaluation2 \
  --num-aug 1000
```
- **Full tuning**:
```bash
run_name="posttrain_aa1_aa2e"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_pretrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  +run_name=${run_name} > posttrain_aa1_aa2e.log &
```
- **Freeze Embeddings**:
```bash
run_name="posttrain_aa1_aa2e_fe"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_pretrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  freeze_weights=True \
  +run_name=${run_name} > posttrain_aa1_aa2e_fe.log &
```
- **Freeze Embeddings for first half**:
```bash
run_name="posttrain_aa1_aa2e_feh"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_pretrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  freeze_weights=True \
  freeze_weights_epochs=6250 \
  +run_name=${run_name} > posttrain_aa1_aa2e_feh.log &
```
- **Run LoRA tuning:** Switch to the LoRA config, point at the freshly built data, and load the base checkpoint:
```bash
run_name="posttrain_aa1_aa2e_lora"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_pretrain_lora \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  +run_name=${run_name} > posttrain_aa1_aa2e_lora.log &
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
