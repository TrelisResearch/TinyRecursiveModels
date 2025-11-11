# Additional Documentation (see README-original for the original README)

>[!TIP]
> Watch [this Youtube video](https://youtube.com/live/8RUzN9odRzI?feature=share) to better understand how to use this repo.

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

## Pretraining Ablations
### 100k epochs adding re-arc
```bash
run_name="pretrain_rearc_100k"
git pull && \
git switch main && \
find kaggle/combined -name '*.json.gz' -print0 | xargs -0 gunzip -f && \
uv run python3 -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/rearc-pretrain \
  --train-only-subsets rearc \
  --num-aug 0 \
  --subsets rearc && \
uv run python3 -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2-pretrain \
  --subsets concept training2 evaluation2 \
  --test-set-name evaluation2 && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain \
  data_paths=['data/rearc-pretrain','data/arc2-pretrain'] \
  arch=trm \
  +project_name='Arc2concept-aug-1000-ACT-torch' \
  +run_name="${run_name}" > pretrain_rearc_100k.log &
```
### No translations and 256 augs only

```bash
run_name="pretrain_100k_no_transl_256a"
git pull && \
uv run python3 -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --no-enable-train-translation \
  --subsets concept training2 evaluation2 \
  --test-set-name evaluation2 \
  --num-aug 256 && \
PYTHONUNBUFFERED=1 nohup uv run torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain \
  data_paths=['data/arc2concept-aug-1000'] \
  arch=trm \
  +run_name="${run_name}" > pretrain_100k_no_transl_256a.log &
```


## Final Runs
### 200k epochs on all data
```bash
run_name="pretrain_all_200k"
git switch meta && \
git pull && \
uv run python3 -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2-pretrain-final \
  --subsets concept tama training2 evaluation2train evaluation2eval \
  --test-set-name evaluation2eval && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain_all_8x \
  data_paths=['data/arc2-pretrain-final'] \
  arch=trm \
  +run_name="${run_name}" > pretrain_all_200k.log &
```
and then **continued pretraining:**
```bash
hf download Trelis/TRM-ARC-AGI-II-all-200k \
  step_769740 \
  --local-dir pretrained
```

```bash
run_name="pretrain_all_200k_ctd"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain_all_8x \
  data_paths=['data/arc2-pretrain-final'] \
  load_checkpoint="pretrained/step_769740" \
  +run_name="${run_name}" > pretrain_all_200k_ctd.log &
```

### 1M epochs on hard data
```bash
run_name="pretrain_hard_1M"
git switch meta && \
git pull && \
uv run python3 -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2-pretrain-final \
  --subsets traininghard evaluation2train evaluation2eval \
  --test-set-name evaluation2eval && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain_hard_8x \
  data_paths=['data/arc2-pretrain-final'] \
  arch=trm \
  +run_name="${run_name}" > pretrain_hard_1M.log &
```
### Post-training testing on L4s
```bash
uv pip install hf_transfer && \
git switch meta && \
git pull && \
hf download Sanjin2024/TinyRecursiveModels-ARC-AGI-1 \
  step_155718 \
  --local-dir pretrained && \
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2-aug-1000 \
  --subsets evaluation2 \
  --test-set-name evaluation2 \
  --num-aug 1000
```
**L4 Testing**
```bash
run_name="posttrain_L4_test"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  arch.puzzle_emb_len=16 \
  +run_name=${run_name} > posttrain_L4_test.log &
```
and add this to measure evaluation:
```bash
  eval_interval=50 \
  eval_max_augmentations=64 \
```

**H100 Testing**
```bash
cd ../workspace/TinyRecursiveModels-private && \
uv pip install hf_transfer && \
git switch meta && \
git pull && \
hf download Sanjin2024/TinyRecursiveModels-ARC-AGI-1 \
  step_155718 \
  --local-dir pretrained && \
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-theval2clean-aug-1000 \
  --subsets traininghard evaluation2clean \
  --test-set-name evaluation2clean \
  --num-aug 1000
```
**Standard LR**
```bash
run_name="posttrain_H100_5em5"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-theval2clean-aug-1000']" \
  data_paths_test="['data/arc-theval2clean-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  arch.puzzle_emb_len=16 \
  +run_name=${run_name} > posttrain_H100_5em5.log &
```
**2x LR**
```bash
run_name="posttrain_H100_1em4"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-theval2clean-aug-1000']" \
  data_paths_test="['data/arc-theval2clean-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  lr=1e-4 \
  puzzle_emb_lr=1e-2 \
  arch.puzzle_emb_len=16 \
  +run_name=${run_name} > posttrain_H100_1em4.log &
```

### Post-training at the midpoints
```bash
cd ../workspace/TinyRecursiveModels-private && \
uv pip install hf_transfer && \
git switch meta && \
git pull && \
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-evaluation2test-aug-1000 \
  --subsets evaluation2test \
  --test-set-name evaluation2test \
  --num-aug 1000
```
**all 200k midpoint**
```bash
run_name="posttrain_all_200k_midpoint"
hf download Trelis/TRM-ARC-AGI-II-all-200k \
  step_307898 \
  --local-dir pretrained && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-evaluation2test-aug-1000']" \
  data_paths_test="['data/arc-evaluation2test-aug-1000']" \
  load_checkpoint="pretrained/step_307898" \
  +run_name=${run_name} > posttrain_all_200k_midpoint.log &
```
**hard 1M midpoint**
```bash
run_name="posttrain_hard_1M_midpoint"
hf download Trelis/TRM-ARC-AGI-II-hard-1M \
  step_280350 \
  --local-dir pretrained && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-evaluation2test-aug-1000']" \
  data_paths_test="['data/arc-evaluation2test-aug-1000']" \
  load_checkpoint="pretrained/step_280350" \
  +run_name=${run_name} > posttrain_hard_1M_midpoint.log &
```
**aa1 model**
```bash
run_name="posttrain_aa1"
hf download Sanjin2024/TinyRecursiveModels-ARC-AGI-1 \
  step_155718 \
  --local-dir pretrained && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-evaluation2test-aug-1000']" \
  data_paths_test="['data/arc-evaluation2test-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  arch.puzzle_emb_len=16 \
  +run_name=${run_name} > posttrain_aa1.log &
```
**aa2 model**
```bash
run_name="posttrain_aa2"
hf download Trelis/TRM-ARC-AGI-II \
  step_723914 \
  --local-dir pretrained && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-evaluation2test-aug-1000']" \
  data_paths_test="['data/arc-evaluation2test-aug-1000']" \
  load_checkpoint="pretrained/step_723914" \
  arch.puzzle_emb_len=16 \
  +run_name=${run_name} > posttrain_aa2.log &
```
### Post-training at the midpoints, evening run
```bash
cd ../workspace/TinyRecursiveModels-private && \
uv pip install hf_transfer && \
git switch meta && \
git pull && \
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-evaluation2test-aug-1000 \
  --subsets evaluation2test \
  --test-set-name evaluation2test \
  --num-aug 1000
```
**all 200k**
```bash
run_name="posttrain_all_200k_midpoint_evening"
hf download Trelis/TRM-ARC-AGI-II-all-200k \
  step_384872 \
  --local-dir pretrained && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-evaluation2test-aug-1000']" \
  data_paths_test="['data/arc-evaluation2test-aug-1000']" \
  load_checkpoint="pretrained/step_384872" \
  +run_name=${run_name} > posttrain_all_200k_midpoint_evening.log &
```
**hard 1M**
```bash
run_name="posttrain_hard_1M_midpoint_evening"
hf download Trelis/TRM-ARC-AGI-II-hard-1M \
  step_358225 \
  --local-dir pretrained && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-evaluation2test-aug-1000']" \
  data_paths_test="['data/arc-evaluation2test-aug-1000']" \
  load_checkpoint="pretrained/step_358225" \
  +run_name=${run_name} > posttrain_hard_1M_midpoint_evening.log &
```
### Post-training - morning run
```bash
cd ../workspace/TinyRecursiveModels-private && \
uv pip install hf_transfer && \
git switch meta && \
git pull && \
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-evaluation2test-aug-1000 \
  --subsets evaluation2test \
  --test-set-name evaluation2test \
  --num-aug 1000
```
**all 200k**
```bash
run_name="posttrain_all_200k_morning"
hf download Trelis/TRM-ARC-AGI-II-all-200k \
  step_615792 \
  --local-dir pretrained && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-evaluation2test-aug-1000']" \
  data_paths_test="['data/arc-evaluation2test-aug-1000']" \
  load_checkpoint="pretrained/step_615792" \
  +run_name=${run_name} > posttrain_all_200k_morning.log &
```
**hard 1M**
```bash
run_name="posttrain_hard_1M_morning"
hf download Trelis/TRM-ARC-AGI-II-hard-1M \
  step_591850 \
  --local-dir pretrained && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-evaluation2test-aug-1000']" \
  data_paths_test="['data/arc-evaluation2test-aug-1000']" \
  load_checkpoint="pretrained/step_591850" \
  +run_name=${run_name} > posttrain_hard_1M_morning.log &
```

## Pre-Training Details
### High Epochs on High Quality Data
```bash
run_name="pretrain_500k"
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2-pretrain-500k \
  --subsets evaluation2A evaluation2B traininghard \
  --test-set-name evaluation2B && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain_8x \
  data_paths=['data/arc2-pretrain-500k'] \
  arch=trm \
  +run_name="${run_name}" > pretrain_500k.log &
```

### High Epochs on High Quality Data - no noise
```bash
run_name="pretrain_500k_noiseless"
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2-pretrain-500k \
  --subsets evaluation2A evaluation2B traininghard \
  --test-set-name evaluation2B && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain_8x \
  data_paths=['data/arc2-pretrain-500k'] \
  arch=trm_noiseless \
  +run_name="${run_name}" > pretrain_500k_noiseless.log &
```

### ctd pre-training synth
```bash
uv pip install hf_transfer
hf download Trelis/TRM-ARC-AGI-II \
  step_723914 \
  --local-dir pretrained
```

```bash
run_name="ctd_train_AAII_50k"
  uv run python -m dataset.build_arc_dataset \
    --input-file-prefix kaggle/combined/arc-agi \
    --output-dir data/arc2concept-aug-1000 \
    --subsets concept tama training2 evaluation2-80 evaluation2-40 \
    --test-set-name evaluation2-40 && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_ctd_pretrain \
  data_paths="['data/arc2concept-aug-1000']" \
  load_checkpoint=/workspace/TinyRecursiveModels-private/pretrained/step_723914 \
  +run_name="${run_name}" > ctd_train_AAII_50k.log &
```

### Slim Pre-training
```bash
run_name="pretrain_slim"
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2-pretrain \
  --subsets traininghard evaluation2 \
  --test-set-name evaluation2 && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain_slim \
  data_paths=['data/arc2-pretrain'] \
  arch=trm-slim \
  +run_name="${run_name}" > pretrain_slim.log &
```
### Synthetic Pipeline
#### Pre-training Synth
```bash
run_name="pretrain_slim_synth"
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2-pretrain \
  --subsets concept evaluation2-40 \
  --test-set-name evaluation2-40 && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain_slim \
  data_paths=['data/arc2-pretrain'] \
  arch=trm-slim \
  +run_name="${run_name}" > pretrain_slim_synth.log &
```

### ctd pre-training synth
```bash
uv pip install hf_transfer
hf download Trelis/TRM-pretrain-slim-synth \
  step_133377 \
  --local-dir pretrained
```

```bash
run_name="pretrain_slim_synth_2"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain_slim \
  load_checkpoint=/workspace/TinyRecursiveModels-private/pretrained/step_133377 \
  data_paths=['data/arc2-pretrain'] \
  arch=trm-slim-2 \
  +run_name="${run_name}" > pretrain_slim_synth_2.log &
```
Only add the following if not reloading the same checkpoint on the same GPU:
```bash
freeze_weights=True \
freeze_weights_epochs=6500 \
```

#### Post-training Synth
```bash
uv pip install hf_transfer
hf download Trelis/TRM-pretrain-slim-synth \
  step_113369 \
  --local-dir pretrained
```

```bash
run_name="posttrain_slim_synth-step_113369"
uv run python3 -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2-posttrain \
  --subsets concept evaluation2-80 \
  --test-set-name evaluation2-80 && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_posttrain_slim \
  load_checkpoint=/workspace/TinyRecursiveModels-private/pretrained/step_113369 \
  data_paths=['data/arc2-posttrain'] \
  arch=trm-slim \
  +run_name="${run_name}" > posttrain_slim_synth-step_113369.log &
```

### Production Pipeline
#### Production Pre-training
```bash
run_name="pretrain_slim_prod"
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2-pretrain \
  --subsets concept evaluation2 \
  --test-set-name concept && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain_slim \
  data_paths=['data/arc2-pretrain'] \
  arch=trm-slim \
  +run_name="${run_name}" > pretrain_slim_prod.log &
```

### From Scratch Ablations
#### 2x d_model
```bash
run_name="pretrain_2x"
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets concept training2 evaluation2 \
  --test-set-name evaluation2 && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain \
  arch=trm-double \
  +run_name="${run_name}" > pretrain_2x.log &
```

#### 30 augs
```bash
run_name="pretrain_30augs"
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets concept training2 evaluation2 \
  --test-set-name evaluation2 \
  --num-aug 30 && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain \
  eval_interval=2500 \
  +run_name="${run_name}" > pretrain_30augs.log &
```

#### Half cycles and ACT steps
```bash
run_name="pretrain_half_cycles"
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets concept training2 evaluation2 \
  --test-set-name evaluation2 && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain \
  arch=trm-half-cycles \
  eval_interval=2500 \
  +run_name="${run_name}" > pretrain_half_cycles.log &
```

### Original ctd pretraining
```bash
uv pip install hf_transfer
hf download Trelis/TRM-ARC-AGI-II \
  all_config.yaml losses.py step_723914 trm.py \
  --local-dir pretrained
```

```bash
uv pip install hf_transfer
hf download Trelis/TRM-ctd-pt-20ke \
  step_72477 \
  --local-dir pretrained
```

```bash
run_name="ctdpretrain_120k_ep"
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets concept training2 evaluation2 \
  --test-set-name evaluation2 && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain \
  load_checkpoint=/workspace/TinyRecursiveModels-private/pretrained/step_72477 \
  freeze_weights=True \
  freeze_weights_epochs=10000 \
  +run_name="${run_name}" > ctdpretrain_120k_ep.log &
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

### Push trained model to HF

Utility script at [./utils/push_to_hf.py](./utils/push_to_hf.py)

## Post-training
### Post-training ablation of pos embs AND 500k models on evaluation 2C (NOISELESS posttraining)
```bash
uv pip install hf_transfer
hf download Sanjin2024/TinyRecursiveModels-ARC-AGI-1 \
  step_155718 \
  --local-dir pretrained && \
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2C-aug-1000 \
  --subsets evaluation2C \
  --test-set-name evaluation2C \
  --num-aug 1000
```
**8x emb position**
```bash
run_name="posttrain_aa1_8emb_nlpt"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  arch.puzzle_emb_len=8 \
  arch=trm_noiseless \
  +run_name=${run_name} > posttrain_aa1_8emb_nlpt.log &
```
**aa2 model**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-ARC-AGI-II \
  step_723914 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2C-aug-1000 \
  --subsets evaluation2C \
  --test-set-name evaluation2C \
  --num-aug 1000
```
```bash
run_name="posttrain_aa2_8emb_nlpt"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_723914" \
  arch.puzzle_emb_len=8 \
  arch=trm_noiseless \
  +run_name=${run_name} > posttrain_aa2_8emb_nlpt.log &
```
**Noised 500k model**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-500k-noised \
  step_87633 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2C-aug-1000 \
  --subsets evaluation2C \
  --test-set-name evaluation2C \
  --num-aug 1000
```
```bash
run_name="posttrain_500k_noised_step_87633_nlpt"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_87633" \
  arch=trm_noiseless \
  +run_name=${run_name} > posttrain_500k_noised_step_87633_nlpt.log &
```
**Noiseless 500k model**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-500k-noiseless \
  step_87633 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2C-aug-1000 \
  --subsets evaluation2C \
  --test-set-name evaluation2C \
  --num-aug 1000
```
```bash
run_name="posttrain_500k_noiseless_step_87633_nlpt"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_87633" \
  arch=trm_noiseless \
  +run_name=${run_name} > posttrain_500k_noiseless_step_87633_nlpt.log &
```
**Noised 500k model - step_175266**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-500k-noised \
  step_175266 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2C-aug-1000 \
  --subsets evaluation2C \
  --test-set-name evaluation2C \
  --num-aug 1000
```
```bash
run_name="posttrain_500k_noised_step_175266_nlpt"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_175266" \
  arch=trm_noiseless \
  +run_name=${run_name} > posttrain_500k_noised_step_175266_nlpt.log &
```
### Post-training ablation of pos embs AND 500k models on evaluation 2C (noised posttraining)
```bash
uv pip install hf_transfer
hf download Sanjin2024/TinyRecursiveModels-ARC-AGI-1 \
  step_155718 \
  --local-dir pretrained && \
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2C-aug-1000 \
  --subsets evaluation2C \
  --test-set-name evaluation2C \
  --num-aug 1000
```
**Single emb position**
```bash
run_name="posttrain_aa1_1emb"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  +run_name=${run_name} > posttrain_aa1_1emb.log &
```
**8x emb position**
```bash
run_name="posttrain_aa1_8emb"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  arch.puzzle_emb_len=8 \
  +run_name=${run_name} > posttrain_aa1_8emb.log &
```
**aa2 model**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-ARC-AGI-II \
  step_723914 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2C-aug-1000 \
  --subsets evaluation2C \
  --test-set-name evaluation2C \
  --num-aug 1000
```
```bash
run_name="posttrain_aa2_8emb"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_723914" \
  arch.puzzle_emb_len=8 \
  +run_name=${run_name} > posttrain_aa2_8emb.log &
```
**Noised 500k model**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-500k-noised \
  step_87633 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2C-aug-1000 \
  --subsets evaluation2C \
  --test-set-name evaluation2C \
  --num-aug 1000
```
```bash
run_name="posttrain_500k_noised_step_87633"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_87633" \
  +run_name=${run_name} > posttrain_500k_noised_step_87633.log &
```
**Noiseless 500k model**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-500k-noiseless \
  step_87633 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2C-aug-1000 \
  --subsets evaluation2C \
  --test-set-name evaluation2C \
  --num-aug 1000
```
```bash
run_name="posttrain_500k_noiseless_step_87633"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_87633" \
  +run_name=${run_name} > posttrain_500k_noiseless_step_87633.log &
```
**Noised 500k model - step_175266**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-500k-noised \
  step_175266 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2C-aug-1000 \
  --subsets evaluation2C \
  --test-set-name evaluation2C \
  --num-aug 1000
```
```bash
run_name="posttrain_500k_noised_step_175266"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_175266" \
  +run_name=${run_name} > posttrain_500k_noised_step_175266.log &
```

**Noised 500k model - step_175266 - double epochs**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-500k-noised \
  step_175266 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2C-aug-1000 \
  --subsets evaluation2C \
  --test-set-name evaluation2C \
  --num-aug 1000
```
```bash
run_name="posttrain_500k_noised_step_175266_2xe"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_175266" \
  epochs=25000 \
  +run_name=${run_name} > posttrain_500k_noised_step_175266_2xe.log &
```
**Noised 500k model - step_87633 - double epochs**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-500k-noised \
  step_87633 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2C-aug-1000 \
  --subsets evaluation2C \
  --test-set-name evaluation2C \
  --num-aug 1000
```
```bash
run_name="posttrain_500k_noised_step_87633_2xe"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_87633" \
  epochs=25000 \
  +run_name=${run_name} > posttrain_500k_noised_step_87633_2xe.log &
```
**Noiseless 500k model - 2x epochs**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-500k-noiseless \
  step_87633 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2C-aug-1000 \
  --subsets evaluation2C \
  --test-set-name evaluation2C \
  --num-aug 1000
```
```bash
run_name="posttrain_500k_noiseless_step_87633_2x"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2C-aug-1000']" \
  data_paths_test="['data/arc-eval2C-aug-1000']" \
  load_checkpoint="pretrained/step_87633" \
  epochs=25000 \
  +run_name=${run_name} > posttrain_500k_noiseless_step_87633_2x.log &
```

### Evaluation 2B Comparison
**Noised 500k model - step_175266**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-500k-noised \
  step_175266 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2B-aug-1000 \
  --subsets evaluation2B \
  --test-set-name evaluation2B \
  --num-aug 1000
```

```bash
run_name="posttrain_500k_noised_step_175266"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2B-aug-1000']" \
  data_paths_test="['data/arc-eval2B-aug-1000']" \
  load_checkpoint="pretrained/step_175266" \
  +run_name=${run_name} > posttrain_500k_noised_step_175266.log &
```
**Noiseless 500k model**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-500k-noiseless \
  step_175266 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2B-aug-1000 \
  --subsets evaluation2B \
  --test-set-name evaluation2B \
  --num-aug 1000
```
```bash
run_name="posttrain_500k_noiseless_step_87633"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2B-aug-1000']" \
  data_paths_test="['data/arc-eval2B-aug-1000']" \
  load_checkpoint="pretrained/step_87633" \
  +run_name=${run_name} > posttrain_500k_noiseless_step_87633.log &
```

**aa2 model**
```bash
uv pip install hf_transfer
hf download Trelis/TRM-ARC-AGI-II \
  step_723914 \
  --local-dir pretrained
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2B-aug-1000 \
  --subsets evaluation2B \
  --test-set-name evaluation2B \
  --num-aug 1000
```
```bash
run_name="posttrain_aa2_8emb"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2B-aug-1000']" \
  data_paths_test="['data/arc-eval2B-aug-1000']" \
  load_checkpoint="pretrained/step_723914" \
  arch.puzzle_emb_len=8 \
  +run_name=${run_name} > posttrain_aa2_8emb.log &
```
### Chunked Post-training
`scripts/chunked_posttrain.py` automates the full pipeline: dataset splitting, augmented builds, sequential training, submission collection, and merged pass@k scoring.

Chunks of 38
```bash
nohup uv run python -m scripts.chunked_posttrain \
  --subset evaluation2clean \
  --chunk-size 38 \
  --enable-wandb \
  --wandb-project arc-eval2clean \
  --num-aug 1000 \
  --overwrite-splits \
  --rebuild-datasets \
  --overwrite-submissions \
  > chunked_posttrain.out 2>&1 &
tail -f chunked_posttrain.out            # orchestrator progress
tail -f logs/posttrain_eval2cleanA.log   # per-chunk training log (A/B/C)
```
Merged submission metrics (no freezing of embeddings):
  pass@1: 0.0219
  pass@2: 0.0307

And with freezing of embeddings:


One single chunk
```bash
nohup uv run python -m scripts.chunked_posttrain \
  --subset evaluation2clean \
  --chunk-size 120 \
  --enable-wandb \
  --wandb-project arc-eval2clean \
  --num-aug 1000 \
  --skip-download \
  --overwrite-splits \
  --rebuild-datasets \
  --overwrite-submissions \
  > chunked_posttrain.out 2>&1 &
tail -f chunked_posttrain.out            # orchestrator progress
tail -f logs/posttrain_eval2cleanA.log   # per-chunk training log (A/B/C)
```
Merged submission metrics (no freezing of embeddings):
  pass@2: 0.02193

with freezing for first 3125 steps:
  pass@1: 0.0219
  pass@2: 0.0307

- Automatically downloads `Sanjin2024/TinyRecursiveModels-ARC-AGI-1/step_155718` (skip with `--skip-download` if already cached).
- Splits `kaggle/combined/arc-agi_evaluation2clean_{challenges,solutions}.json` into 38-puzzle chunks (`evaluation2cleanA/B/C` by default) and stores the new JSONs alongside the originals.
- Builds `data/arc-eval2clean{A,B,C}-aug-1000` with `dataset.build_arc_dataset`.
- Launches three post-train runs (`posttrain_eval2clean{A,B,C}`) and logs to `logs/<run>.log`.
- Collects each `submission.json`, copies them to `submissions/`, merges into `submissions/eval2clean_merged_submission.json`, and prints aggregated pass@k (default pass@1/pass@2).

Useful flags:
- `--skip-download`, `--skip-datasets`, `--skip-train`, `--skip-merge` to reuse previous artifacts.
- `--extra-override` to pass additional Hydra overrides (repeat flag as needed).
- `--no-copy-submissions` to leave submissions in their checkpoint folders.
- `--wandb-project`, `--wandb-entity` to pin all chunk runs under a single W&B project/dashboard.

### Post-training on TAMA
```bash
uv run python3 -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/tama \
  --subsets tama \
  --test-set-name tama
```

**aa1 model**
```bash
run_name="postrain_aa1"
uv pip install hf_transfer
hf download Sanjin2024/TinyRecursiveModels-ARC-AGI-1 \
  step_155718 \
  --local-dir pretrained && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/tama']" \
  data_paths_test="['data/tama']" \
  load_checkpoint="pretrained/step_155718" \
  +run_name=${run_name} > postrain_aa1.log &
```

**aa2 model**
```bash
run_name="postrain_aa2_ctd_115343"
uv pip install hf_transfer
uv run hf download Trelis/TRM-AAII-ctd-50k \
  step_115343 \
  --local-dir pretrained && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/tama']" \
  data_paths_test="['data/tama']" \
  load_checkpoint="pretrained/step_115343" \
  +run_name=${run_name} > postrain_aa2_ctd_115343.log &
```
**aa2 slim model from oct 27th**
```bash
run_name="postrain_aa2_slim_prod_43568"
uv pip install hf_transfer
hf download Trelis/TRM-slim-prod \
  step_43568 \
  --local-dir pretrained && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain_slim \
  data_paths="['data/tama']" \
  data_paths_test="['data/tama']" \
  load_checkpoint="pretrained/step_43568" \
  arch.num_heads=4 \
  +run_name=${run_name} > postrain_aa2_slim_prod_43568.log &
```

**aa2 slim model from oct 28th**
```bash
run_name="postrain_aa2_slim_the2_44072"
uv pip install hf_transfer
hf download Trelis/TRM-slim-the2 \
  step_44072 \
  --local-dir pretrained && \
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain_slim \
  data_paths="['data/tama']" \
  data_paths_test="['data/tama']" \
  load_checkpoint="pretrained/step_44072" \
  +run_name=${run_name} > postrain_aa2_slim_prod_44072.log &
```

### evaluation2B dataset on aa1 model meta-trained on evaluation2A
```bash
uv pip install hf_transfer
hf download Trelis/TRM-meta-1x \
  step_3330 \
  --local-dir pretrained && \
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2-aug-1000 \
  --subsets evaluation2B \
  --test-set-name evaluation2B \
  --num-aug 1000
```

```bash
run_name="posttrain_meta_1x"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_3330" \
  +run_name=${run_name} > posttrain_meta_1x.log &
```

### eval2 dataset on aa1 model
- **Download checkpoint:** Start from the published ARC checkpoint (example below) so the adapters can piggyback on the same architecture:
```bash
uv pip install hf_transfer
hf download Sanjin2024/TinyRecursiveModels-ARC-AGI-1 \
  step_155718 \
  --local-dir pretrained && \
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2-aug-1000 \
  --subsets evaluation2clean1 \
  --test-set-name evaluation2clean1 \
  --num-aug 1000
```
- **Single Task**:
```bash
run_name="posttrain_single_long"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain_long \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  +run_name=${run_name} > posttrain_single_long.log &
```
- **8x Tasks**:
```bash
uv pip install hf_transfer
hf download Sanjin2024/TinyRecursiveModels-ARC-AGI-1 \
  step_155718 \
  --local-dir pretrained && \
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2-aug-1000 \
  --subsets evaluation2clean8 \
  --test-set-name evaluation2clean8 \
  --num-aug 1000
```
```bash
run_name="posttrain_8x"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  +run_name=${run_name} > posttrain_8x.log &
```
- **38x Tasks**:
```bash
uv pip install hf_transfer
hf download Sanjin2024/TinyRecursiveModels-ARC-AGI-1 \
  step_155718 \
  --local-dir pretrained && \
uv run python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc-eval2-aug-1000 \
  --subsets evaluation2-38 \
  --test-set-name evaluation2-38 \
  --num-aug 1000
```
```bash
run_name="posttrain_38x"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  +run_name=${run_name} > posttrain_38x.log &
```

- **Puzz Dropout**:
```bash
run_name="posttrain_aa1_aa2e_puzz_drp"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  arch.puzzle_emb_dropout=0.1 \
  +run_name=${run_name} > posttrain_aa1_aa2e_puzz_drp.log &
```

- **Input Grid Dropout**:
```bash
run_name="posttrain_aa1_aa2e_grid_drp"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  arch.grid_token_dropout=0.05 \
  +run_name=${run_name} > posttrain_aa1_aa2e_grid_drp.log &
```
- **Grid Noise**:
```bash
run_name="posttrain_aa1_aa2e_grid_noise"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  arch.grid_noise_prob=0.05 \
  arch.grid_noise_fraction=0.02 \
  +run_name=${run_name} > posttrain_aa1_aa2e_grid_noise.log &
```


- **Full tuning - 4x halt max steps**:
```bash
run_name="posttrain_aa1_aa2e_4x_halt"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  eval_max_augmentations=64 \
  halt_max_steps_eval=64 \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  +run_name=${run_name} > posttrain_aa1_aa2e_4x_halt.log &
```

- **4x LR**:
```bash
run_name="posttrain_aa1_aa2e_4x_LR"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  lr=2.8e-4 \
  puzzle_emb_lr=2.8e-2 \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  +run_name=${run_name} > posttrain_aa1_aa2e_4x_LR.log &
```

- **0.1x WD**:
```bash
run_name="posttrain_aa1_aa2e_0.1x_WD"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  weight_decay=0.01 \
  puzzle_emb_weight_decay=0.01 \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  +run_name=${run_name} > posttrain_aa1_aa2e_0.1x_WD.log &
```

- **Full tuning - mean init**:
```bash
run_name="posttrain_aa1_aa2e_feq"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  puzzle_emb_reinit_strategy="mean" \
  freeze_weights=True \
  freeze_weights_epochs=3125 \
  +run_name=${run_name} > posttrain_aa1_aa2e_feq.log &
```

```bash
run_name="posttrain_aa1_aa2e_96bsz"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  puzzle_emb_reinit_strategy="mean" \
  freeze_weights=True \
  freeze_weights_epochs=3125 \
  global_batch_size=96 \
  eval_global_batch_size=384 \
  lr=0.125e-4 \
  +run_name=${run_name} > posttrain_aa1_aa2e_96bsz.log &
```

```bash
run_name="posttrain_aa1_aa2e_384bsz_4ka"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  puzzle_emb_reinit_strategy="mean" \
  freeze_weights=True \
  freeze_weights_epochs=3125 \
  +run_name=${run_name} > posttrain_aa1_aa2e_384bsz_4ka.log &
```
- **Full tuning - normal init**:
```bash
run_name="posttrain_aa1_tem2_norm"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  puzzle_emb_reinit_strategy="normal" \
  +run_name=${run_name} > posttrain_aa1_tem2_norm.log &
```
- **Freeze Embeddings**:
```bash
run_name="posttrain_aa1_tem2_fe"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-tem2-aug-1000']" \
  data_paths_test="['data/arc-tem2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  freeze_weights=True \
  puzzle_emb_reinit_strategy="normal" \
  +run_name=${run_name} > posttrain_aa1_tem2_fe.log &
```
- **Freeze Embeddings for first half**:
```bash
run_name="posttrain_aa1_tem2_feh"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-tem2-aug-1000']" \
  data_paths_test="['data/arc-tem2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  freeze_weights=True \
  freeze_weights_epochs=6250 \
  puzzle_emb_reinit_strategy="mean" \
  +run_name=${run_name} > posttrain_aa1_tem2_feh.log &
```

```bash
run_name="posttrain_aa1_tem2_feh_norm"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_posttrain \
  data_paths="['data/arc-tem2-aug-1000']" \
  data_paths_test="['data/arc-tem2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  freeze_weights=True \
  freeze_weights_epochs=6250 \
  puzzle_emb_reinit_strategy="normal" \
  +run_name=${run_name} > posttrain_aa1_tem2_feh_norm.log &
```
<!-- - **Run LoRA tuning:** Switch to the LoRA config, point at the freshly built data, and load the base checkpoint:
```bash
run_name="posttrain_aa1_aa2e_lora"
PYTHONUNBUFFERED=1 nohup torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  --config-name cfg_pretrain_lora \
  data_paths="['data/arc-eval2-aug-1000']" \
  data_paths_test="['data/arc-eval2-aug-1000']" \
  load_checkpoint="pretrained/step_155718" \
  +run_name=${run_name} > posttrain_aa1_aa2e_lora.log &
``` -->
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
