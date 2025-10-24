# ── 3b · byobu ───────────────────────────────────────────────────────────
DEBIAN_FRONTEND=noninteractive apt-get install -y byobu
byobu-enable

# ── 4 · uv + venv ───────────────────────────────────────────────────────────
python -m pip install --upgrade --no-cache-dir uv
if [ -d .venv ]; then
  echo "[info] .venv already exists; activate it"
else
  uv venv .venv -p 3.10
fi
. .venv/bin/activate
grep -q "TinyRecursiveModels-private/.venv/bin/activate" /root/.bashrc || \
  echo "source /workspace/TinyRecursiveModels-private/.venv/bin/activate" >> /root/.bashrc

# Upgrade pip, wheel, setuptools (per README, using uv)
uv pip install --upgrade pip wheel setuptools

# ── 5 · CUDA 12.6 PyTorch nightly (as specified in README) ──────────────────
export PYTORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu126"
uv pip install --upgrade --extra-index-url "$PYTORCH_INDEX_URL" "torch==2.8.*"
uv pip install --pre --upgrade --no-cache-dir torchvision torchaudio --index-url "$PYTORCH_INDEX_URL" 

# ── 6 · Build helpers ───────────────────────────────────────────────────────
uv pip install --upgrade --no-cache-dir packaging ninja wheel setuptools setuptools-scm numba huggingface_hub

# ── 7 · FlashAttention build dependencies ───────────────────────────────────
uv pip install --no-cache-dir psutil

# ── 8 · FlashAttention (for efficient attention) ────────────────────────────
# from https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3
uv pip install torch==2.8.0
uv pip install \
https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# ── 9 · Project deps ────────────────────────────────────────────────────────
[ -f requirements.txt ] && uv pip install --no-cache-dir -r requirements.txt

# ── 10 · adam-atan2 optimizer (required by TRM) ─────────────────────────────
uv pip install --no-cache-dir --no-build-isolation adam-atan2-pytorch # NOTE: not adam-atan2, see https://github.com/sapientinc/HRM/issues/25#issuecomment-3162492484

# ── 11 · wandb login (if API key provided) ──────────────────────────────────
if [ -n "${WANDB_API_KEY}" ]; then
  wandb login "$WANDB_API_KEY"
  echo "[info] Logged into wandb"
else
  echo "[warn] WANDB_API_KEY not set; skip wandb login (you can run "wandb login" manually later)"
fi
