
# ── 4 · uv + venv ───────────────────────────────────────────────────────────
python -m pip install --upgrade --no-cache-dir uv
if [ -d .venv ]; then
  echo "[info] .venv already exists; activate it"
else
  uv venv .venv -p 3.10
fi
. .venv/bin/activate
grep -q "TinyRecursiveModels/.venv/bin/activate" /root/.bashrc || \
  echo "source /workspace/TinyRecursiveModels/.venv/bin/activate" >> /root/.bashrc

# Upgrade pip, wheel, setuptools (per README, using uv)
uv pip install --upgrade pip wheel setuptools

# ── 5 · CUDA 12.6 PyTorch nightly (as specified in README) ──────────────────
export PYTORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu126"
uv pip install --pre --upgrade --no-cache-dir torchvision torchaudio --index-url "$PYTORCH_INDEX_URL" 
uv pip install --upgrade --extra-index-url "$PYTORCH_INDEX_URL" "torch==2.8.*"

# ── 6 · Build helpers ───────────────────────────────────────────────────────
uv pip install --upgrade --no-cache-dir packaging ninja wheel setuptools setuptools-scm numba huggingface_hub

# ── 7 · FlashAttention build dependencies ───────────────────────────────────
uv pip install --no-cache-dir psutil

# ── 8 · FlashAttention (for efficient attention) ────────────────────────────
# from https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3
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

# ── 12 · Smoke test (non-fatal) ─────────────────────────────────────────────
python - <<PYT || true
import sys, torch
try:
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
        print("cuda device 0:", torch.cuda.get_device_name(0))
    import flash_attn
    print("flash_attn import: OK")
    import adam_atan2
    print("adam_atan2 import: OK")
    print("\n[success] Environment setup complete!")
except Exception as e:
    print("[warn] import check failed:", e, file=sys.stderr)
PYT

# ── 13 · Hand off ───────────────────────────────────────────────────────────
exec /start.sh
