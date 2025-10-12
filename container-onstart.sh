bash -lc '
set -euo pipefail

# ── 0 · Env toggles ──────────────────────────────────────────────────────────
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PIP_ROOT_USER_ACTION=ignore
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

# ── 1 · System packages ─────────────────────────────────────────────────────
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git nano ninja-build build-essential python3-dev python3-venv \
  libnuma1 libnuma-dev pkg-config curl ca-certificates

# ── 2 · Git identity (optional) ─────────────────────────────────────────────
[ -n "${GIT_USER_NAME:-}"  ] && git config --global user.name  "$GIT_USER_NAME"
[ -n "${GIT_USER_EMAIL:-}" ] && git config --global user.email "$GIT_USER_EMAIL"

# ── 3 · Repo ────────────────────────────────────────────────────────────────
cd /workspace
if [ -d TinyRecursiveModels/.git ]; then
  git -C TinyRecursiveModels pull --ff-only
else
  if [ -n "${GITHUB_PAT:-}" ]; then
    git clone "https://${GITHUB_PAT}@github.com/TrelisResearch/TinyRecursiveModels.git"
  else
    git clone https://github.com/TrelisResearch/TinyRecursiveModels.git
  fi
fi
cd TinyRecursiveModels

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
uv pip install --pre --upgrade --no-cache-dir torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"

# ── 6 · Build helpers ───────────────────────────────────────────────────────
uv pip install --upgrade --no-cache-dir packaging ninja wheel setuptools setuptools-scm numba huggingface_hub

# ── 7 · FlashAttention build dependencies ───────────────────────────────────
uv pip install --no-cache-dir psutil

# ── 8 · FlashAttention (for efficient attention) ────────────────────────────
uv pip install --no-cache-dir --no-build-isolation flash-attn || {
  echo "[warn] flash-attn wheel not found; falling back to source build"
  uv pip install --no-cache-dir --no-build-isolation flash-attn
}

# ── 9 · Project deps ────────────────────────────────────────────────────────
[ -f requirements.txt ] && uv pip install --no-cache-dir -r requirements.txt

# ── 10 · adam-atan2 optimizer (required by TRM) ─────────────────────────────
uv pip install --no-cache-dir --no-build-isolation adam-atan2

# ── 11 · wandb login (if API key provided) ──────────────────────────────────
if [ -n "${WANDB_API_KEY}" ]; then
  wandb login "$WANDB_API_KEY"
  echo "[info] Logged into wandb"
else
  echo "[warn] WANDB_API_KEY not set; skip wandb login (you can run 'wandb login' manually later)"
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
'
