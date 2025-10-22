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
set -euo pipefail
BRANCH="${BRANCH:-main}"  # set desired branch; defaults to main

cd /workspace

if [ -d TinyRecursiveModels/.git ]; then
  # Ensure remote points to the private repo
  CURRENT_URL="$(git -C TinyRecursiveModels remote get-url origin)"
  if ! echo "$CURRENT_URL" | grep -q "github.com/TrelisResearch/TinyRecursiveModels-private.git"; then
    git -C TinyRecursiveModels remote set-url origin https://github.com/TrelisResearch/TinyRecursiveModels-private.git
  fi

  git -C TinyRecursiveModels fetch --tags --prune
  git -C TinyRecursiveModels checkout "$BRANCH" || true
  git -C TinyRecursiveModels pull --ff-only
else
  if [ -n "${GITHUB_PAT:-}" ]; then
    # Safer than embedding the token in the URL (avoids showing up in process lists or storing in origin)
    git -c credential.helper= -c "http.extraHeader=Authorization: Bearer ${GITHUB_PAT}" \
      clone --filter=blob:none --recurse-submodules \
      https://github.com/TrelisResearch/TinyRecursiveModels-private.git TinyRecursiveModels
  else
    git clone --filter=blob:none --recurse-submodules \
      https://github.com/TrelisResearch/TinyRecursiveModels-private.git TinyRecursiveModels
  fi
  git -C TinyRecursiveModels checkout "$BRANCH" || true
fi

# If you use Git LFS in the repo, pull large files:
if command -v git-lfs >/dev/null 2>&1; then
  git -C TinyRecursiveModels lfs pull || true
fi

cd TinyRecursiveModels
# Only run init.sh if it exists and is executable
if [ -x ./init.sh ]; then
  bash ./init.sh
fi
'