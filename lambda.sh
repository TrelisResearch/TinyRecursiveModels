#!/usr/bin/env bash
set -euo pipefail

# Load secrets (if file exists)
if [ -f "$HOME/.lambda.env" ]; then
  source "$HOME/.lambda.env"
fi

# ── 0 · Env toggles ──────────────────────────────────────────────────────────
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PIP_ROOT_USER_ACTION=ignore
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
export GITHUB_PAT="${GITHUB_PAT:-}"

# ── 1 · System packages ─────────────────────────────────────────────────────
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git nano ninja-build build-essential python3-dev python3-venv \
  libnuma1 libnuma-dev pkg-config curl ca-certificates

# ── 2 · Git identity (optional) ─────────────────────────────────────────────
[ -n "${GIT_USER_NAME:-}"  ] && git config --global user.name  "$GIT_USER_NAME"
[ -n "${GIT_USER_EMAIL:-}" ] && git config --global user.email "$GIT_USER_EMAIL"

# ── 3 · Repo ────────────────────────────────────────────────────────────────
if [ -d TinyRecursiveModels-private/.git ]; then
    git -C TinyRecursiveModels-private pull --ff-only
else
    git clone "https://${GITHUB_PAT}@github.com/TrelisResearch/TinyRecursiveModels-private.git" TinyRecursiveModels-private
fi

cd TinyRecursiveModels-private

git checkout base-in

#4. Run installs
bash lambda-init.sh