#!/usr/bin/env python3
"""
push_to_hf.py — Create (if needed) and upload a model folder to Hugging Face,
optionally under an organization, with private/public visibility.

Examples:
  python utils/push_to_hf.py \
    --repo-id Trelis/test-Arc2concept-aug-1000-ACT-torch_pretrain_att_arc2concept_4 \
    --model-dir ./checkpoints/Arc2concept-aug-1000-ACT-torch/pretrain_att_arc2concept_4 \
    --commit-message "Test upload" \
    --private

  python utils/push_to_hf.py \
    --repo-id my-org/bert-mini-est \
    --model-dir /path/to/checkpoint \
    --public \
    --commit-message "Upload v1.0" \
    --branch main
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, HfFolder, upload_folder
from huggingface_hub.utils import HfHubHTTPError


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    """
    Resolve a Hugging Face token in this order:
      1) --token argument
      2) HF_TOKEN env var
      3) HUGGINGFACE_HUB_TOKEN env var
      4) cached login (~/.huggingface/token)
    Returns None if nothing found (some endpoints allow anonymous, but model creation will not).
    """
    if cli_token:
        return cli_token.strip()
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token.strip()
    cached = HfFolder.get_token()
    return cached.strip() if cached else None


def bool_from_flags(private_flag: bool, public_flag: bool) -> bool:
    """
    Determine repo visibility from mutually exclusive flags.
    Defaults to private=True if neither flag is provided (safer default).
    """
    if private_flag and public_flag:
        print("ERROR: --private and --public are mutually exclusive.", file=sys.stderr)
        sys.exit(2)
    if public_flag:
        return False
    return True  # default private


def ensure_repo(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    private: bool,
    token: Optional[str],
    exist_ok: bool = True,
) -> None:
    """
    Create the repo if it doesn't exist. If it exists, optionally adjust visibility.
    """
    # Create or ensure existence
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=exist_ok,
        token=token,
    )

    # Double-check and fix visibility if needed (create_repo won't toggle on existing)
    try:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
        is_private_now = info.private
        if bool(is_private_now) != bool(private):
            # Update visibility to match requested flag
            api.update_repo_visibility(
                repo_id=repo_id,
                private=private,
                repo_type=repo_type,
                token=token,
            )
            print(f"Adjusted visibility for {repo_id} to {'private' if private else 'public'}.")
    except HfHubHTTPError as e:
        print(f"WARNING: Could not verify/adjust visibility: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Push a local model folder to Hugging Face Hub.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target repo ID in the form ORG_OR_USER/REPO_NAME (e.g., my-org/my-model).",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to local folder containing model files to upload.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Set repo to private (default if neither flag is passed).",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Set repo to public (mutually exclusive with --private).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN / HUGGINGFACE_HUB_TOKEN / cached login if omitted.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Type of repo to create/use. Default: model.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Target Git branch / revision to commit to (e.g., 'main'). Optional.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload via push_to_hf.py",
        help="Commit message for the upload. Default: 'Upload via push_to_hf.py'.",
    )
    parser.add_argument(
        "--path-in-repo",
        default="",
        help="Optional subfolder path in the repo to place the uploaded files (default: root).",
    )
    parser.add_argument(
        "--allow-external-storage",
        action="store_true",
        help="Allow files to be stored on HF's external object storage (useful for very large files).",
    )
    args = parser.parse_args()

    # Resolve token
    token = resolve_token(args.token)

    # Basic checks
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        print(f"ERROR: --model-dir not found or not a directory: {model_dir}", file=sys.stderr)
        sys.exit(1)

    private = bool_from_flags(args.private, args.public)

    # Informative header
    print("=== Hugging Face Upload ===")
    print(f"Repo ID         : {args.repo_id}")
    print(f"Repo Type       : {args.repo_type}")
    print(f"Model Dir       : {model_dir}")
    print(f"Visibility      : {'private' if private else 'public'}")
    print(f"Branch          : {args.branch or '(default)'}")
    print(f"Path in Repo    : {args.path_in_repo or '(root)'}")
    print(f"Token Source    : {'--token/env/cached' if token else '(none)'}")
    print("===========================")

    # Creating a repo requires auth; uploading to private ALWAYS requires auth.
    if token is None:
        print(
            "ERROR: No Hugging Face token found. "
            "Pass --token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN, or run `huggingface-cli login`.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create or ensure the repo, then upload
    api = HfApi()

    # Ensure you have permissions (e.g., to push to an org, you must be a member with write access).
    try:
        ensure_repo(api, args.repo_id, args.repo_type, private, token, exist_ok=True)
        print(f"Repo ready: https://huggingface.co/{args.repo_id}")
    except HfHubHTTPError as e:
        print(f"ERROR: Could not create or access repo '{args.repo_id}': {e}", file=sys.stderr)
        sys.exit(1)

    # Upload all files from model_dir
    try:
        commit_info = upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=str(model_dir),
            path_in_repo=args.path_in_repo or None,
            commit_message=args.commit_message,
            token=token,
            revision=args.branch,  # can be None
            # allow_external_storage=args.allow_external_storage,
        )
        # commit_info is a CommitInfo object with .commit_url, .oid, etc.
        print("Upload complete.")
        if getattr(commit_info, "commit_url", None):
            print(f"Commit URL: {commit_info.commit_url}")
    except HfHubHTTPError as e:
        print(f"ERROR: Upload failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()