#!/usr/bin/env python3
"""
Automate chunked post-training runs on ARC evaluation datasets.

Steps:
 1. Split `evaluation2clean` challenges/solutions into fixed-size chunks.
 2. Build augmented datasets for each chunk.
 3. Launch post-training sequentially for every chunk.
 4. Collect per-chunk submissions and merge them into a single file with pass@k metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parent.parent

# Allow running either as a module or as a standalone script.
if __name__ == "__main__" and __package__ is None:  # pragma: no cover - runtime import fix
    sys.path.append(str(ROOT))
    from scripts.merge_arc_submissions import (  # type: ignore
        compute_pass_k,
        merge_submissions,
        serialize_submission,
    )
else:  # pragma: no cover - runtime import fix
    from .merge_arc_submissions import compute_pass_k, merge_submissions, serialize_submission


@dataclass
class ChunkConfig:
    index: int
    suffix: str
    subset_name: str
    puzzles: Dict[str, Any]
    solutions: Dict[str, Any]
    challenges_path: Path
    solutions_path: Path
    dataset_dir: Path
    run_name: str


def chunk_suffix(index: int) -> str:
    if index < 26:
        return chr(ord("A") + index)
    return f"{index + 1:02d}"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle)


def prepare_chunks(args) -> List[ChunkConfig]:
    base_challenges_path = Path(f"{args.input_prefix}_{args.subset}_challenges.json")
    base_solutions_path = Path(f"{args.input_prefix}_{args.subset}_solutions.json")

    if not base_challenges_path.exists():
        raise FileNotFoundError(f"Challenges file not found: {base_challenges_path}")
    if not base_solutions_path.exists():
        raise FileNotFoundError(f"Solutions file not found: {base_solutions_path}")

    challenges = load_json(base_challenges_path)
    solutions = load_json(base_solutions_path)

    puzzle_ids = sorted(challenges.keys())
    if not puzzle_ids:
        raise ValueError(f"No puzzles found in {base_challenges_path}")

    chunks: List[ChunkConfig] = []
    for idx, start in enumerate(range(0, len(puzzle_ids), args.chunk_size)):
        end = start + args.chunk_size
        chunk_ids = puzzle_ids[start:end]
        suffix = chunk_suffix(idx)
        subset_name = f"{args.subset}{suffix}"

        chunk_challenges = {p: challenges[p] for p in chunk_ids}
        chunk_solutions = {p: solutions[p] for p in chunk_ids}

        chunk = ChunkConfig(
            index=idx,
            suffix=suffix,
            subset_name=subset_name,
            puzzles=chunk_challenges,
            solutions=chunk_solutions,
            challenges_path=Path(f"{args.input_prefix}_{subset_name}_challenges.json"),
            solutions_path=Path(f"{args.input_prefix}_{subset_name}_solutions.json"),
            dataset_dir=Path(f"{args.output_dir_prefix}{suffix}-aug-{args.num_aug}"),
            run_name=f"{args.run_prefix}{suffix}",
        )
        chunks.append(chunk)

    print(f"Prepared {len(chunks)} chunk(s) from {len(puzzle_ids)} puzzles.")
    return chunks


def write_chunk_files(chunk: ChunkConfig, overwrite: bool) -> None:
    for path, payload in (
        (chunk.challenges_path, chunk.puzzles),
        (chunk.solutions_path, chunk.solutions),
    ):
        if path.exists() and not overwrite:
            print(f"[Chunk {chunk.subset_name}] Skipping existing file: {path}")
            continue
        print(f"[Chunk {chunk.subset_name}] Writing {path}")
        dump_json(path, payload)


def run_command(cmd: Sequence[str], cwd: Path, env: Optional[Dict[str, str]] = None, log_path: Optional[Path] = None, dry_run: bool = False) -> None:
    display_cmd = " ".join(cmd)
    print(f"Running: {display_cmd}")
    if dry_run:
        return

    if log_path is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def ensure_checkpoint(args) -> None:
    if args.skip_download:
        return

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_path = checkpoint_dir / args.checkpoint_step
    if target_path.exists():
        print(f"Checkpoint present at {target_path}, skipping download.")
        return

    print("Installing hf_transfer (if needed)...")
    run_command(
        ["uv", "pip", "install", "hf_transfer"],
        cwd=ROOT,
        dry_run=args.dry_run,
    )

    print(f"Downloading checkpoint {args.model_repo}:{args.checkpoint_step} -> {checkpoint_dir}")
    run_command(
        [
            "hf",
            "download",
            args.model_repo,
            args.checkpoint_step,
            "--local-dir",
            str(checkpoint_dir),
        ],
        cwd=ROOT,
        dry_run=args.dry_run,
    )


def build_dataset(chunk: ChunkConfig, args) -> None:
    if args.skip_datasets:
        print(f"[Chunk {chunk.subset_name}] Skipping dataset build (requested).")
        return
    if chunk.dataset_dir.exists() and not args.rebuild_datasets:
        print(f"[Chunk {chunk.subset_name}] Dataset already exists at {chunk.dataset_dir}, skipping.")
        return

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "dataset.build_arc_dataset",
        "--input-file-prefix",
        args.input_prefix,
        "--output-dir",
        str(chunk.dataset_dir),
        "--subsets",
        chunk.subset_name,
        "--test-set-name",
        chunk.subset_name,
        "--num-aug",
        str(args.num_aug),
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    run_command(cmd, cwd=ROOT, dry_run=args.dry_run)


def run_training(chunk: ChunkConfig, args) -> None:
    if args.skip_train:
        print(f"[Chunk {chunk.subset_name}] Training skipped (requested).")
        return

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if not args.enable_wandb:
        env.setdefault("WANDB_DISABLED", "true")

    data_path = str(chunk.dataset_dir)
    checkpoint_path = Path(args.checkpoint_dir) / args.checkpoint_step
    if not checkpoint_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}. Run without --skip-download to fetch it.")

    cmd = [
        "torchrun",
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--rdzv_backend",
        args.rdzv_backend,
        "--rdzv_endpoint",
        args.rdzv_endpoint,
        "pretrain.py",
        "--config-name",
        args.config_name,
        f"data_paths=['{data_path}']",
        f"data_paths_test=['{data_path}']",
        f"load_checkpoint={checkpoint_path}",
        f"+run_name={chunk.run_name}",
    ]
    if args.wandb_project:
        cmd.append(f"+project_name='{args.wandb_project}'")
    if args.wandb_entity:
        cmd.append(f"+entity='{args.wandb_entity}'")
    for override in args.extra_override:
        cmd.append(override)

    log_dir = Path(args.log_dir)
    log_path = log_dir / f"{chunk.run_name}.log"

    print(f"[Chunk {chunk.subset_name}] Launching training -> log: {log_path}")
    run_command(cmd, cwd=ROOT, env=env, log_path=log_path, dry_run=args.dry_run)


def locate_submission(chunk: ChunkConfig) -> Path:
    checkpoint_root = ROOT / "checkpoints"
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"No checkpoints directory found at {checkpoint_root}")

    candidates = sorted(checkpoint_root.glob(f"*/{chunk.run_name}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Run directory for {chunk.run_name} not found under {checkpoint_root}")

    run_dir = candidates[0]
    evaluator_dirs = [p for p in run_dir.glob("evaluator_*") if (p / "submission.json").exists()]
    if not evaluator_dirs:
        raise FileNotFoundError(f"No evaluator outputs found in {run_dir}")

    def step_key(path: Path) -> int:
        name = path.name
        if "_step_" in name:
            try:
                return int(name.split("_step_")[-1])
            except ValueError:
                return -1
        return -1

    best_dir = max(evaluator_dirs, key=step_key)
    submission_path = best_dir / "submission.json"
    if not submission_path.exists():
        raise FileNotFoundError(f"submission.json not found in {best_dir}")

    print(f"[Chunk {chunk.subset_name}] Using submission from {best_dir.name}")
    return submission_path


def copy_submission(source: Path, chunk: ChunkConfig, args) -> Path:
    if args.no_copy_submissions:
        return source

    submissions_dir = Path(args.submission_dir)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    dest = submissions_dir / f"{chunk.run_name}_submission.json"
    if args.dry_run:
        print(f"[Chunk {chunk.subset_name}] (dry-run) Would copy {source} -> {dest}")
        return dest

    if dest.exists() and not args.overwrite_submissions:
        print(f"[Chunk {chunk.subset_name}] Submission already copied to {dest}")
        return dest

    print(f"[Chunk {chunk.subset_name}] Copying submission -> {dest}")
    shutil.copy2(source, dest)
    return dest


def merge_and_score(submission_paths: List[Path], args) -> None:
    if not submission_paths:
        print("No submission paths provided for merging.")
        return

    if args.dry_run:
        print("(dry-run) Skipping merge and metric computation.")
        return

    merged = merge_submissions(submission_paths)
    merged_json = serialize_submission(merged)

    output_path = Path(args.merge_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(merged_json, handle)

    solutions_path = Path(args.solutions_path)
    metrics = compute_pass_k(merged, solutions_path, args.pass_ks)
    print("\nMerged submission metrics:")
    for k in sorted(metrics.keys()):
        print(f"  pass@{k}: {metrics[k]:.4f}")
    if not args.dry_run:
        print(f"Merged submission saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-prefix", default="kaggle/combined/arc-agi")
    parser.add_argument("--subset", default="evaluation2clean")
    parser.add_argument("--chunk-size", type=int, default=38)
    parser.add_argument("--num-aug", type=int, default=1000)
    parser.add_argument("--output-dir-prefix", default="data/arc-eval2clean")
    parser.add_argument("--run-prefix", default="posttrain_eval2clean")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--checkpoint-dir", default="pretrained")
    parser.add_argument("--checkpoint-step", default="step_155718")
    parser.add_argument("--model-repo", default="Sanjin2024/TinyRecursiveModels-ARC-AGI-1")
    parser.add_argument("--skip-download", action="store_true")

    parser.add_argument("--config-name", default="cfg_posttrain")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--rdzv-backend", default="c10d")
    parser.add_argument("--rdzv-endpoint", default="localhost:0")
    parser.add_argument("--extra-override", action="append", default=[])

    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--submission-dir", default="submissions")
    parser.add_argument("--merge-output", default="submissions/eval2clean_merged_submission.json")
    parser.add_argument("--solutions-path", default="kaggle/combined/arc-agi_evaluation2clean_solutions.json")
    parser.add_argument("--pass-ks", type=int, nargs="+", default=[1, 2])

    parser.add_argument("--overwrite-splits", action="store_true")
    parser.add_argument("--rebuild-datasets", action="store_true")
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--no-copy-submissions", action="store_true")
    parser.add_argument("--overwrite-submissions", action="store_true")
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    ensure_checkpoint(args)
    chunks = prepare_chunks(args)

    submission_paths: List[Path] = []
    for chunk in chunks:
        print(f"\n=== Processing chunk {chunk.index + 1}/{len(chunks)}: {chunk.subset_name} ({len(chunk.puzzles)} puzzles) ===")
        write_chunk_files(chunk, overwrite=args.overwrite_splits)
        build_dataset(chunk, args)
        run_training(chunk, args)

        if args.dry_run:
            continue

        try:
            submission_source = locate_submission(chunk)
        except FileNotFoundError as exc:
            if args.skip_train:
                raise
            raise RuntimeError(f"Failed to locate submission for {chunk.run_name}.") from exc

        submission_path = copy_submission(submission_source, chunk, args)
        submission_paths.append(submission_path)

    if not args.skip_merge:
        print("\n=== Merging submissions ===")
        merge_and_score(submission_paths, args)
    else:
        print("Skipping merge phase (requested).")


if __name__ == "__main__":
    main()
