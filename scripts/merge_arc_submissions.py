#!/usr/bin/env python3
"""
Merge multiple ARC submission files and compute pass@k metrics.

Each submission is expected to follow the format produced by `evaluators/arc.py`,
with puzzle identifiers as keys and a list of per-input attempt dictionaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Grid = Tuple[Tuple[int, ...], ...]


def load_json(path: Path):
    with path.open("r") as handle:
        return json.load(handle)


def to_grid(value: Sequence[Sequence[int]]) -> Grid:
    return tuple(tuple(int(cell) for cell in row) for row in value)


def normalize_submission_attempts(raw_attempts: List[Dict[str, Sequence[Sequence[int]]]]) -> List[List[Grid]]:
    normalized: List[List[Grid]] = []
    for attempt_bundle in raw_attempts:
        # Ensure we iterate attempts in order (attempt_1, attempt_2, ...)
        attempt_keys = sorted(attempt_bundle.keys())
        grids = [to_grid(attempt_bundle[key]) for key in attempt_keys]
        normalized.append(grids)
    return normalized


def merge_submissions(paths: Iterable[Path]) -> Dict[str, List[List[Grid]]]:
    merged: Dict[str, List[List[Grid]]] = {}
    for path in paths:
        data = load_json(path)
        for puzzle_id, attempts in data.items():
            if puzzle_id in merged:
                raise ValueError(
                    f"Duplicate puzzle '{puzzle_id}' encountered while merging {path}"
                )
            merged[puzzle_id] = normalize_submission_attempts(attempts)
    return merged


def compute_pass_k(
    merged: Dict[str, List[List[Grid]]],
    solutions_path: Path,
    pass_ks: Sequence[int],
) -> Dict[int, float]:
    solutions_raw = load_json(solutions_path)
    pass_totals = {k: 0.0 for k in pass_ks}
    puzzles_evaluated = 0

    for puzzle_id, attempts_per_input in merged.items():
        if puzzle_id not in solutions_raw:
            raise KeyError(
                f"Puzzle '{puzzle_id}' missing from solutions file {solutions_path}"
            )
        gt_outputs = [to_grid(grid) for grid in solutions_raw[puzzle_id]]

        if len(attempts_per_input) != len(gt_outputs):
            raise ValueError(
                f"Puzzle '{puzzle_id}' test count mismatch: "
                f"{len(attempts_per_input)} attempts vs {len(gt_outputs)} solutions"
            )

        puzzles_evaluated += 1
        for pass_k in pass_ks:
            solved_cases = 0
            for attempt_bundle, gt in zip(attempts_per_input, gt_outputs):
                # Allow multiple attempts per test example; cap at pass_k
                for candidate in attempt_bundle[:pass_k]:
                    if candidate == gt:
                        solved_cases += 1
                        break
            pass_totals[pass_k] += solved_cases / len(gt_outputs)

    if puzzles_evaluated == 0:
        raise RuntimeError("No puzzles evaluated. Check submission inputs.")

    return {k: pass_totals[k] / puzzles_evaluated for k in pass_ks}


def serialize_submission(merged: Dict[str, List[List[Grid]]]) -> Dict[str, List[Dict[str, List[List[int]]]]]:
    submission_serialized: Dict[str, List[Dict[str, List[List[int]]]]] = {}
    for puzzle_id, attempts_per_example in merged.items():
        serialized_examples = []
        for attempt_bundle in attempts_per_example:
            serialized_examples.append(
                {
                    f"attempt_{idx + 1}": [list(row) for row in grid]
                    for idx, grid in enumerate(attempt_bundle)
                }
            )
        submission_serialized[puzzle_id] = serialized_examples
    return submission_serialized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submissions",
        nargs="+",
        required=True,
        help="Paths to submission.json files to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the merged submission.json file.",
    )
    parser.add_argument(
        "--solutions",
        type=Path,
        default=Path("kaggle/combined/arc-agi_evaluation2clean_solutions.json"),
        help="Path to the reference solutions JSON.",
    )
    parser.add_argument(
        "--pass-ks",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Pass@k values to compute.",
    )

    args = parser.parse_args()
    submission_paths = [Path(p) for p in args.submissions]

    merged = merge_submissions(submission_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as handle:
        json.dump(serialize_submission(merged), handle)

    pass_metrics = compute_pass_k(merged, args.solutions, args.pass_ks)
    for k, value in pass_metrics.items():
        print(f"pass@{k}: {value:.4f}")


if __name__ == "__main__":
    main()
