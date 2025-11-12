import os
import math
import json
from typing import Tuple, List, Dict, Optional
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata

from argdantic import ArgParser
from pydantic import BaseModel

def _sample_batch(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    start_index: int,
    global_batch_size: int,
    max_examples_per_puzzle: Optional[int] = None,
):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        allowed = puzzle_size if max_examples_per_puzzle is None else min(max_examples_per_puzzle, puzzle_size)
        if allowed <= 0:
            continue

        append_size = min(allowed, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        if append_size == puzzle_size:
            choices = rng.permutation(puzzle_size)
        else:
            choices = rng.choice(puzzle_size, append_size, replace=False)
        batch.append(puzzle_start + choices)

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.
    rank: int
    num_replicas: int
    max_eval_augmentations: Optional[int] = None
    grid_noise_prob: float = 0.0
    grid_noise_fraction: float = 0.0
    max_examples_per_puzzle: Optional[int] = None

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Merge multiple metadata
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
            mean_puzzle_examples += current_metadata.mean_puzzle_examples*current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets
        )

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0
        if self.config.max_eval_augmentations is not None and self.config.max_eval_augmentations < 0:
            raise ValueError("max_eval_augmentations must be non-negative when provided.")

    def _load_metadata(self, dataset_path) -> PuzzleDatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets: # Load subset
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                self._data[set_name_] = {
                    field_name: np.load(os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                    for field_name, mmap_mode in field_mmap_modes.items()
                }

                if self.config.test_set_mode and self.config.max_eval_augmentations is not None:
                    self._data[set_name_] = self._limit_eval_augmentations(self._data[set_name_], self.config.max_eval_augmentations)

                group_indices = self._data[set_name_]["group_indices"]
                num_groups = group_indices.size - 1
                puzzle_group_ids = np.empty(int(group_indices[-1]), dtype=np.int32)
                for group_id in range(num_groups):
                    start = int(group_indices[group_id])
                    end = int(group_indices[group_id + 1])
                    puzzle_group_ids[start:end] = group_id
                self._data[set_name_]["puzzle_group_ids"] = puzzle_group_ids


    def _limit_eval_augmentations(self, dataset: Dict[str, np.ndarray], max_aug: int):
        # Keep the original puzzle plus up to max_aug augmented variants per group.
        if max_aug is None:
            return dataset
        max_group_size = max_aug + 1  # include base puzzle
        group_indices = dataset["group_indices"]
        puzzle_indices = dataset["puzzle_indices"]
        inputs = dataset["inputs"]
        labels = dataset["labels"]
        puzzle_ids = dataset["puzzle_identifiers"]

        new_inputs = []
        new_labels = []
        new_puzzle_ids = []
        new_puzzle_indices = [0]
        new_group_indices = [0]

        for group_idx in range(group_indices.size - 1):
            start = int(group_indices[group_idx])
            end = int(group_indices[group_idx + 1])
            group_size = end - start
            keep = min(group_size, max_group_size)
            if keep <= 0:
                continue

            for puzzle_idx in range(start, start + keep):
                ex_start = int(puzzle_indices[puzzle_idx])
                ex_end = int(puzzle_indices[puzzle_idx + 1])
                new_inputs.append(inputs[ex_start:ex_end])
                new_labels.append(labels[ex_start:ex_end])
                new_puzzle_ids.append(puzzle_ids[puzzle_idx])
                new_puzzle_indices.append(new_puzzle_indices[-1] + (ex_end - ex_start))

            new_group_indices.append(new_group_indices[-1] + keep)

        if len(new_inputs) == 0:
            return dataset

        # Concatenate along example dimension
        limited_dataset = {
            "inputs": np.concatenate(new_inputs, axis=0),
            "labels": np.concatenate(new_labels, axis=0),
            "puzzle_identifiers": np.asarray(new_puzzle_ids, dtype=puzzle_ids.dtype),
            "puzzle_indices": np.asarray(new_puzzle_indices, dtype=puzzle_indices.dtype),
            "group_indices": np.asarray(new_group_indices, dtype=group_indices.dtype),
        }
        return limited_dataset

    def _collate_batch(self, batch):
        # Convert dtype
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
                "task_identifiers": -1
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        # To tensor
        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _apply_grid_noise(self, batch: Dict[str, torch.Tensor]) -> None:
        if self.split != "train":
            return
        if self.config.grid_noise_prob <= 0 or self.config.grid_noise_fraction <= 0:
            return

        inputs = batch.get("inputs")
        if inputs is None or inputs.ndim != 2:
            return

        device = inputs.device
        apply_mask = torch.rand(inputs.size(0), device=device) < self.config.grid_noise_prob
        if not torch.any(apply_mask):
            return

        seq_len = inputs.size(1)
        if math.isqrt(seq_len) ** 2 != seq_len:
            return

        color_candidates = torch.arange(2, 12, device=device, dtype=inputs.dtype)

        for idx in torch.nonzero(apply_mask, as_tuple=False).flatten():
            grid = inputs[idx]
            valid_mask = grid >= 2
            if not torch.any(valid_mask):
                continue

            tokens = grid[valid_mask]
            values, counts = torch.unique(tokens, return_counts=True)
            if values.numel() == 0:
                continue

            major_token = values[counts.argmax()]
            major_positions = torch.nonzero(grid == major_token, as_tuple=False).flatten()
            if major_positions.numel() == 0:
                continue

            num_to_flip = max(1, int(self.config.grid_noise_fraction * major_positions.numel()))
            if num_to_flip >= major_positions.numel():
                num_to_flip = major_positions.numel()

            flip_indices = torch.randperm(major_positions.numel(), device=device)[:num_to_flip]
            flip_positions = major_positions[flip_indices]

            replacement_colors = color_candidates[color_candidates != major_token]
            if replacement_colors.numel() == 0:
                continue

            rand_idx = torch.randint(0, replacement_colors.numel(), (flip_positions.numel(),), device=device)
            new_values = replacement_colors[rand_idx].to(grid.dtype)
            grid[flip_positions] = new_values
    
    def _iter_test(self):
        for set_i, (set_name, dataset) in enumerate(self._data.items()):  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices],
                    "task_identifiers": dataset["puzzle_group_ids"][puzzle_indices]
                })
                self._apply_grid_noise(batch)

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        # Increase epoch count
        self._iters += 1
        epoch_seed = self.config.seed + self._iters

        dataset_order_rng = np.random.Generator(np.random.Philox(seed=epoch_seed))
        dataset_states = []
        for idx, (set_name, dataset) in enumerate(self._data.items()):  # type: ignore
            local_seed = epoch_seed + 1000 * (idx + 1)
            rng = np.random.Generator(np.random.Philox(seed=local_seed))
            group_order = np.concatenate(
                [rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)]
            )
            if group_order.size == 0:
                continue
            dataset_states.append({
                "name": set_name,
                "dataset": dataset,
                "rng": rng,
                "group_order": group_order,
                "start_index": 0,
            })

        active = dataset_states[:]
        while active:
            remaining = np.array([state["group_order"].size - state["start_index"] for state in active], dtype=np.int64)
            positive = remaining > 0
            if not np.any(positive):
                break
            weights = remaining[positive].astype(np.float64)
            weights = weights / weights.sum()
            choice_idx = dataset_order_rng.choice(np.arange(len(active))[positive], p=weights)
            state = active[choice_idx]

            dataset = state["dataset"]
            start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                state["rng"],
                group_order=state["group_order"],
                puzzle_indices=dataset["puzzle_indices"],
                group_indices=dataset["group_indices"],
                start_index=state["start_index"],
                global_batch_size=self.config.global_batch_size,
                max_examples_per_puzzle=self.config.max_examples_per_puzzle,
            )
            state["start_index"] = start_index

            global_effective_batch_size = batch_puzzle_indices.size

            if global_effective_batch_size < self.config.global_batch_size:
                active.pop(choice_idx)
                continue

            batch_indices = batch_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
            batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
            batch = self._collate_batch({
                "inputs": dataset["inputs"][batch_indices],
                "labels": dataset["labels"][batch_indices],
                "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices],
                "task_identifiers": dataset["puzzle_group_ids"][batch_puzzle_indices]
            })
            self._apply_grid_noise(batch)

            yield state["name"], batch, global_effective_batch_size

            if state["start_index"] >= state["group_order"].size:
                active.pop(choice_idx)
                
    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
