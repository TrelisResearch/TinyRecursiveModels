from typing import List, Optional

import pydantic
import numpy as np


# Global list mapping each dihedral transform id to its inverse.
# Index corresponds to the original tid, and the value is its inverse.
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]

# ARC constants shared across builders/evaluators
ARC_GRID_SIZE = 30
ARC_GRID_TOKENS = ARC_GRID_SIZE * ARC_GRID_SIZE
ARC_DIHEDRAL_TOKEN_BASE = 12
ARC_COLOR_TOKEN_BASE = ARC_DIHEDRAL_TOKEN_BASE + 8  # supports colors 0-9
ARC_METADATA_TOKENS = 1 + 10  # dihedral + full color mapping (including 0)
ARC_METADATA_UNIQUE_TOKENS = 8 + 10  # 8 dihedral IDs, 10 color values


class PuzzleDatasetMetadata(pydantic.BaseModel):
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: List[str]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""
    
    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)       # horizontal flip
    elif tid == 5:
        return np.flipud(arr)       # vertical flip
    elif tid == 6:
        return arr.T                # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        return arr
    
    
def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])
