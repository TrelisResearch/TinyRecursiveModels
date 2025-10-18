import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch


def load_checkpoint_embeddings(checkpoint_path: Path) -> torch.Tensor:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    key = "_orig_mod.model.inner.puzzle_emb.weights"
    if key not in state_dict:
        raise KeyError(f"Puzzle embedding weights not found in checkpoint (missing {key}).")
    embeddings = state_dict[key].float()
    return embeddings


def load_identifier_names(identifier_path: Path) -> List[str]:
    with identifier_path.open() as f:
        identifiers = json.load(f)
    if not isinstance(identifiers, list):
        raise ValueError("identifiers.json must contain a list of puzzle names")
    return identifiers


def filter_base_tasks(identifiers: List[str]) -> Tuple[List[int], List[str]]:
    indices: List[int] = []
    names: List[str] = []
    for idx, name in enumerate(identifiers):
        if idx == 0:
            continue  # skip <blank>
        if "|||" in name:
            continue  # skip augmented variants
        indices.append(idx)
        names.append(name)
    return indices, names


def cosine_similarity_pairs(embeddings: torch.Tensor, indices: List[int], threshold: float) -> Tuple[int, List[Tuple[int, int, float]]]:
    if not indices:
        return 0, []
    subset = embeddings[indices]
    subset = torch.nn.functional.normalize(subset, dim=1)
    sim = subset @ subset.T
    mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    meeting = torch.nonzero((sim >= threshold) & mask)
    count = meeting.shape[0]
    samples: List[Tuple[int, int, float]] = []
    for entry in meeting[:10]:
        i, j = entry.tolist()
        samples.append((i, j, float(sim[i, j])))
    return count, samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute cosine similarity between puzzle embeddings")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint file")
    parser.add_argument("--identifiers", type=Path, required=True, help="Path to identifiers.json")
    parser.add_argument("--threshold", type=float, default=0.9, help="Cosine similarity threshold")
    args = parser.parse_args()

    embeddings = load_checkpoint_embeddings(args.checkpoint)
    identifiers = load_identifier_names(args.identifiers)
    base_indices, base_names = filter_base_tasks(identifiers)

    count, sample_pairs = cosine_similarity_pairs(embeddings, base_indices, args.threshold)
    print(f"Base tasks considered: {len(base_indices)}")
    print(f"Pairs with cosine similarity >= {args.threshold:.3f}: {count}")
    if sample_pairs:
        print("Sample pairs:")
        for i, j, sim in sample_pairs:
            print(f"  {base_names[i]} <> {base_names[j]}: cos={sim:.4f}")


if __name__ == "__main__":
    main()
