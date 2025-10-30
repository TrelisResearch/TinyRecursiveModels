from typing import Optional, Any, Sequence, List, Tuple, Dict, Set
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import numpy as np
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None  # type: ignore

if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
    wandb = None  # type: ignore
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper

# Enable TF32 tensor cores on compatible GPUs (e.g., L4, H100) for faster matmuls.
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class MetaLearningConfig(pydantic.BaseModel):
    enabled: bool = False
    inner_lr: Optional[float] = None
    inner_lr_embed: Optional[float] = None
    inner_steps: int = 1


class MetaEvalConfig(pydantic.BaseModel):
    data_paths: List[str]
    global_batch_size: Optional[int] = None
    max_eval_augmentations: Optional[int] = None


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []
    halt_max_steps_eval: Optional[int] = None

    # Hyperparams
    global_batch_size: int
    eval_global_batch_size: Optional[int] = None
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float
    puzzle_emb_reinit_strategy: str = "mean"

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    entity: Optional[str] = None  # wandb entity
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    checkpoint_every_n_steps: Optional[int] = None
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []
    eval_max_augmentations: Optional[int] = None

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings
    freeze_weights_epochs: Optional[int] = None # If set and freeze_weights is True, number of initial epochs to keep trunk frozen

    meta_learning: MetaLearningConfig = MetaLearningConfig()
    meta_eval: Optional[MetaEvalConfig] = None

    # Dataloader controls
    dataloader_num_workers: int = 1
    dataloader_prefetch_factor: int = 8
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    grid_noise_prob: Optional[float] = None
    grid_noise_fraction: Optional[float] = None

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    optimizer_tags: Sequence[str]
    carry: Any

    step: int
    total_steps: int
    blank_identifier_id: int
    freeze_trunk_until_step: Optional[int]
    trunk_frozen_active: bool


class MetaQueryFetcher:
    """Lightweight index for pulling query batches by puzzle identifier."""

    def __init__(self, dataset: PuzzleDataset, max_batch_size: int):
        self._dataset = dataset
        self._dataset._lazy_load_dataset()  # type: ignore[attr-defined]
        self._metadata = dataset.metadata
        self._entries: Dict[int, List[Tuple[Dict[str, np.ndarray], int, int, int]]] = {}
        self._max_batch_size = max_batch_size

        for data in self._dataset._data.values():  # type: ignore[attr-defined]
            puzzle_ids = data["puzzle_identifiers"]
            puzzle_indices = data["puzzle_indices"]
            task_ids = data.get("puzzle_group_ids")

            for puzzle_idx in range(puzzle_ids.shape[0]):
                pid = int(puzzle_ids[puzzle_idx])
                start = int(puzzle_indices[puzzle_idx])
                end = int(puzzle_indices[puzzle_idx + 1])
                task_id = int(task_ids[puzzle_idx]) if task_ids is not None else -1
                self._entries.setdefault(pid, []).append((data, start, end, task_id))

    @property
    def metadata(self) -> PuzzleDatasetMetadata:
        return self._metadata

    def fetch(self, puzzle_ids: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        unique_ids = torch.unique(puzzle_ids.detach().cpu()).tolist()
        inputs: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        puzzle_identifier_list: List[np.ndarray] = []
        task_identifier_list: List[np.ndarray] = []
        per_puzzle_count: Dict[int, int] = {}

        max_total = max(self._max_batch_size, 1)
        remaining_total = max_total
        total_collected = 0

        # Avoid degenerate division when there are more puzzles than capacity
        active_ids = unique_ids[:max_total]
        ids_len = max(len(active_ids), 1)
        per_puzzle_limit = max(1, math.ceil(max_total / ids_len))

        for pid in active_ids:
            entries = self._entries.get(int(pid))
            if not entries:
                continue

            per_puzzle_count.setdefault(pid, 0)

            for data, start, end, task_id in entries:
                if remaining_total <= 0:
                    break

                remaining_for_pid = per_puzzle_limit - per_puzzle_count[pid]
                if remaining_for_pid <= 0:
                    break

                block_size = end - start
                take = min(block_size, remaining_for_pid, remaining_total)
                if take <= 0:
                    continue

                slice_start = start
                slice_end = start + take

                inputs.append(data["inputs"][slice_start:slice_end])
                labels.append(data["labels"][slice_start:slice_end])

                puzzle_dtype = data["puzzle_identifiers"].dtype
                task_dtype = (
                    data["puzzle_group_ids"].dtype
                    if data.get("puzzle_group_ids") is not None
                    else np.int32
                )
                puzzle_identifier_list.append(np.full(take, pid, dtype=puzzle_dtype))
                task_identifier_list.append(np.full(take, task_id, dtype=task_dtype))

                per_puzzle_count[pid] += take
                total_collected += take
                remaining_total -= take

                if remaining_total <= 0:
                    break

            if remaining_total <= 0:
                break

        if not inputs:
            return None

        inputs_arr = np.concatenate(inputs, axis=0)
        labels_arr = np.concatenate(labels, axis=0)
        puzzle_arr = np.concatenate(puzzle_identifier_list, axis=0)
        task_arr = np.concatenate(task_identifier_list, axis=0)

        batch_np = {
            "inputs": inputs_arr,
            "labels": labels_arr,
            "puzzle_identifiers": puzzle_arr,
            "task_identifiers": task_arr,
        }

        # Reuse dataset collate to match train/eval behaviour (dtype + ignore label mapping).
        collated = self._dataset._collate_batch(batch_np)  # type: ignore[attr-defined]
        return collated

def create_dataloader(
    config: PretrainConfig,
    split: str,
    rank: int,
    world_size: int,
    max_eval_augmentations: Optional[int] = None,
    data_paths_override: Optional[List[str]] = None,
    **kwargs,
):
    noise_prob = config.grid_noise_prob if config.grid_noise_prob is not None else getattr(config.arch, "grid_noise_prob", 0.0)
    noise_fraction = config.grid_noise_fraction if config.grid_noise_fraction is not None else getattr(config.arch, "grid_noise_fraction", 0.0)

    kwargs.setdefault("grid_noise_prob", noise_prob)
    kwargs.setdefault("grid_noise_fraction", noise_fraction)
    if data_paths_override is not None:
        dataset_paths = data_paths_override
    else:
        if split == "test" and config.data_paths_test and len(config.data_paths_test) > 0:
            dataset_paths = config.data_paths_test
        else:
            dataset_paths = config.data_paths
    if dataset_paths is None:
        raise ValueError(f"No dataset paths provided for split '{split}'.")

    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=dataset_paths,
            rank=rank,
            num_replicas=world_size,
            max_eval_augmentations=max_eval_augmentations,
            **kwargs,
        ),
        split=split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=getattr(config, "dataloader_num_workers", 4),
        prefetch_factor=getattr(config, "dataloader_prefetch_factor", 8),
        pin_memory=getattr(config, "dataloader_pin_memory", True),
        persistent_workers=getattr(config, "dataloader_persistent_workers", True)
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )
    if config.halt_max_steps_eval is not None:
        model_cfg.setdefault("halt_max_steps_eval", config.halt_max_steps_eval)

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if (not config.meta_learning.enabled) and ("DISABLE_COMPILE" not in os.environ):
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    optimizer_tags: List[str]

    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamAtan2(
                model.parameters(),
                lr=0.0001,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
        optimizer_tags = ["trunk"]
    elif config.freeze_weights:
        freeze_epochs = config.freeze_weights_epochs
        add_trunk_optimizer = freeze_epochs is not None and max(0, freeze_epochs) < config.epochs
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
        optimizer_tags = ["embedding"]
        if add_trunk_optimizer:
            optimizers.append(
                AdamAtan2(
                    model.parameters(),
                    lr=0.0001,  # Needs to be set by scheduler
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2)
                )
            )
            optimizer_lrs.append(config.lr)
            optimizer_tags.append("trunk")
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamAtan2(
                model.parameters(),
                lr=0.0001,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]
        optimizer_tags = ["embedding", "trunk"]

    return model, optimizers, optimizer_lrs, optimizer_tags

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs, optimizer_tags = create_model(config, train_metadata, rank=rank, world_size=world_size)

    freeze_trunk_until_step: Optional[int] = None
    if config.freeze_weights and any(tag == "trunk" for tag in optimizer_tags):
        freeze_epochs = config.freeze_weights_epochs
        if freeze_epochs is not None:
            freeze_epochs = max(0, freeze_epochs)
            if freeze_epochs >= config.epochs:
                freeze_trunk_until_step = total_steps
            elif config.epochs > 0:
                freeze_ratio = freeze_epochs / config.epochs
                freeze_trunk_until_step = max(0, math.ceil(total_steps * freeze_ratio))

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        optimizer_tags=optimizer_tags,
        carry=None,
        blank_identifier_id=train_metadata.blank_identifier_id,
        freeze_trunk_until_step=freeze_trunk_until_step,
        trunk_frozen_active=freeze_trunk_until_step is not None and freeze_trunk_until_step > 0
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def _remap_compiled_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Handle checkpoints saved from torch.compile (keys prefixed with _orig_mod.)."""
    if not any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        return state_dict

    remapped = {}
    prefix = "_orig_mod."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            remapped[key[len(prefix):]] = value
        else:
            remapped[key] = value
    return remapped


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")
        state_dict = _remap_compiled_state_dict(state_dict)

        # Resize and reset puzzle emb if needed
        puzzle_emb_keys = [
            "model.inner.puzzle_emb.weights",
            "_orig_mod.model.inner.puzzle_emb.weights",
        ]
        for puzzle_emb_name in puzzle_emb_keys:
            if puzzle_emb_name in state_dict:
                expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
                puzzle_emb = state_dict[puzzle_emb_name]
                if puzzle_emb.shape != expected_shape:
                    print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                    strategy = getattr(config, "puzzle_emb_reinit_strategy", "mean").lower()
                    puzzle_emb_float = puzzle_emb.to(torch.float32)
                    mean_vec = torch.mean(puzzle_emb_float, dim=0, keepdim=True)
                    if strategy == "mean":
                        state_dict[puzzle_emb_name] = mean_vec.expand(expected_shape).to(puzzle_emb.dtype).contiguous()
                    elif strategy == "normal":
                        std_vec = torch.std(puzzle_emb_float, dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
                        noise = torch.randn(expected_shape, device=mean_vec.device, dtype=torch.float32)
                        new_weights = noise * std_vec + mean_vec
                        state_dict[puzzle_emb_name] = new_weights.to(puzzle_emb.dtype).contiguous()
                    else:
                        raise ValueError(f"Unsupported puzzle_emb_reinit_strategy: {strategy}")
                break
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
        if len(unexpected_keys):
            print(f"Warning: unexpected checkpoint keys skipped: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        if len(missing_keys):
            print(f"Warning: missing checkpoint keys initialized from defaults: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def compute_grad_norms(model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (embedding_grad_norm, trunk_grad_norm)."""
    try:
        device = next(model.parameters()).device  # type: ignore
    except StopIteration:
        device = torch.device("cpu")

    embed_sq = torch.zeros((), device=device, dtype=torch.float32)
    trunk_sq = torch.zeros((), device=device, dtype=torch.float32)

    embed_keys = ("puzzle_emb", "aug_dihedral_emb", "aug_color_pair_emb")

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_sq = param.grad.detach().float().pow(2).sum()
        if any(key in name for key in embed_keys):
            embed_sq += grad_sq
        else:
            trunk_sq += grad_sq

    core_model = getattr(model, "model", None)
    if core_model is not None:
        inner = getattr(core_model, "inner", None)
        puzzle_emb = getattr(inner, "puzzle_emb", None)
        if puzzle_emb is not None and getattr(puzzle_emb, "local_weights", None) is not None:
            local_grad = puzzle_emb.local_weights.grad  # type: ignore
            if local_grad is not None:
                embed_sq += local_grad.detach().float().pow(2).sum()

    return embed_sq.sqrt(), trunk_sq.sqrt()


def _get_puzzle_embedding_module(model: nn.Module):
    core_model = getattr(model, "model", None)
    inner = getattr(core_model, "inner", None) if core_model is not None else None
    return getattr(inner, "puzzle_emb", None) if inner is not None else None


def _clear_sparse_embedding_grads(model: nn.Module) -> None:
    puzzle_emb = _get_puzzle_embedding_module(model)
    if puzzle_emb is None:
        return
    local_weights = getattr(puzzle_emb, "local_weights", None)
    if local_weights is not None:
        grad = getattr(local_weights, "grad", None)
        if grad is not None:
            local_weights.grad = None


class _SparseEmbeddingInnerLoopState:
    def __init__(self):
        self.backed_ids: Set[int] = set()
        self.backups: List[Tuple[torch.Tensor, torch.Tensor]] = []


def _backup_sparse_embedding_rows(
    puzzle_emb: nn.Module, ids: torch.Tensor, state: _SparseEmbeddingInnerLoopState
) -> None:
    if ids.numel() == 0:
        return

    id_list = ids.detach().tolist()
    new_ids = [idx for idx in id_list if idx not in state.backed_ids]
    if not new_ids:
        return

    state.backed_ids.update(new_ids)
    ids_tensor = ids.new_tensor(new_ids)
    with torch.no_grad():
        state.backups.append((ids_tensor, puzzle_emb.weights[ids_tensor].detach().clone()))


def _apply_sparse_embedding_inner_step(
    puzzle_emb: nn.Module,
    blank_id: int,
    inner_lr: float,
    state: _SparseEmbeddingInnerLoopState,
) -> None:
    local_weights = getattr(puzzle_emb, "local_weights", None)
    local_ids = getattr(puzzle_emb, "local_ids", None)
    if local_weights is None or local_ids is None:
        return

    grad = getattr(local_weights, "grad", None)
    if grad is None:
        return

    valid_mask = local_ids != blank_id
    if valid_mask.sum().item() == 0:
        local_weights.grad = None
        return

    valid_ids = local_ids[valid_mask].to(dtype=torch.long)
    grad_vals = grad[valid_mask]
    if valid_ids.numel() == 0:
        local_weights.grad = None
        return

    unique_ids, inverse = torch.unique(valid_ids, return_inverse=True)
    grad_accum = torch.zeros(
        (unique_ids.shape[0], grad_vals.shape[1]),
        device=grad_vals.device,
        dtype=grad_vals.dtype,
    )
    grad_accum.scatter_add_(0, inverse.unsqueeze(-1).expand_as(grad_vals), grad_vals)

    _backup_sparse_embedding_rows(puzzle_emb, unique_ids, state)

    grad_update = grad_accum.to(dtype=puzzle_emb.weights.dtype)

    with torch.no_grad():
        puzzle_emb.weights[unique_ids].add_(grad_update, alpha=-inner_lr)

    local_weights.grad = None


def _restore_sparse_embedding_from_backup(
    puzzle_emb: Optional[nn.Module], state: _SparseEmbeddingInnerLoopState
) -> None:
    if puzzle_emb is None or not state.backups:
        return

    with torch.no_grad():
        for ids, values in state.backups:
            puzzle_emb.weights[ids] = values


def compute_embedding_cosine(model: nn.Module, identifiers: torch.Tensor, blank_id: int) -> torch.Tensor:
    """Mean pairwise cosine similarity between unique puzzle embeddings referenced by `identifiers`."""
    unique_ids = torch.unique(identifiers)
    unique_ids = unique_ids[unique_ids != blank_id]
    if unique_ids.numel() <= 1:
        return torch.zeros((), device=identifiers.device, dtype=torch.float32)

    puzzle_emb = _get_puzzle_embedding_module(model)
    if puzzle_emb is None:
        return torch.zeros((), device=identifiers.device, dtype=torch.float32)

    weight_matrix = puzzle_emb.weights[unique_ids.long()].to(torch.float32)  # type: ignore
    norms = weight_matrix.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normalized = weight_matrix / norms
    cosine_matrix = normalized @ normalized.T
    off_diag = cosine_matrix.sum() - torch.trace(cosine_matrix)
    pairs = unique_ids.numel() * (unique_ids.numel() - 1)
    return off_diag / pairs


def compute_embedding_cosine_within_task(
    model: nn.Module,
    identifiers: torch.Tensor,
    task_ids: torch.Tensor,
    blank_id: int,
    blank_task_id: int = -1,
) -> torch.Tensor:
    """Mean pairwise cosine similarity averaged over tasks, considering unique embeddings per task."""
    if task_ids is None:
        return torch.zeros((), device=identifiers.device, dtype=torch.float32)

    valid_mask = (identifiers != blank_id) & (task_ids != blank_task_id)
    if not bool(valid_mask.any()):
        return torch.zeros((), device=identifiers.device, dtype=torch.float32)

    puzzle_emb = _get_puzzle_embedding_module(model)
    if puzzle_emb is None:
        return torch.zeros((), device=identifiers.device, dtype=torch.float32)

    identifiers = identifiers[valid_mask]
    task_ids = task_ids[valid_mask]

    cosines = []
    for task in torch.unique(task_ids):
        task_mask = task_ids == task
        task_unique_ids = torch.unique(identifiers[task_mask])
        task_unique_ids = task_unique_ids[task_unique_ids != blank_id]
        if task_unique_ids.numel() <= 1:
            continue

        weight_matrix = puzzle_emb.weights[task_unique_ids.long()].to(torch.float32)  # type: ignore
        norms = weight_matrix.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = weight_matrix / norms
        cosine_matrix = normalized @ normalized.T
        off_diag = cosine_matrix.sum() - torch.trace(cosine_matrix)
        pairs = task_unique_ids.numel() * (task_unique_ids.numel() - 1)
        cosines.append(off_diag / pairs)

    if len(cosines) == 0:
        return torch.zeros((), device=identifiers.device, dtype=torch.float32)

    return torch.stack(cosines).mean()


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch_meta(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
    query_fetcher: MetaQueryFetcher,
) -> Optional[Dict[str, float]]:
    meta_cfg = config.meta_learning
    assert meta_cfg.enabled, "train_batch_meta should only run when meta-learning is enabled."

    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return None

    device = torch.device("cuda")

    support_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    query_data = query_fetcher.fetch(support_batch["puzzle_identifiers"])

    if query_data is None:
        # Fall back to support batch if no query data is available.
        query_batch = {k: v.clone() for k, v in support_batch.items()}
    else:
        query_batch = {k: tensor.to(device, non_blocking=True) for k, tensor in query_data.items()}

    support_mask = support_batch["puzzle_identifiers"] != train_state.blank_identifier_id
    support_effective_size = max(int(support_mask.sum().item()), 1)

    valid_mask = query_batch["puzzle_identifiers"] != train_state.blank_identifier_id
    query_local_size = int(valid_mask.sum().item())
    query_global_size_tensor = torch.tensor(
        [float(query_local_size)], device=device, dtype=torch.float32
    )
    if world_size > 1:
        dist.all_reduce(query_global_size_tensor, op=dist.ReduceOp.SUM)
    query_effective_size = max(int(query_global_size_tensor.item()), 1)

    params = [param for param in train_state.model.parameters()]
    param_backup = [param.detach().clone() for param in params]
    puzzle_emb = _get_puzzle_embedding_module(train_state.model)
    sparse_state = _SparseEmbeddingInnerLoopState() if puzzle_emb is not None else None

    inner_lr_trunk = float(meta_cfg.inner_lr) if meta_cfg.inner_lr is not None else float(config.lr)
    inner_lr_embed = float(meta_cfg.inner_lr_embed) if meta_cfg.inner_lr_embed is not None else float(config.puzzle_emb_lr)
    inner_steps = max(int(meta_cfg.inner_steps), 1)

    support_loss_accum = torch.zeros((), device=device, dtype=torch.float32)
    support_total_effective = torch.zeros((), device=device, dtype=torch.float32)

    for _ in range(inner_steps):
        for param in params:
            param.grad = None

        _clear_sparse_embedding_grads(train_state.model)

        with torch.device(device):
            support_carry = train_state.model.initial_carry(support_batch)  # type: ignore

        _, support_loss, _, _, _ = train_state.model(
            carry=support_carry, batch=support_batch, return_keys=[]
        )

        (support_loss / support_effective_size).backward()
        support_loss_accum += support_loss.detach().to(device=device, dtype=torch.float32)
        support_total_effective += torch.tensor(float(support_effective_size), device=device, dtype=torch.float32)

        with torch.no_grad():
            for param in params:
                if param.grad is None:
                    continue
                param.add_(param.grad, alpha=-inner_lr_trunk)

        if puzzle_emb is not None and sparse_state is not None:
            _apply_sparse_embedding_inner_step(
                puzzle_emb=puzzle_emb,
                blank_id=train_state.blank_identifier_id,
                inner_lr=inner_lr_embed,
                state=sparse_state,
            )

    for param in params:
        param.grad = None

    _clear_sparse_embedding_grads(train_state.model)

    with torch.device(device):
        query_carry = train_state.model.initial_carry(query_batch)  # type: ignore

    query_carry, query_loss, metrics, _, _ = train_state.model(
        carry=query_carry, batch=query_batch, return_keys=[]
    )

    (query_loss / query_effective_size).backward()

    with torch.no_grad():
        for param, backup in zip(params, param_backup):
            param.copy_(backup)

    if sparse_state is not None:
        _restore_sparse_embedding_from_backup(puzzle_emb, sparse_state)

    if world_size > 1:
        for param in params:
            if param.grad is not None:
                dist.all_reduce(param.grad)

    grad_embed_norm, grad_trunk_norm = compute_grad_norms(train_state.model)

    if metrics is not None:
        count_tensor = metrics.get("count")
        grad_embed = grad_embed_norm.detach()
        grad_trunk = grad_trunk_norm.detach()
        embedding_cosine = compute_embedding_cosine(
            train_state.model,
            query_batch["puzzle_identifiers"],
            train_state.blank_identifier_id,
        ).detach()
        embedding_cosine_within = compute_embedding_cosine_within_task(
            train_state.model,
            query_batch["puzzle_identifiers"],
            query_batch.get("task_identifiers"),
            train_state.blank_identifier_id,
        ).detach()

        if count_tensor is not None:
            scaling = torch.clamp(count_tensor.to(grad_embed.dtype), min=1.0)
            metrics["grad_embed_norm"] = grad_embed * scaling
            metrics["grad_trunk_norm"] = grad_trunk * scaling
            metrics["embedding_cosine"] = embedding_cosine * scaling
            metrics["embedding_cosine_within_task"] = embedding_cosine_within * scaling
        else:
            metrics["grad_embed_norm"] = grad_embed
            metrics["grad_trunk_norm"] = grad_trunk
            metrics["embedding_cosine"] = embedding_cosine
            metrics["embedding_cosine_within_task"] = embedding_cosine_within
    else:
        metrics = {}

    support_count = torch.clamp(support_total_effective, min=1.0)
    support_loss_avg = (support_loss_accum / support_count).to(device=device, dtype=query_loss.dtype)
    metrics["support_loss_avg"] = support_loss_avg.detach()
    metrics["support_effective_count"] = support_count.detach()
    metrics["meta_inner_steps"] = torch.tensor(float(inner_steps), device=device, dtype=query_loss.dtype)

    trunk_frozen = False
    if train_state.freeze_trunk_until_step is not None:
        if train_state.step <= train_state.freeze_trunk_until_step:
            trunk_frozen = True
        else:
            if train_state.trunk_frozen_active and rank == 0:
                print(f"Unfreezing trunk optimizer at step {train_state.step}")
            train_state.trunk_frozen_active = False
            train_state.freeze_trunk_until_step = None

    lr_this_step = None
    for optim, base_lr, tag in zip(
        train_state.optimizers, train_state.optimizer_lrs, train_state.optimizer_tags
    ):
        if tag == "trunk" and trunk_frozen:
            optim.zero_grad()
            continue
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step

        optim.step()
        optim.zero_grad()

    if metrics is not None and len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            raw_metric_keys = {"support_loss_avg", "meta_inner_steps", "support_effective_count"}
            raw_metrics = {}
            for key in list(reduced_metrics.keys()):
                if key in raw_metric_keys:
                    raw_metrics[key] = reduced_metrics.pop(key) / max(world_size, 1)

            count = max(reduced_metrics.get("count", query_effective_size), 1)
            outer_scale = query_effective_size
            scaled_metrics = {
                f"train/{k}": v / (outer_scale if k.endswith("loss") else count)
                for k, v in reduced_metrics.items()
            }

            for key, value in raw_metrics.items():
                scaled_metrics[f"train/{key}"] = value

            scaled_metrics["train/lr"] = lr_this_step
            return scaled_metrics

    return None

    return None


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Gradient diagnostics
    grad_embed_norm, grad_trunk_norm = compute_grad_norms(train_state.model)
    if metrics is not None:
        count_tensor = metrics.get("count")
        grad_embed = grad_embed_norm.detach()
        grad_trunk = grad_trunk_norm.detach()
        embedding_cosine = compute_embedding_cosine(
            train_state.model,
            batch["puzzle_identifiers"],
            train_state.blank_identifier_id,
        ).detach()
        embedding_cosine_within = compute_embedding_cosine_within_task(
            train_state.model,
            batch["puzzle_identifiers"],
            batch["task_identifiers"],
            train_state.blank_identifier_id,
        ).detach()

        if count_tensor is not None:
            scaling = torch.clamp(count_tensor.to(grad_embed.dtype), min=1.0)
            metrics["grad_embed_norm"] = grad_embed * scaling
            metrics["grad_trunk_norm"] = grad_trunk * scaling
            metrics["embedding_cosine"] = embedding_cosine * scaling
            metrics["embedding_cosine_within_task"] = embedding_cosine_within * scaling
        else:
            metrics["grad_embed_norm"] = grad_embed
            metrics["grad_trunk_norm"] = grad_trunk
            metrics["embedding_cosine"] = embedding_cosine
            metrics["embedding_cosine_within_task"] = embedding_cosine_within

    # Apply optimizer
    trunk_frozen = False
    if train_state.freeze_trunk_until_step is not None:
        if train_state.step <= train_state.freeze_trunk_until_step:
            trunk_frozen = True
        else:
            if train_state.trunk_frozen_active and rank == 0:
                print(f"Unfreezing trunk optimizer at step {train_state.step}")
            train_state.trunk_frozen_active = False
            train_state.freeze_trunk_until_step = None

    lr_this_step = None
    for optim, base_lr, tag in zip(train_state.optimizers, train_state.optimizer_lrs, train_state.optimizer_tags):
        if tag == "trunk" and trunk_frozen:
            optim.zero_grad()
            continue
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if metrics is not None and len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate_meta(
    config: PretrainConfig,
    train_state: TrainState,
    support_loader: Optional[torch.utils.data.DataLoader],
    support_metadata: Optional[PuzzleDatasetMetadata],
    query_fetcher: Optional[MetaQueryFetcher],
    rank: int,
    world_size: int,
) -> Optional[Dict[str, float]]:
    if support_loader is None or support_metadata is None or query_fetcher is None:
        return None
    meta_cfg = config.meta_learning
    if not meta_cfg.enabled:
        return None

    device = torch.device("cuda")
    inner_steps = max(int(meta_cfg.inner_steps), 1)
    inner_lr_trunk = float(meta_cfg.inner_lr) if meta_cfg.inner_lr is not None else float(config.lr)
    inner_lr_embed = (
        float(meta_cfg.inner_lr_embed) if meta_cfg.inner_lr_embed is not None else float(config.puzzle_emb_lr)
    )

    aggregated: Dict[str, Dict[str, torch.Tensor]] = {}

    blank_identifier_id = train_state.blank_identifier_id
    if support_metadata.blank_identifier_id is not None:
        blank_identifier_id = int(support_metadata.blank_identifier_id)

    model = train_state.model
    original_mode = model.training

    for set_name, support_batch, _global_batch_size in support_loader:
        support_batch = {k: v.cuda(non_blocking=True) for k, v in support_batch.items()}
        query_data = query_fetcher.fetch(support_batch["puzzle_identifiers"])
        if query_data is None:
            query_batch = {k: v.clone().to(device=device, non_blocking=True) for k, v in support_batch.items()}
        else:
            query_batch = {k: tensor.to(device=device, non_blocking=True) for k, tensor in query_data.items()}

        support_mask = support_batch["puzzle_identifiers"] != blank_identifier_id
        support_effective_size = max(int(support_mask.sum().item()), 1)

        params = [param for param in model.parameters()]
        param_backup = [param.detach().clone() for param in params]
        puzzle_emb = _get_puzzle_embedding_module(model)
        sparse_state = _SparseEmbeddingInnerLoopState() if puzzle_emb is not None else None

        support_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        support_effective_total = 0.0

        with torch.autograd.set_grad_enabled(True):
            model.train()
            for _ in range(inner_steps):
                for param in params:
                    param.grad = None

                _clear_sparse_embedding_grads(model)

                with torch.device(device):
                    support_carry = model.initial_carry(support_batch)  # type: ignore

                _, support_loss, _, _, _ = model(carry=support_carry, batch=support_batch, return_keys=[])

                (support_loss / support_effective_size).backward()
                support_loss_sum += support_loss.detach().to(device=device, dtype=torch.float32)
                support_effective_total += float(support_effective_size)

                with torch.no_grad():
                    for param in params:
                        if param.grad is None:
                            continue
                        param.add_(param.grad, alpha=-inner_lr_trunk)

                if puzzle_emb is not None and sparse_state is not None:
                    _apply_sparse_embedding_inner_step(
                        puzzle_emb=puzzle_emb,
                        blank_id=blank_identifier_id,
                        inner_lr=inner_lr_embed,
                        state=sparse_state,
                    )

        for param in params:
            param.grad = None
        _clear_sparse_embedding_grads(model)

        model.eval()
        with torch.device(device):
            query_carry = model.initial_carry(query_batch)  # type: ignore
        with torch.no_grad():
            query_carry, query_loss, metrics, _, _ = model(
                carry=query_carry, batch=query_batch, return_keys=[]
            )

        puzzle_ids = query_batch["puzzle_identifiers"]
        task_ids = query_batch.get("task_identifiers")
        cosine_val = compute_embedding_cosine(
            model,
            puzzle_ids,
            blank_identifier_id,
        ).detach()
        cosine_within_val = compute_embedding_cosine_within_task(
            model,
            puzzle_ids,
            task_ids,
            blank_identifier_id,
        ).detach()
        if "count" in metrics:
            count_metric = metrics["count"].clamp(min=1)
            metrics["embedding_cosine"] = cosine_val * count_metric
            metrics["embedding_cosine_within_task"] = cosine_within_val * count_metric
        else:
            metrics["embedding_cosine"] = cosine_val
            metrics["embedding_cosine_within_task"] = cosine_within_val

        support_loss_sum = support_loss_sum.detach()
        support_effective_tensor = torch.tensor(
            float(support_effective_total), device=device, dtype=torch.float32
        )
        metrics["support_loss_sum"] = support_loss_sum
        metrics["support_effective_count"] = support_effective_tensor

        aggregated_set = aggregated.setdefault(set_name, {})
        for key, value in metrics.items():
            val = value.detach()
            if key in aggregated_set:
                aggregated_set[key] += val
            else:
                aggregated_set[key] = val.clone()

        with torch.no_grad():
            for param, backup in zip(params, param_backup):
                param.copy_(backup)
        if sparse_state is not None and puzzle_emb is not None:
            _restore_sparse_embedding_from_backup(puzzle_emb, sparse_state)
        _clear_sparse_embedding_grads(model)

    model.train(original_mode)

    if not aggregated:
        return {} if rank == 0 else None

    results: Dict[str, float] = {}
    for set_name in sorted(aggregated.keys()):
        metric_dict = aggregated[set_name]
        metric_keys = sorted(metric_dict.keys())
        metric_tensor = torch.stack([metric_dict[key] for key in metric_keys])
        if world_size > 1:
            dist.reduce(metric_tensor, dst=0)
        if rank == 0:
            metric_values = metric_tensor.cpu().numpy()
            metrics_map = {key: float(metric_values[idx]) for idx, key in enumerate(metric_keys)}

            support_loss_sum = metrics_map.pop("support_loss_sum", 0.0)
            support_effective_count = metrics_map.pop("support_effective_count", 0.0)
            if support_effective_count > 0:
                results[f"meta_eval/{set_name}/support_loss_avg"] = support_loss_sum / support_effective_count
            else:
                results[f"meta_eval/{set_name}/support_loss_avg"] = 0.0
            results[f"meta_eval/{set_name}/support_tokens"] = support_effective_count

            count_value = metrics_map.pop("count", 0.0)
            if count_value > 0:
                for key, value in metrics_map.items():
                    results[f"meta_eval/{set_name}/{key}"] = value / count_value
                results[f"meta_eval/{set_name}/count"] = count_value
            else:
                for key, value in metrics_map.items():
                    results[f"meta_eval/{set_name}/{key}"] = value
                results[f"meta_eval/{set_name}/count"] = count_value

    if rank == 0:
        return results
    return None

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = None
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            # Aggregate metrics
            puzzle_ids = batch["puzzle_identifiers"]
            task_ids = batch["task_identifiers"]
            cosine_val = compute_embedding_cosine(
                train_state.model,
                puzzle_ids,
                train_state.blank_identifier_id,
            ).detach()
            cosine_within_val = compute_embedding_cosine_within_task(
                train_state.model,
                puzzle_ids,
                task_ids,
                train_state.blank_identifier_id,
            ).detach()
            if "count" in metrics:
                count = metrics["count"].clamp(min=1)
                metrics["embedding_cosine"] = cosine_val * count
                metrics["embedding_cosine_within_task"] = cosine_within_val * count
            else:
                metrics["embedding_cosine"] = cosine_val
                metrics["embedding_cosine_within_task"] = cosine_within_val

            set_id = set_ids[set_name]

            del carry, loss, preds, all_finish, batch, puzzle_ids

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Synchronize metric structures across ranks so every process participates in collectives
        if world_size > 1:
            gathered_keys = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_keys, metric_keys)
            metric_keys = next((k for k in gathered_keys if k is not None), None)

        if metric_keys is None:
            metric_keys = []

        if metric_values is None:
            metric_values = torch.zeros(
                (len(set_ids), len(metric_keys)), dtype=torch.float32, device="cuda"
            )

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                if metric_values.numel() == 0:
                    dist.barrier()
                else:
                    dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if wandb is None or wandb.run is None or config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    if wandb is not None and wandb.run is not None:
        wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    try:
        eval_loader, eval_metadata = create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.eval_global_batch_size or config.global_batch_size,
            rank=RANK,
            world_size=WORLD_SIZE,
            max_eval_augmentations=config.eval_max_augmentations,
        )
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    local_batch_size = config.global_batch_size // WORLD_SIZE

    query_fetcher = None
    if config.meta_learning.enabled:
        if eval_loader is None:
            raise ValueError("Meta-learning requires a test split for query batches.")
        query_fetcher = MetaQueryFetcher(eval_loader.dataset, local_batch_size)  # type: ignore[arg-type]

    meta_eval_support_loader: Optional[torch.utils.data.DataLoader] = None
    meta_eval_support_metadata: Optional[PuzzleDatasetMetadata] = None
    meta_eval_query_fetcher: Optional[MetaQueryFetcher] = None
    if config.meta_learning.enabled and config.meta_eval is not None:
        meta_eval_global_batch_size = (
            config.meta_eval.global_batch_size
            if config.meta_eval.global_batch_size is not None
            else (config.eval_global_batch_size or config.global_batch_size)
        )
        if meta_eval_global_batch_size % WORLD_SIZE != 0:
            raise ValueError("meta_eval.global_batch_size must be divisible by world_size.")
        meta_eval_support_loader, meta_eval_support_metadata = create_dataloader(
            config,
            "train",
            rank=RANK,
            world_size=WORLD_SIZE,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=meta_eval_global_batch_size,
            data_paths_override=config.meta_eval.data_paths,
        )
        meta_eval_query_loader, _meta_eval_query_metadata = create_dataloader(
            config,
            "test",
            rank=RANK,
            world_size=WORLD_SIZE,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=meta_eval_global_batch_size,
            max_eval_augmentations=config.meta_eval.max_eval_augmentations,
            data_paths_override=config.meta_eval.data_paths,
        )
        meta_eval_query_fetcher = MetaQueryFetcher(
            meta_eval_query_loader.dataset, meta_eval_global_batch_size // WORLD_SIZE  # type: ignore[arg-type]
        )

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        if wandb is not None:
            wandb.init(
                project=config.project_name,
                entity=config.entity,
                name=config.run_name,
                config=config.model_dump(),
                settings=wandb.Settings(_disable_stats=True)
            )  # type: ignore
            wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
            save_code_and_config(config)
        else:
            save_code_and_config(config)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            if config.meta_learning.enabled:
                assert query_fetcher is not None
                metrics = train_batch_meta(
                    config,
                    train_state,
                    batch,
                    global_batch_size,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                    query_fetcher=query_fetcher,
                )
            else:
                metrics = train_batch(
                    config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE
                )

            if RANK == 0 and metrics is not None and wandb is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)
            if (
                RANK == 0
                and config.checkpoint_every_n_steps is not None
                and train_state.step % config.checkpoint_every_n_steps == 0
            ):
                save_train_state(config, train_state)

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config, 
                train_state_eval, 
                eval_loader, 
                eval_metadata, 
                evaluators,
                rank=RANK, 
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)

            if (
                config.meta_learning.enabled
                and meta_eval_support_loader is not None
                and meta_eval_support_metadata is not None
                and meta_eval_query_fetcher is not None
            ):
                meta_metrics = evaluate_meta(
                    config,
                    train_state_eval,
                    meta_eval_support_loader,
                    meta_eval_support_metadata,
                    meta_eval_query_fetcher,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                )
                if meta_metrics is not None:
                    if metrics is None:
                        metrics = {}
                    metrics.update(meta_metrics)

            if RANK == 0 and metrics is not None and wandb is not None:
                wandb.log(metrics, step=train_state.step)
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    launch()
