from typing import Optional, Any, Sequence, List, Tuple, Dict
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
    puzzle_aug_weight_decay: float = 0.1
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


def create_dataloader(
    config: PretrainConfig,
    split: str,
    rank: int,
    world_size: int,
    max_eval_augmentations: Optional[int] = None,
    data_paths_override: Optional[List[str]] = None,
    group_offset_start: int = 0,
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
            group_offset_start=group_offset_start,
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


def create_model(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    *,
    num_task_identifiers: int,
    rank: int,
    world_size: int,
):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        num_task_identifiers=num_task_identifiers,
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
        if "DISABLE_COMPILE" not in os.environ:
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
    optimizers: List[torch.optim.Optimizer] = []
    optimizer_lrs: List[float] = []
    optimizer_tags: List[str] = []

    def _add_sparse_optimizer(module: nn.Module, weight_decay: float, tag: str) -> None:
        optimizers.append(
            CastedSparseEmbeddingSignSGD_Distributed(
                module.buffers(),  # type: ignore[arg-type]
                lr=0,
                weight_decay=weight_decay,
                world_size=world_size,
            )
        )
        optimizer_lrs.append(config.puzzle_emb_lr)
        optimizer_tags.append(tag)

    def _add_trunk_optimizer() -> None:
        optimizers.append(
            AdamAtan2(
                model.parameters(),
                lr=0.0001,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            )
        )
        optimizer_lrs.append(config.lr)
        optimizer_tags.append("trunk")

    if config.arch.puzzle_emb_ndim > 0:
        core_model = _get_core_model(model)
        inner = getattr(core_model, "inner", None) if core_model is not None else None
        task_emb_module = getattr(inner, "task_emb", None) if inner is not None else None
        delta_emb_module = None
        if inner is not None:
            delta_emb_module = getattr(inner, "aug_delta_emb", None)
            if delta_emb_module is None:
                delta_emb_module = getattr(inner, "puzzle_emb", None)
        if task_emb_module is not None:
            _add_sparse_optimizer(task_emb_module, config.puzzle_emb_weight_decay, "task_embedding")
        if delta_emb_module is not None:
            _add_sparse_optimizer(delta_emb_module, config.puzzle_aug_weight_decay, "aug_embedding")

    if config.arch.puzzle_emb_ndim == 0:
        _add_trunk_optimizer()
    elif config.freeze_weights:
        freeze_epochs = config.freeze_weights_epochs
        add_trunk_optimizer = freeze_epochs is not None and max(0, freeze_epochs) < config.epochs
        if add_trunk_optimizer:
            _add_trunk_optimizer()
    else:
        _add_trunk_optimizer()

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


def init_train_state(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
    num_task_identifiers: int,
):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs, optimizer_tags = create_model(
        config,
        train_metadata,
        num_task_identifiers=num_task_identifiers,
        rank=rank,
        world_size=world_size,
    )

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


def _align_state_dict_for_compile(
    state_dict: Dict[str, torch.Tensor], model: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Ensure checkpoint keys match whether the target model is compiled or not.
    Compiled modules prefix parameters with `_orig_mod.`; uncompiled ones do not.
    """
    model_state_keys = model.state_dict().keys()
    model_expects_prefix = any(key.startswith("_orig_mod.") for key in model_state_keys)
    ckpt_has_prefix = any(key.startswith("_orig_mod.") for key in state_dict.keys())

    # If both sides agree, nothing to do.
    if model_expects_prefix == ckpt_has_prefix:
        return state_dict

    prefix = "_orig_mod."
    remapped: Dict[str, torch.Tensor] = {}
    if ckpt_has_prefix and not model_expects_prefix:
        # Strip prefix from checkpoint keys to fit an uncompiled model.
        for key, value in state_dict.items():
            if key.startswith(prefix):
                remapped[key[len(prefix):]] = value
            else:
                remapped[key] = value
    elif model_expects_prefix and not ckpt_has_prefix:
        # Add prefix so weights load into a compiled model.
        for key, value in state_dict.items():
            if key.startswith(prefix):
                remapped[key] = value
            else:
                remapped[prefix + key] = value
    else:
        remapped = state_dict

    return remapped


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")
        state_dict = _align_state_dict_for_compile(state_dict, model)

        # Resize and reset embedding tables if needed
        embedding_specs = []
        core_model = _get_core_model(model)
        inner = getattr(core_model, "inner", None) if core_model is not None else None
        if inner is not None:
            if hasattr(inner, "task_emb"):
                embedding_specs.append(("task_emb", inner.task_emb, "base"))
            if hasattr(inner, "aug_delta_emb"):
                embedding_specs.append(("aug_delta_emb", inner.aug_delta_emb, "delta"))
            elif hasattr(inner, "puzzle_emb"):
                embedding_specs.append(("puzzle_emb", inner.puzzle_emb, "legacy"))

        strategy = getattr(config, "puzzle_emb_reinit_strategy", "mean").lower()
        legacy_keys = [
            "model.inner.puzzle_emb.weights",
            "_orig_mod.model.inner.puzzle_emb.weights",
        ]
        legacy_tensor = None
        for key in legacy_keys:
            if key in state_dict:
                legacy_tensor = state_dict[key]
                break

        for name, module, kind in embedding_specs:
            weights = getattr(module, "weights", None)
            if weights is None:
                continue
            expected_shape: torch.Size = weights.shape  # type: ignore
            possible_keys = [
                f"model.inner.{name}.weights",
                f"_orig_mod.model.inner.{name}.weights",
            ]
            for key in possible_keys:
                if key not in state_dict and kind == "delta" and legacy_tensor is not None:
                    state_dict[key] = legacy_tensor.clone()
                    legacy_tensor = None
                if key not in state_dict:
                    continue
                tensor = state_dict[key]
                if tensor.shape == expected_shape:
                    break
                print(f"Resetting {name} as shape differs. Found {tensor.shape}, Expected {expected_shape}")
                if kind in {"base", "legacy"}:
                    tensor_float = tensor.to(torch.float32)
                    mean_vec = torch.mean(tensor_float, dim=0, keepdim=True)
                    if strategy == "mean":
                        state_dict[key] = mean_vec.expand(expected_shape).to(tensor.dtype).contiguous()
                    elif strategy == "normal":
                        std_vec = torch.std(tensor_float, dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
                        noise = torch.randn(expected_shape, device=mean_vec.device, dtype=torch.float32)
                        new_weights = noise * std_vec + mean_vec
                        state_dict[key] = new_weights.to(tensor.dtype).contiguous()
                    else:
                        raise ValueError(f"Unsupported puzzle_emb_reinit_strategy: {strategy}")
                else:
                    state_dict[key] = torch.zeros(expected_shape, dtype=tensor.dtype, device=tensor.device)
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

    embed_keys = ("puzzle_emb", "aug_dihedral_emb", "aug_color_pair_emb", "task_emb", "aug_delta_emb")

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_sq = param.grad.detach().float().pow(2).sum()
        if any(key in name for key in embed_keys):
            embed_sq += grad_sq
        else:
            trunk_sq += grad_sq

    core_model = _get_core_model(model)
    if core_model is not None:
        inner = getattr(core_model, "inner", None)
        if inner is not None:
            for attr_name in ("puzzle_emb", "aug_delta_emb", "task_emb"):
                emb_module = getattr(inner, attr_name, None)
                if emb_module is None:
                    continue
                local_weights = getattr(emb_module, "local_weights", None)
                if local_weights is None:
                    continue
                local_grad = getattr(local_weights, "grad", None)
                if local_grad is not None:
                    embed_sq += local_grad.detach().float().pow(2).sum()

    return embed_sq.sqrt(), trunk_sq.sqrt()


def _get_core_model(model: nn.Module) -> Optional[nn.Module]:
    return getattr(getattr(model, "_orig_mod", model), "model", None)


def _get_embedding_weight_buffers(model: nn.Module):
    """Return (task_weights, delta_weights, blank_task_identifier)."""
    core_model = _get_core_model(model)
    inner = getattr(core_model, "inner", None) if core_model is not None else None
    if inner is None:
        return None, None, None
    task_emb = getattr(inner, "task_emb", None)
    delta_emb = getattr(inner, "aug_delta_emb", None)
    if delta_emb is None:
        delta_emb = getattr(inner, "puzzle_emb", None)
    task_weights = getattr(task_emb, "weights", None) if task_emb is not None else None
    delta_weights = getattr(delta_emb, "weights", None) if delta_emb is not None else None
    blank_task_identifier = getattr(inner, "blank_task_identifier", None)
    return task_weights, delta_weights, blank_task_identifier


def _collect_unique_embeddings(
    model: nn.Module,
    identifiers: torch.Tensor,
    task_ids: Optional[torch.Tensor],
    blank_id: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Gather combined embeddings and associated task ids for each unique puzzle identifier."""
    valid_mask = identifiers != blank_id
    if not bool(valid_mask.any()):
        return None, None

    ids = identifiers[valid_mask].to(torch.int64)
    tasks = task_ids[valid_mask].to(torch.int64) if task_ids is not None else None

    sorted_ids, sort_idx = torch.sort(ids)
    sorted_tasks = tasks[sort_idx] if tasks is not None else None
    unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    if unique_ids.numel() == 0:
        return None, None

    if sorted_tasks is not None:
        offsets = torch.cumsum(counts, dim=0) - counts
        unique_tasks = sorted_tasks[offsets]
    else:
        unique_tasks = None

    task_weights, delta_weights, blank_task_identifier = _get_embedding_weight_buffers(model)
    if delta_weights is None:
        return None, unique_tasks

    combined = delta_weights[unique_ids.long()].to(torch.float32)
    if task_weights is not None and unique_tasks is not None:
        clamped_tasks = unique_tasks
        if blank_task_identifier is not None:
            pad_tensor = torch.full_like(unique_tasks, blank_task_identifier)
            clamped_tasks = torch.where(unique_tasks >= 0, unique_tasks, pad_tensor)
        combined = combined + task_weights[clamped_tasks.long()].to(torch.float32)

    return combined, unique_tasks


def compute_embedding_cosine(
    model: nn.Module,
    identifiers: torch.Tensor,
    task_ids: Optional[torch.Tensor],
    blank_id: int,
) -> torch.Tensor:
    """Mean pairwise cosine similarity between unique puzzle embeddings referenced by `identifiers`."""
    embeddings, _ = _collect_unique_embeddings(model, identifiers, task_ids, blank_id)
    if embeddings is None or embeddings.size(0) <= 1:
        return torch.zeros((), device=identifiers.device, dtype=torch.float32)

    weight_matrix = embeddings
    norms = weight_matrix.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normalized = weight_matrix / norms
    cosine_matrix = normalized @ normalized.T
    off_diag = cosine_matrix.sum() - torch.trace(cosine_matrix)
    pairs = weight_matrix.size(0) * (weight_matrix.size(0) - 1)
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

    embeddings, unique_tasks = _collect_unique_embeddings(model, identifiers, task_ids, blank_id)
    if embeddings is None or unique_tasks is None:
        return torch.zeros((), device=identifiers.device, dtype=torch.float32)

    valid_task_mask = unique_tasks != blank_task_id
    if not bool(valid_task_mask.any()):
        return torch.zeros((), device=identifiers.device, dtype=torch.float32)

    task_values = unique_tasks[valid_task_mask]
    task_embeddings = embeddings[valid_task_mask]

    cosines: List[torch.Tensor] = []
    for task in torch.unique(task_values):
        task_mask = task_values == task
        if task_mask.sum() <= 1:
            continue
        weight_matrix = task_embeddings[task_mask]
        norms = weight_matrix.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = weight_matrix / norms
        cosine_matrix = normalized @ normalized.T
        off_diag = cosine_matrix.sum() - torch.trace(cosine_matrix)
        pairs = task_mask.sum() * (task_mask.sum() - 1)
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
            batch.get("task_identifiers"),
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
                task_ids,
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
    global wandb
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
    task_identifier_capacity = train_metadata.total_groups
    eval_group_offset = task_identifier_capacity
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
            group_offset_start=eval_group_offset,
        )
        if eval_metadata is not None:
            task_identifier_capacity = max(
                task_identifier_capacity,
                eval_group_offset + eval_metadata.total_groups,
            )
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(
        config,
        train_metadata,
        rank=RANK,
        world_size=WORLD_SIZE,
        num_task_identifiers=task_identifier_capacity,
    )

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        if wandb is not None:
            try:
                wandb.init(
                    project=config.project_name,
                    entity=config.entity,
                    name=config.run_name,
                    config=config.model_dump(),
                    settings=wandb.Settings(_disable_stats=True),
                )  # type: ignore
                wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
                save_code_and_config(config)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[WARN] Failed to initialize W&B ({exc}); disabling logging.")
                wandb = None
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
