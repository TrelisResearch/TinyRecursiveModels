from typing import Optional, Any, Sequence, List, Tuple, Dict
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
import tqdm
try:
    import wandb
except ImportError:
    wandb = None
if os.environ.get('WANDB_DISABLED', '').lower() in {'1', 'true', 'yes'}:
    wandb = None
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper
from models.losses import IGNORE_LABEL_ID
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig

class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class TestTimeAdaptConfig(pydantic.BaseModel):
    enabled: bool = False
    inner_steps: int = 0
    inner_batch_size: Optional[int] = None
    max_support_examples: Optional[int] = None
    log_interval: int = 5
    update_embeddings: bool = True

    @pydantic.model_validator(mode='after')
    def _validate(self):
        if self.enabled:
            if self.inner_steps <= 0:
                raise ValueError('inner_steps must be > 0 when test-time adaptation is enabled.')
            if self.inner_batch_size is not None and self.inner_batch_size <= 0:
                raise ValueError('inner_batch_size must be positive when provided.')
            if self.max_support_examples is not None and self.max_support_examples <= 0:
                raise ValueError('max_support_examples must be positive when provided.')
            if self.log_interval <= 0:
                raise ValueError('log_interval must be positive when provided.')
        return self

class PretrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_paths: List[str]
    data_paths_test: List[str] = []
    evaluators: List[EvaluatorConfig] = []
    halt_max_steps_eval: Optional[int] = None
    global_batch_size: int
    eval_global_batch_size: Optional[int] = None
    epochs: int
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float
    puzzle_emb_reinit_strategy: str = 'mean'
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    entity: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None
    seed: int = 0
    checkpoint_every_eval: bool = False
    checkpoint_every_n_steps: Optional[int] = None
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0
    eval_save_outputs: List[str] = []
    eval_max_augmentations: Optional[int] = None
    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False
    freeze_weights_epochs: Optional[int] = None
    disable_compile: bool = False
    test_time_adapt: Optional[TestTimeAdaptConfig] = None
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

@dataclass
class PuzzleSlice:
    dataset_key: str
    set_name: str
    puzzle_id: int
    group_id: int
    start: int
    end: int
    data: Dict[str, np.ndarray]

def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, max_eval_augmentations: Optional[int]=None, **kwargs):
    noise_prob = config.grid_noise_prob if config.grid_noise_prob is not None else getattr(config.arch, 'grid_noise_prob', 0.0)
    noise_fraction = config.grid_noise_fraction if config.grid_noise_fraction is not None else getattr(config.arch, 'grid_noise_fraction', 0.0)
    kwargs.setdefault('grid_noise_prob', noise_prob)
    kwargs.setdefault('grid_noise_fraction', noise_fraction)
    dataset = PuzzleDataset(PuzzleDatasetConfig(seed=config.seed, dataset_paths=config.data_paths_test if len(config.data_paths_test) > 0 and split == 'test' else config.data_paths, rank=rank, num_replicas=world_size, max_eval_augmentations=max_eval_augmentations, **kwargs), split=split)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=getattr(config, 'dataloader_num_workers', 4), prefetch_factor=getattr(config, 'dataloader_prefetch_factor', 8), pin_memory=getattr(config, 'dataloader_pin_memory', True), persistent_workers=getattr(config, 'dataloader_persistent_workers', True))
    return (dataloader, dataset.metadata)

def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(**config.arch.__pydantic_extra__, batch_size=config.global_batch_size // world_size, vocab_size=train_metadata.vocab_size, seq_len=train_metadata.seq_len, num_puzzle_identifiers=train_metadata.num_puzzle_identifiers, causal=False)
    if config.halt_max_steps_eval is not None:
        model_cfg.setdefault('halt_max_steps_eval', config.halt_max_steps_eval)
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    with torch.device('cuda'):
        model: nn.Module = model_cls(model_cfg)
        model = model.cuda()
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
        if getattr(config, 'disable_compile', False):
            os.environ.setdefault('DISABLE_COMPILE', '1')
        if 'DISABLE_COMPILE' not in os.environ:
            model = torch.compile(model)
        model = model.cuda()
        if rank == 0:
            load_checkpoint(model, config)
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)
    optimizer_tags: List[str]
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [AdamAtan2(model.parameters(), lr=0.0001, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))]
        optimizer_lrs = [config.lr]
        optimizer_tags = ['trunk']
    elif config.freeze_weights:
        freeze_epochs = config.freeze_weights_epochs
        add_trunk_optimizer = freeze_epochs is not None and max(0, freeze_epochs) < config.epochs
        optimizers = [CastedSparseEmbeddingSignSGD_Distributed(model.model.puzzle_emb.buffers(), lr=0, weight_decay=config.puzzle_emb_weight_decay, world_size=world_size)]
        optimizer_lrs = [config.puzzle_emb_lr]
        optimizer_tags = ['embedding']
        if add_trunk_optimizer:
            optimizers.append(AdamAtan2(model.parameters(), lr=0.0001, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2)))
            optimizer_lrs.append(config.lr)
            optimizer_tags.append('trunk')
    else:
        optimizers = [CastedSparseEmbeddingSignSGD_Distributed(model.model.puzzle_emb.buffers(), lr=0, weight_decay=config.puzzle_emb_weight_decay, world_size=world_size), AdamAtan2(model.parameters(), lr=0.0001, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]
        optimizer_tags = ['embedding', 'trunk']
    return (model, optimizers, optimizer_lrs, optimizer_tags)

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0] * sd[0][k].to(device)
        for i in range(1, len(nets)):
            comb_net += alpha[i] * sd[i][k].to(device)
        sd_alpha[k] = comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float=0.0, num_cycles: float=0.5):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))

def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
    (model, optimizers, optimizer_lrs, optimizer_tags) = create_model(config, train_metadata, rank=rank, world_size=world_size)
    freeze_trunk_until_step: Optional[int] = None
    if config.freeze_weights and any((tag == 'trunk' for tag in optimizer_tags)):
        freeze_epochs = config.freeze_weights_epochs
        if freeze_epochs is not None:
            freeze_epochs = max(0, freeze_epochs)
            if freeze_epochs >= config.epochs:
                freeze_trunk_until_step = total_steps
            elif config.epochs > 0:
                freeze_ratio = freeze_epochs / config.epochs
                freeze_trunk_until_step = max(0, math.ceil(total_steps * freeze_ratio))
    return TrainState(step=0, total_steps=total_steps, model=model, optimizers=optimizers, optimizer_lrs=optimizer_lrs, optimizer_tags=optimizer_tags, carry=None, blank_identifier_id=train_metadata.blank_identifier_id, freeze_trunk_until_step=freeze_trunk_until_step, trunk_frozen_active=freeze_trunk_until_step is not None and freeze_trunk_until_step > 0)

def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f'step_{train_state.step}'))

def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f'Loading checkpoint {config.load_checkpoint}')
        state_dict = torch.load(config.load_checkpoint, map_location='cuda')
        puzzle_emb_name = '_orig_mod.model.inner.puzzle_emb.weights'
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f'Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}')
                strategy = getattr(config, 'puzzle_emb_reinit_strategy', 'mean').lower()
                puzzle_emb_float = puzzle_emb.to(torch.float32)
                mean_vec = torch.mean(puzzle_emb_float, dim=0, keepdim=True)
                if strategy == 'mean':
                    state_dict[puzzle_emb_name] = mean_vec.expand(expected_shape).to(puzzle_emb.dtype).contiguous()
                elif strategy == 'normal':
                    std_vec = torch.std(puzzle_emb_float, dim=0, keepdim=True, unbiased=False).clamp_min(1e-06)
                    noise = torch.randn(expected_shape, device=mean_vec.device, dtype=torch.float32)
                    new_weights = noise * std_vec + mean_vec
                    state_dict[puzzle_emb_name] = new_weights.to(puzzle_emb.dtype).contiguous()
                else:
                    raise ValueError(f'Unsupported puzzle_emb_reinit_strategy: {strategy}')
        (missing_keys, unexpected_keys) = model.load_state_dict(state_dict, strict=False, assign=True)
        if len(unexpected_keys):
            print(f"Warning: unexpected checkpoint keys skipped: {unexpected_keys[:5]}{('...' if len(unexpected_keys) > 5 else '')}")
        if len(missing_keys):
            print(f"Warning: missing checkpoint keys initialized from defaults: {missing_keys[:5]}{('...' if len(missing_keys) > 5 else '')}")

def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(current_step=train_state.step, base_lr=base_lr, num_warmup_steps=round(config.lr_warmup_steps), num_training_steps=train_state.total_steps, min_ratio=config.lr_min_ratio)

def compute_grad_norms(model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (embedding_grad_norm, trunk_grad_norm)."""
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    embed_sq = torch.zeros((), device=device, dtype=torch.float32)
    trunk_sq = torch.zeros((), device=device, dtype=torch.float32)
    embed_keys = ('puzzle_emb', 'aug_dihedral_emb', 'aug_color_pair_emb')
    for (name, param) in model.named_parameters():
        if param.grad is None:
            continue
        grad_sq = param.grad.detach().float().pow(2).sum()
        if any((key in name for key in embed_keys)):
            embed_sq += grad_sq
        else:
            trunk_sq += grad_sq
    core_model = getattr(model, 'model', None)
    if core_model is not None:
        inner = getattr(core_model, 'inner', None)
        puzzle_emb = getattr(inner, 'puzzle_emb', None)
        if puzzle_emb is not None and getattr(puzzle_emb, 'local_weights', None) is not None:
            local_grad = puzzle_emb.local_weights.grad
            if local_grad is not None:
                embed_sq += local_grad.detach().float().pow(2).sum()
    return (embed_sq.sqrt(), trunk_sq.sqrt())

def _get_puzzle_embedding_module(model: nn.Module):
    core_model = getattr(model, 'model', None)
    inner = getattr(core_model, 'inner', None) if core_model is not None else None
    return getattr(inner, 'puzzle_emb', None) if inner is not None else None

def compute_embedding_cosine(model: nn.Module, identifiers: torch.Tensor, blank_id: int) -> torch.Tensor:
    """Mean pairwise cosine similarity between unique puzzle embeddings referenced by `identifiers`."""
    unique_ids = torch.unique(identifiers)
    unique_ids = unique_ids[unique_ids != blank_id]
    if unique_ids.numel() <= 1:
        return torch.zeros((), device=identifiers.device, dtype=torch.float32)
    puzzle_emb = _get_puzzle_embedding_module(model)
    if puzzle_emb is None:
        return torch.zeros((), device=identifiers.device, dtype=torch.float32)
    weight_matrix = puzzle_emb.weights[unique_ids.long()].to(torch.float32)
    norms = weight_matrix.norm(dim=-1, keepdim=True).clamp_min(1e-06)
    normalized = weight_matrix / norms
    cosine_matrix = normalized @ normalized.T
    off_diag = cosine_matrix.sum() - torch.trace(cosine_matrix)
    pairs = unique_ids.numel() * (unique_ids.numel() - 1)
    return off_diag / pairs

def compute_embedding_cosine_within_task(model: nn.Module, identifiers: torch.Tensor, task_ids: torch.Tensor, blank_id: int, blank_task_id: int=-1) -> torch.Tensor:
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
        weight_matrix = puzzle_emb.weights[task_unique_ids.long()].to(torch.float32)
        norms = weight_matrix.norm(dim=-1, keepdim=True).clamp_min(1e-06)
        normalized = weight_matrix / norms
        cosine_matrix = normalized @ normalized.T
        off_diag = cosine_matrix.sum() - torch.trace(cosine_matrix)
        pairs = task_unique_ids.numel() * (task_unique_ids.numel() - 1)
        cosines.append(off_diag / pairs)
    if len(cosines) == 0:
        return torch.zeros((), device=identifiers.device, dtype=torch.float32)
    return torch.stack(cosines).mean()

def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, 'evaluators.')(data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__)
            evaluators.append(cls)
    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return
    batch = {k: v.cuda() for (k, v) in batch.items()}
    if train_state.carry is None:
        with torch.device('cuda'):
            train_state.carry = train_state.model.initial_carry(batch)
    (train_state.carry, loss, metrics, _, _) = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
    (1 / global_batch_size * loss).backward()
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
    (grad_embed_norm, grad_trunk_norm) = compute_grad_norms(train_state.model)
    if metrics is not None:
        count_tensor = metrics.get('count')
        grad_embed = grad_embed_norm.detach()
        grad_trunk = grad_trunk_norm.detach()
        embedding_cosine = compute_embedding_cosine(train_state.model, batch['puzzle_identifiers'], train_state.blank_identifier_id).detach()
        embedding_cosine_within = compute_embedding_cosine_within_task(train_state.model, batch['puzzle_identifiers'], batch['task_identifiers'], train_state.blank_identifier_id).detach()
        if count_tensor is not None:
            scaling = torch.clamp(count_tensor.to(grad_embed.dtype), min=1.0)
            metrics['grad_embed_norm'] = grad_embed * scaling
            metrics['grad_trunk_norm'] = grad_trunk * scaling
            metrics['embedding_cosine'] = embedding_cosine * scaling
            metrics['embedding_cosine_within_task'] = embedding_cosine_within * scaling
        else:
            metrics['grad_embed_norm'] = grad_embed
            metrics['grad_trunk_norm'] = grad_trunk
            metrics['embedding_cosine'] = embedding_cosine
            metrics['embedding_cosine_within_task'] = embedding_cosine_within
    trunk_frozen = False
    if train_state.freeze_trunk_until_step is not None:
        if train_state.step <= train_state.freeze_trunk_until_step:
            trunk_frozen = True
        else:
            if train_state.trunk_frozen_active and rank == 0:
                print(f'Unfreezing trunk optimizer at step {train_state.step}')
            train_state.trunk_frozen_active = False
            train_state.freeze_trunk_until_step = None
    lr_this_step = None
    for (optim, base_lr, tag) in zip(train_state.optimizers, train_state.optimizer_lrs, train_state.optimizer_tags):
        if tag == 'trunk' and trunk_frozen:
            optim.zero_grad()
            continue
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()
    if metrics is not None and len(metrics):
        assert not any((v.requires_grad for v in metrics.values()))
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)
        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for (i, k) in enumerate(metric_keys)}
            count = max(reduced_metrics['count'], 1)
            reduced_metrics = {f'train/{k}': v / (global_batch_size if k.endswith('loss') else count) for (k, v) in reduced_metrics.items()}
            reduced_metrics['train/lr'] = lr_this_step
            return reduced_metrics

def evaluate_standard(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, evaluators: List[Any], rank: int, world_size: int, cpu_group: Optional[dist.ProcessGroup]):
    reduced_metrics = None
    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)
        set_ids = {k: idx for (idx, k) in enumerate(eval_metadata.sets)}
        save_preds = {}
        metric_keys = None
        metric_values = None
        carry = None
        processed_batches = 0
        for (set_name, batch, global_batch_size) in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f'Processing batch {processed_batches}: {set_name}')
            batch = {k: v.cuda() for (k, v) in batch.items()}
            with torch.device('cuda'):
                carry = train_state.model.initial_carry(batch)
            inference_steps = 0
            while True:
                (carry, loss, metrics, preds, all_finish) = train_state.model(carry=carry, batch=batch, return_keys=return_keys)
                inference_steps += 1
                if all_finish:
                    break
            if rank == 0:
                print(f'  Completed inference in {inference_steps} steps')
            for collection in (batch, preds):
                for (k, v) in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())
            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)
            puzzle_ids = batch['puzzle_identifiers']
            task_ids = batch['task_identifiers']
            cosine_val = compute_embedding_cosine(train_state.model, puzzle_ids, train_state.blank_identifier_id).detach()
            cosine_within_val = compute_embedding_cosine_within_task(train_state.model, puzzle_ids, task_ids, train_state.blank_identifier_id).detach()
            if 'count' in metrics:
                count = metrics['count'].clamp(min=1)
                metrics['embedding_cosine'] = cosine_val * count
                metrics['embedding_cosine_within_task'] = cosine_within_val * count
            else:
                metrics['embedding_cosine'] = cosine_val
                metrics['embedding_cosine_within_task'] = cosine_within_val
            set_id = set_ids[set_name]
            del carry, loss, preds, all_finish, batch, puzzle_ids
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device='cuda')
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            del metrics
        save_preds = {k: torch.cat(v, dim=0) for (k, v) in save_preds.items()}
        if config.checkpoint_path is not None and len(save_preds):
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(save_preds, os.path.join(config.checkpoint_path, f'step_{train_state.step}_all_preds.{rank}'))
        del save_preds
        if world_size > 1:
            gathered_keys = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_keys, metric_keys)
            metric_keys = next((k for k in gathered_keys if k is not None), None)
        if metric_keys is None:
            metric_keys = []
        if metric_values is None:
            metric_values = torch.zeros((len(set_ids), len(metric_keys)), dtype=torch.float32, device='cuda')
        if metric_values is not None:
            if world_size > 1:
                if metric_values.numel() == 0:
                    dist.barrier()
                else:
                    dist.reduce(metric_values, dst=0)
            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {set_name: {metric_name: reduced_metrics[set_id, metric_id] for (metric_id, metric_name) in enumerate(metric_keys)} for (set_id, set_name) in enumerate(set_ids)}
                for (set_name, m) in reduced_metrics.items():
                    count = m.pop('count')
                    reduced_metrics[set_name] = {k: v / count for (k, v) in m.items()}
        if rank == 0:
            print(f'\nRunning {len(evaluators)} evaluator(s)...')
        for (i, evaluator) in enumerate(evaluators):
            if rank == 0:
                print(f'Running evaluator {i + 1}/{len(evaluators)}: {evaluator.__class__.__name__}')
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(config.checkpoint_path, f'evaluator_{evaluator.__class__.__name__}_step_{train_state.step}')
                os.makedirs(evaluator_save_path, exist_ok=True)
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}
                reduced_metrics.update(metrics)
                print(f'  Completed {evaluator.__class__.__name__}')
        if rank == 0:
            print('All evaluators completed!')
    return reduced_metrics

def _build_puzzle_lookup(dataset: PuzzleDataset) -> Tuple[Dict[int, List[PuzzleSlice]], List[PuzzleSlice]]:
    (data_map, key_info) = dataset.get_loaded_data()
    puzzle_lookup: Dict[int, List[PuzzleSlice]] = {}
    ordered: List[PuzzleSlice] = []

    def _sort_key(key: str):
        info = key_info[key]
        return (info['dataset_index'], info['set_name'], key)
    for key in sorted(data_map.keys(), key=_sort_key):
        info = key_info[key]
        data = data_map[key]
        puzzle_ids = data['puzzle_identifiers']
        puzzle_indices = data['puzzle_indices']
        puzzle_group_ids = data['puzzle_group_ids']
        num_puzzles = puzzle_ids.shape[0]
        for puzzle_idx in range(num_puzzles):
            start = int(puzzle_indices[puzzle_idx])
            end = int(puzzle_indices[puzzle_idx + 1])
            slice_info = PuzzleSlice(dataset_key=key, set_name=info['set_name'], puzzle_id=int(puzzle_ids[puzzle_idx]), group_id=int(puzzle_group_ids[puzzle_idx]), start=start, end=end, data=data)
            puzzle_lookup.setdefault(slice_info.puzzle_id, []).append(slice_info)
            ordered.append(slice_info)
    return (puzzle_lookup, ordered)

def _collect_examples_for_puzzle(slices: List[PuzzleSlice], max_examples: Optional[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    if not slices:
        return (None, None, None)
    inputs_list = []
    labels_list = []
    for slice_info in slices:
        if slice_info.start >= slice_info.end:
            continue
        inputs_list.append(slice_info.data['inputs'][slice_info.start:slice_info.end])
        labels_list.append(slice_info.data['labels'][slice_info.start:slice_info.end])
    if not inputs_list:
        return (None, None, None)
    inputs = np.concatenate(inputs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    if max_examples is not None and inputs.shape[0] > max_examples:
        inputs = inputs[:max_examples]
        labels = labels[:max_examples]
    return (inputs.astype(np.int32, copy=False), labels.astype(np.int32, copy=False), slices[0].group_id)

def _create_distributed_batches(inputs: Optional[np.ndarray], labels: Optional[np.ndarray], *, puzzle_id: int, group_id: int, global_batch_size: int, pad_id: int, blank_id: int, ignore_label_id: Optional[int], rank: int, world_size: int) -> List[Tuple[Dict[str, torch.Tensor], int]]:
    if inputs is None or labels is None or inputs.shape[0] == 0:
        return []
    assert global_batch_size % world_size == 0, 'Inner batch size must divide across replicas.'
    local_batch_size = global_batch_size // world_size
    batches: List[Tuple[Dict[str, torch.Tensor], int]] = []
    total_examples = inputs.shape[0]
    pointer = 0
    seq_len = inputs.shape[1]
    while pointer < total_examples:
        end = min(pointer + global_batch_size, total_examples)
        chunk_inputs = inputs[pointer:end]
        chunk_labels = labels[pointer:end]
        actual = int(chunk_inputs.shape[0])
        if actual < global_batch_size:
            pad_count = global_batch_size - actual
            pad_inputs = np.full((pad_count, seq_len), pad_id, dtype=chunk_inputs.dtype)
            pad_labels = np.full((pad_count, seq_len), IGNORE_LABEL_ID, dtype=chunk_labels.dtype)
            chunk_inputs = np.concatenate([chunk_inputs, pad_inputs], axis=0)
            chunk_labels = np.concatenate([chunk_labels, pad_labels], axis=0)
        if ignore_label_id is not None:
            np.putmask(chunk_labels, chunk_labels == ignore_label_id, IGNORE_LABEL_ID)
        puzzle_ids = np.full(global_batch_size, puzzle_id, dtype=np.int32)
        task_ids = np.full(global_batch_size, group_id, dtype=np.int32)
        if actual < global_batch_size:
            puzzle_ids[actual:] = blank_id
            task_ids[actual:] = -1
        local_start = rank * local_batch_size
        local_end = local_start + local_batch_size
        batch = {'inputs': torch.from_numpy(np.ascontiguousarray(chunk_inputs[local_start:local_end])), 'labels': torch.from_numpy(np.ascontiguousarray(chunk_labels[local_start:local_end])), 'puzzle_identifiers': torch.from_numpy(np.ascontiguousarray(puzzle_ids[local_start:local_end])), 'task_identifiers': torch.from_numpy(np.ascontiguousarray(task_ids[local_start:local_end]))}
        batches.append((batch, actual))
        pointer += actual
    return batches

def _create_inner_optimizers(
    optimizer_specs: Sequence[Tuple[str, float]],
    model: nn.Module,
    config: PretrainConfig,
    adapt_config: TestTimeAdaptConfig,
    world_size: int,
) -> Tuple[List[torch.optim.Optimizer], List[float], List[str]]:
    optimizers: List[torch.optim.Optimizer] = []
    optimizer_lrs: List[float] = []
    optimizer_tags: List[str] = []

    for tag, base_lr in optimizer_specs:
        if tag == "embedding":
            if not adapt_config.update_embeddings:
                continue
            optimizer = CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=base_lr,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            )
        elif tag == "trunk":
            optimizer = AdamAtan2(
                model.parameters(),
                lr=base_lr,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            )
        else:
            continue

        optimizers.append(optimizer)
        optimizer_lrs.append(base_lr)
        optimizer_tags.append(tag)

    for optimizer in optimizers:
        optimizer.zero_grad()

    return optimizers, optimizer_lrs, optimizer_tags

def evaluate_with_adaptation(config: PretrainConfig, train_state: TrainState, eval_loader: Optional[torch.utils.data.DataLoader], eval_metadata: PuzzleDatasetMetadata, evaluators: List[Any], rank: int, world_size: int, cpu_group: Optional[dist.ProcessGroup]) -> Optional[Dict[str, Any]]:
    adapt_config = config.test_time_adapt
    assert adapt_config is not None and adapt_config.enabled

    global_inner_batch = adapt_config.inner_batch_size or config.global_batch_size
    eval_global_batch = config.eval_global_batch_size or config.global_batch_size

    if global_inner_batch <= 0:
        raise ValueError("inner_batch_size must be positive once test-time adaptation is enabled.")
    if eval_global_batch <= 0:
        raise ValueError("eval_global_batch_size must be positive.")
    if global_inner_batch % world_size != 0:
        raise ValueError("inner_batch_size must be divisible by world_size.")
    if eval_global_batch % world_size != 0:
        raise ValueError("eval_global_batch_size must be divisible by world_size.")

    if rank == 0:
        print(
            f"Test-time adaptation enabled: steps={adapt_config.inner_steps}, "
            f"inner_batch={global_inner_batch}, update_embeddings={adapt_config.update_embeddings}"
        )

    support_dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=config.data_paths,
            global_batch_size=config.global_batch_size,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=rank,
            num_replicas=world_size,
        ),
        split="train",
    )
    query_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    query_dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=query_paths,
            global_batch_size=eval_global_batch,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=rank,
            num_replicas=world_size,
            max_eval_augmentations=config.eval_max_augmentations,
        ),
        split="test",
    )

    support_lookup, _ = _build_puzzle_lookup(support_dataset)
    _, query_slices = _build_puzzle_lookup(query_dataset)

    base_state = {k: v.detach().clone() for k, v in train_state.model.state_dict().items()}

    return_keys = set(config.eval_save_outputs)
    for evaluator in evaluators:
        evaluator.begin_eval()
        return_keys.update(evaluator.required_outputs)

    set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

    save_preds: Dict[str, List[torch.Tensor]] = {}
    metric_keys = None
    metric_values = None
    reduced_metrics: Optional[Dict[str, Dict[str, float]]] = None

    total_tasks = len(query_slices)

    max_global_batch = max(global_inner_batch, eval_global_batch)
    local_task_batch = max_global_batch // world_size

    task_model_cls = load_model_class(config.arch.name)
    task_loss_cls = load_model_class(config.arch.loss.name)

    for task_idx, query_slice in enumerate(query_slices):
        puzzle_id = query_slice.puzzle_id
        set_name = query_slice.set_name
        support_slices = support_lookup.get(puzzle_id, [])

        task_model_cfg = dict(
            **config.arch.__pydantic_extra__,  # type: ignore
            batch_size=local_task_batch,
            vocab_size=eval_metadata.vocab_size,
            seq_len=eval_metadata.seq_len,
            num_puzzle_identifiers=eval_metadata.num_puzzle_identifiers,
            causal=False,
        )
        with torch.device("cuda"):
            task_model: nn.Module = task_model_cls(task_model_cfg)
            task_model = task_loss_cls(task_model, **config.arch.loss.__pydantic_extra__)  # type: ignore
            task_model.cuda()
            task_model.load_state_dict(base_state, strict=True)
        task_model.zero_grad(set_to_none=True)
        task_model.eval()

        support_inputs, support_labels, support_group = _collect_examples_for_puzzle(
            support_slices, adapt_config.max_support_examples
        )
        query_inputs = query_slice.data["inputs"][query_slice.start : query_slice.end].astype(np.int32, copy=False)
        query_labels = query_slice.data["labels"][query_slice.start : query_slice.end].astype(np.int32, copy=False)
        query_group = query_slice.group_id
        task_group = support_group if support_group is not None else query_group

        support_batches = _create_distributed_batches(
            support_inputs,
            support_labels,
            puzzle_id=puzzle_id,
            group_id=task_group,
            global_batch_size=global_inner_batch,
            pad_id=eval_metadata.pad_id,
            blank_id=train_state.blank_identifier_id,
            ignore_label_id=eval_metadata.ignore_label_id,
            rank=rank,
            world_size=world_size,
        )
        query_batches = _create_distributed_batches(
            query_inputs,
            query_labels,
            puzzle_id=puzzle_id,
            group_id=query_group,
            global_batch_size=eval_global_batch,
            pad_id=eval_metadata.pad_id,
            blank_id=train_state.blank_identifier_id,
            ignore_label_id=eval_metadata.ignore_label_id,
            rank=rank,
            world_size=world_size,
        )

        support_count = int(support_inputs.shape[0]) if support_inputs is not None else 0
        query_count = int(query_inputs.shape[0])

        if rank == 0:
            print(
                f"[ADAPT] Task {task_idx + 1}/{total_tasks} set={set_name} "
                f"puzzle_id={puzzle_id} support={support_count} query={query_count}"
            )

        if not query_batches:
            if rank == 0:
                print("  No query examples available; skipping task.")
            continue

        loss_before = None
        if query_count > 0:
            with torch.inference_mode():
                loss_total = 0.0
                count_total = 0
                for batch_cpu, actual in query_batches:
                    if actual == 0:
                        continue
                    batch = {k: v.cuda() for k, v in batch_cpu.items()}
                    carry = task_model.initial_carry(batch)  # type: ignore
                    inference_steps = 0
                    while True:
                        carry, loss, _, _, all_finish = task_model(carry=carry, batch=batch, return_keys=[])
                        inference_steps += 1
                        if all_finish:
                            break
                    loss_total += float(loss.detach().cpu())
                    count_total += actual
                loss_before = loss_total / max(count_total, 1)

        step_logs: List[Dict[str, float]] = []

        if support_batches and adapt_config.inner_steps > 0:
            task_model.train()
            inner_optimizers, inner_lrs, inner_tags = _create_inner_optimizers(
                zip(train_state.optimizer_tags, train_state.optimizer_lrs),
                task_model,
                config,
                adapt_config,
                world_size,
            )
            inner_carry = None
            for inner_step in range(adapt_config.inner_steps):
                batch_cpu, effective_count = support_batches[inner_step % len(support_batches)]
                if effective_count == 0:
                    continue

                batch = {k: v.cuda() for k, v in batch_cpu.items()}
                if inner_carry is None:
                    inner_carry = task_model.initial_carry(batch)  # type: ignore

                inner_carry, inner_loss, _, _, _ = task_model(carry=inner_carry, batch=batch, return_keys=[])
                ((1 / max(effective_count, 1)) * inner_loss).backward()

                if world_size > 1:
                    for param in task_model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad)

                grad_embed_norm, grad_trunk_norm = compute_grad_norms(task_model)

                for optimizer, base_lr in zip(inner_optimizers, inner_lrs):
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = base_lr
                    optimizer.step()
                    optimizer.zero_grad()

                if (
                    (inner_step + 1) % adapt_config.log_interval == 0
                    or inner_step == adapt_config.inner_steps - 1
                ):
                    step_entry = {
                        "step": float(inner_step + 1),
                        "loss": float(inner_loss.detach().cpu()),
                        "grad_embed": float(grad_embed_norm.detach().cpu()),
                        "grad_trunk": float(grad_trunk_norm.detach().cpu()),
                    }
                    step_logs.append(step_entry)
                    if rank == 0:
                        print(
                            f"    inner_step={inner_step + 1} "
                            f"loss={step_entry['loss']:.4f} "
                            f"grad_embed={step_entry['grad_embed']:.4f} "
                            f"grad_trunk={step_entry['grad_trunk']:.4f}"
                        )
                    if rank == 0 and wandb is not None:
                        wandb.log(
                            {
                                "eval/adapt_inner_loss": step_entry["loss"],
                                "eval/adapt_inner_grad_embed": step_entry["grad_embed"],
                                "eval/adapt_inner_grad_trunk": step_entry["grad_trunk"],
                                "eval/adapt_inner_step": inner_step + 1,
                                "eval/adapt_puzzle_id": puzzle_id,
                            },
                            step=train_state.step,
                        )

            task_model.eval()
        else:
            task_model.eval()
            if rank == 0 and adapt_config.inner_steps > 0:
                print("  No support examples; using base weights.")

        loss_after_total = 0.0
        count_after_total = 0

        for batch_cpu, actual in query_batches:
            batch = {k: v.cuda() for k, v in batch_cpu.items()}
            with torch.inference_mode():
                carry = task_model.initial_carry(batch)  # type: ignore
                inference_steps = 0
                while True:
                    carry, loss, metrics, preds, all_finish = task_model(
                        carry=carry, batch=batch, return_keys=return_keys
                    )
                    inference_steps += 1
                    if all_finish:
                        break

            loss_after_total += float(loss.detach().cpu())
            count_after_total += actual

            for collection in (
                batch_cpu,
                {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in preds.items()},
            ):
                for key, value in collection.items():
                    if key in config.eval_save_outputs:
                        save_preds.setdefault(key, [])
                        if isinstance(value, torch.Tensor):
                            save_preds[key].append(value.cpu())

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            puzzle_ids_tensor = batch["puzzle_identifiers"]
            task_ids_tensor = batch["task_identifiers"]
            cosine_val = compute_embedding_cosine(
                task_model,
                puzzle_ids_tensor,
                train_state.blank_identifier_id,
            ).detach()
            cosine_within_val = compute_embedding_cosine_within_task(
                task_model,
                puzzle_ids_tensor,
                task_ids_tensor,
                train_state.blank_identifier_id,
            ).detach()
            if "count" in metrics:
                count_tensor = metrics["count"].clamp(min=1)
                metrics["embedding_cosine"] = cosine_val * count_tensor
                metrics["embedding_cosine_within_task"] = cosine_within_val * count_tensor
            else:
                metrics["embedding_cosine"] = cosine_val
                metrics["embedding_cosine_within_task"] = cosine_within_val

            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

        loss_after = loss_after_total / max(count_after_total, 1)

        if rank == 0:
            summary_log = {
                "eval/adapt_task_loss_before": loss_before if loss_before is not None else float("nan"),
                "eval/adapt_task_loss_after": loss_after,
                "eval/adapt_support_examples": support_count,
                "eval/adapt_query_examples": query_count,
                "eval/adapt_inner_steps": adapt_config.inner_steps if support_batches else 0,
                "eval/adapt_puzzle_id": puzzle_id,
            }
            print(
                f"  loss_before={summary_log['eval/adapt_task_loss_before']:.4f} "
                f"loss_after={summary_log['eval/adapt_task_loss_after']:.4f}"
            )
            if wandb is not None:
                wandb.log(summary_log, step=train_state.step)

        del task_model

    if config.checkpoint_path is not None and len(save_preds):
        os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
        torch.save(
            {k: torch.cat(v, dim=0) for k, v in save_preds.items()},
            os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"),
        )

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

    if metric_values.numel() > 0 and world_size > 1:
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

        for set_name, metrics_dict in reduced_metrics.items():
            count = metrics_dict.pop("count", 1.0)
            reduced_metrics[set_name] = {k: v / max(count, 1.0) for k, v in metrics_dict.items()}

    if rank == 0:
        print(f"\nRunning {len(evaluators)} evaluator(s)...")

    for idx, evaluator in enumerate(evaluators):
        if rank == 0:
            print(f"Running evaluator {idx + 1}/{len(evaluators)}: {evaluator.__class__.__name__}")

        evaluator_save_path = None
        if config.checkpoint_path is not None:
            evaluator_save_path = os.path.join(
                config.checkpoint_path,
                f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
            )
            os.makedirs(evaluator_save_path, exist_ok=True)

        metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
        if rank == 0 and metrics is not None:
            if reduced_metrics is None:
                reduced_metrics = {}
            reduced_metrics.update(metrics)
            print(f"  Completed {evaluator.__class__.__name__}")

    if rank == 0:
        print("All evaluators completed!")

    return reduced_metrics
def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: Optional[torch.utils.data.DataLoader], eval_metadata: PuzzleDatasetMetadata, evaluators: List[Any], rank: int, world_size: int, cpu_group: Optional[dist.ProcessGroup]):
    if config.test_time_adapt is not None and config.test_time_adapt.enabled:
        return evaluate_with_adaptation(config, train_state, eval_loader, eval_metadata, evaluators, rank, world_size, cpu_group)
    return evaluate_standard(config, train_state, eval_loader, eval_metadata, evaluators, rank, world_size, cpu_group)

def save_code_and_config(config: PretrainConfig):
    if wandb is None or wandb.run is None or config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    code_list = [get_model_source_path(config.arch.name), get_model_source_path(config.arch.loss.name)]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))
    config_file = os.path.join(config.checkpoint_path, 'all_config.yaml')
    with open(config_file, 'wt') as f:
        yaml.dump(config.model_dump(), f)
    if wandb is not None and wandb.run is not None:
        wandb.run.log_code(config.checkpoint_path)

def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)
        if config.project_name is None:
            config.project_name = f'{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch'
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join('checkpoints', config.project_name, config.run_name)
        objects = [config]
    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)
    return objects[0]

@hydra.main(config_path='config', config_name='cfg_pretrain', version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None
    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        CPU_PROCESS_GROUP = dist.new_group(backend='gloo')
        assert dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    torch.random.manual_seed(config.seed + RANK)
    if config.eval_interval is not None and config.eval_interval > 0:
        train_epochs_per_iter = config.eval_interval
    else:
        train_epochs_per_iter = max(config.epochs, 1)
    if config.epochs > 0:
        total_iters = config.epochs // train_epochs_per_iter
        assert train_epochs_per_iter > 0 and config.epochs % train_epochs_per_iter == 0, 'Eval interval must be a divisor of total epochs.'
    else:
        total_iters = 0
    (train_loader, train_metadata) = create_dataloader(config, 'train', test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        (eval_loader, eval_metadata) = create_dataloader(config, 'test', test_set_mode=True, epochs_per_iter=1, global_batch_size=config.eval_global_batch_size or config.global_batch_size, rank=RANK, world_size=WORLD_SIZE, max_eval_augmentations=config.eval_max_augmentations)
    except:
        print('NO EVAL DATA FOUND')
        eval_loader = eval_metadata = None
    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print('No evaluator found')
        evaluators = []
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        if wandb is not None:
            wandb.init(project=config.project_name, entity=config.entity, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))
            wandb.log({'num_params': sum((x.numel() for x in train_state.model.parameters()))}, step=0)
            save_code_and_config(config)
        else:
            save_code_and_config(config)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)
    for _iter_id in range(total_iters):
        print(f'[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}')
        if RANK == 0:
            print('TRAIN')
        train_state.model.train()
        for (set_name, batch, global_batch_size) in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)
            if RANK == 0 and metrics is not None and (wandb is not None):
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)
            if config.ema:
                ema_helper.update(train_state.model)
            if RANK == 0 and config.checkpoint_every_n_steps is not None and (train_state.step % config.checkpoint_every_n_steps == 0):
                save_train_state(config, train_state)
        if _iter_id >= config.min_eval_interval:
            if RANK == 0:
                print('EVALUATE')
            if config.ema:
                print('SWITCH TO EMA')
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config, train_state_eval, eval_loader, eval_metadata, evaluators, rank=RANK, world_size=WORLD_SIZE, cpu_group=CPU_PROCESS_GROUP)
            if RANK == 0 and metrics is not None and (wandb is not None):
                wandb.log(metrics, step=train_state.step)
            if RANK == 0:
                print('SAVE CHECKPOINT')
            if RANK == 0 and (config.checkpoint_every_eval or _iter_id == total_iters - 1):
                save_train_state(config, train_state_eval)
            if config.ema:
                del train_state_eval
    if total_iters == 0:
        if RANK == 0:
            print('EVALUATE')
        if config.ema:
            print('SWITCH TO EMA')
            train_state_eval = copy.deepcopy(train_state)
            train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
        else:
            train_state_eval = train_state
        train_state_eval.model.eval()
        metrics = evaluate(config, train_state_eval, eval_loader, eval_metadata, evaluators, rank=RANK, world_size=WORLD_SIZE, cpu_group=CPU_PROCESS_GROUP)
        if RANK == 0 and metrics is not None and (wandb is not None):
            wandb.log(metrics, step=train_state.step)
        if RANK == 0 and config.checkpoint_every_eval:
            print('SAVE CHECKPOINT')
            save_train_state(config, train_state_eval)
        if config.ema:
            del train_state_eval
    if dist.is_initialized():
        dist.destroy_process_group()
    if wandb is not None:
        wandb.finish()
if __name__ == '__main__':
    launch()
