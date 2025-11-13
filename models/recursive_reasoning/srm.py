from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
    SwiGLU,
    rms_norm,
)
from models.sparse_embedding import CastedSparseEmbedding


class SimpleRecursiveReasoningModelConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int

    hidden_size: int
    num_heads: int
    expansion: float
    pos_encodings: str = "rope"
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    num_layers: int = 4
    cycles: int = 4
    cycles_start: Optional[int] = None
    cycles_end: Optional[int] = None

    puzzle_emb_ndim: int = 0
    shared_puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int = 0
    num_base_puzzle_identifiers: Optional[int] = None
    puzzle_emb_dropout: float = 0.0
    grid_token_dropout: float = 0.0
    puzzle_emb_len: int = 0

    forward_dtype: str = "bfloat16"


class SRMBlock(nn.Module):
    def __init__(self, config: SimpleRecursiveReasoningModelConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor, cos_sin: Optional[CosSin]) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


class SRMReasoning(nn.Module):
    def __init__(self, config: SimpleRecursiveReasoningModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SRMBlock(config) for _ in range(config.num_layers)])

    def forward(self, hidden_states: torch.Tensor, input_tokens: torch.Tensor, cos_sin: Optional[CosSin]) -> torch.Tensor:
        hidden_states = hidden_states + input_tokens
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin)
        return hidden_states


class SimpleRecursiveReasoningModel(nn.Module):
    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        self.config = SimpleRecursiveReasoningModelConfig(**config_dict)
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = self.config.hidden_size ** 0.5
        embed_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

        variant_tokens = 0
        shared_tokens = 0

        if self.config.puzzle_emb_ndim > 0:
            variant_tokens = self.config.puzzle_emb_len or -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            variant_tokens = max(1, variant_tokens)
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )
        else:
            self.puzzle_emb = None

        if self.config.shared_puzzle_emb_ndim > 0 and self.config.num_base_puzzle_identifiers is not None:
            shared_tokens = -(self.config.shared_puzzle_emb_ndim // -self.config.hidden_size)
            shared_tokens = max(1, shared_tokens)
            self.shared_puzzle_emb = CastedSparseEmbedding(
                self.config.num_base_puzzle_identifiers,
                self.config.shared_puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )
        else:
            self.shared_puzzle_emb = None

        self.variant_puzzle_emb_len = variant_tokens
        self.shared_puzzle_emb_len = shared_tokens
        self.puzzle_emb_len = variant_tokens + shared_tokens

        self.min_cycles = max(1, self.config.cycles_start or self.config.cycles)
        self.max_cycles = max(self.min_cycles, self.config.cycles_end or self.config.cycles)
        self.current_cycles = max(1, self.min_cycles)

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_std,
                cast_to=self.forward_dtype,
            )
        else:
            self.rotary_emb = None

        self.reasoning = SRMReasoning(self.config)

        self.hidden_template = nn.Buffer(
            trunc_normal_init_(
                torch.empty(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
                std=1.0,
            ),
            persistent=True,
        )

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        return None

    def _dropout_puzzle(self, emb: torch.Tensor) -> torch.Tensor:
        if not self.training or self.config.puzzle_emb_dropout <= 0:
            return emb
        keep_prob = 1.0 - self.config.puzzle_emb_dropout
        keep_mask = (torch.rand(emb.size(0), device=emb.device) >= self.config.puzzle_emb_dropout).to(emb.dtype)
        emb = emb * keep_mask.unsqueeze(-1)
        return emb * (1.0 / keep_prob)

    def _input_embeddings(
        self,
        inputs: torch.Tensor,
        puzzle_identifiers: torch.Tensor,
        base_puzzle_identifiers: Optional[torch.Tensor],
    ) -> torch.Tensor:
        embedding = self.embed_tokens(inputs.to(torch.int32))
        if self.training and self.config.grid_token_dropout > 0:
            embedding = F.dropout(embedding, p=self.config.grid_token_dropout, training=True)

        prefix_tokens = []

        if self.shared_puzzle_emb is not None:
            if base_puzzle_identifiers is None:
                raise ValueError("Shared puzzle embeddings requested but base_puzzle_identifiers missing from batch.")
            shared_emb = self._dropout_puzzle(self.shared_puzzle_emb(base_puzzle_identifiers))
            pad_units = self.shared_puzzle_emb_len * self.config.hidden_size - shared_emb.shape[-1]
            if pad_units > 0:
                shared_emb = F.pad(shared_emb, (0, pad_units))
            prefix_tokens.append(shared_emb.view(-1, self.shared_puzzle_emb_len, self.config.hidden_size))

        if self.puzzle_emb is not None:
            variant_emb = self._dropout_puzzle(self.puzzle_emb(puzzle_identifiers))
            pad_units = self.variant_puzzle_emb_len * self.config.hidden_size - variant_emb.shape[-1]
            if pad_units > 0:
                variant_emb = F.pad(variant_emb, (0, pad_units))
            prefix_tokens.append(variant_emb.view(-1, self.variant_puzzle_emb_len, self.config.hidden_size))

        if prefix_tokens:
            embedding = torch.cat((*prefix_tokens, embedding), dim=-2)

        if hasattr(self, "embed_pos"):
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def forward(self, carry: Optional[torch.Tensor], batch: Dict[str, torch.Tensor]):
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        inputs = self._input_embeddings(
            batch["inputs"],
            batch["puzzle_identifiers"],
            batch.get("base_puzzle_identifiers"),
        )

        hidden = inputs
        total_cycles = max(1, getattr(self, "current_cycles", self.config.cycles))
        for _ in range(total_cycles):
            hidden = self.reasoning(hidden, inputs, cos_sin)

        logits = self.lm_head(hidden)[:, self.puzzle_emb_len :]
        outputs = {"logits": logits}
        return None, outputs

    def set_cycle_progress(self, progress: float):
        progress = float(max(0.0, min(1.0, progress)))
        if self.max_cycles == self.min_cycles:
            self.current_cycles = max(1, self.max_cycles)
            return
        span = self.max_cycles - self.min_cycles
        target = self.min_cycles + span * progress
        self.current_cycles = max(1, int(round(target)))
