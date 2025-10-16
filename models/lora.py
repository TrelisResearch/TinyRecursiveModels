from typing import Optional

from torch import nn

from models.layers import CastedLinear


def enable_lora_for_linears(
    module: nn.Module,
    rank: int,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    train_base: bool = False,
    train_bias: bool = False,
) -> None:
    """
    Enable LoRA adapters for every CastedLinear submodule within `module`.

    Args:
        module: Root module to traverse.
        rank: LoRA rank. <=0 disables LoRA.
        alpha: Scaling factor (defaults to rank when None).
        dropout: Optional dropout applied to the LoRA branch.
        train_base: If False, freeze original linear weights.
        train_bias: If False, freeze original biases.
    """
    if rank <= 0:
        return

    for submodule in module.modules():
        if isinstance(submodule, CastedLinear):
            submodule.enable_lora(
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                train_base=train_base,
                train_bias=train_bias,
            )
