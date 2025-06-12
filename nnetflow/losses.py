from .engine import Tensor
from .module import Module
from typing import List, Optional, Any



class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        assert logits.shape[-1] == targets.shape[-1], "Logits and targets must have the same last dimension."
        probs = logits.softmax(axis=-1)
        loss = -targets * probs.log()
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss 

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}')"


class L1Loss(Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape, "Input and target must have the same shape."
        loss = (input - target).abs()
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def __repr__(self) -> str:
        return f"L1Loss(reduction='{self.reduction}')"


class MSELoss(Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape, "Input and target must have the same shape."
        _inter = input - target
        assert isinstance(_inter, Tensor), "tracking the error foud"
        loss = _inter * _inter 
        assert isinstance(loss, Tensor), "tracking the error found 2 "
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"