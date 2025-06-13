from .engine import Tensor
from .module import Module
from typing import List, Optional, Any



class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probs = logits.softmax(axis=-1) 
        loss = -targets * probs.log()
        if self.reduction == 'mean':
            return loss.mean().reshape(1)
        elif self.reduction == 'sum':
            return loss.sum().reshape(1)
        else:
            return loss.reshape(-1)

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}')"


class L1Loss(Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = (input - target).abs()
        if self.reduction == 'mean':
            return loss.mean().reshape(1)
        elif self.reduction == 'sum':
            return loss.sum().reshape(1)
        else:
            return loss.reshape(-1)

    def __repr__(self) -> str:
        return f"L1Loss(reduction='{self.reduction}')"


class MSELoss(Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = (input - target) ** 2  # do element wise subtraction and square 
        
        if self.reduction == 'mean':
            return loss.mean().reshape(1) 
        elif self.reduction == 'sum':
            return loss.sum().reshape(1)
        else:
            return loss.reshape(-1) 

    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"