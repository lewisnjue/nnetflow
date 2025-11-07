from nnetflow.engine import Tensor 
import numpy as np 


# loss functions 


def mse_loss(predictions:Tensor,targets:Tensor) -> Tensor: 
    """Mean Squared Error Loss 
    Args:
        predictions (Tensor): Predicted values 
        targets (Tensor): Ground truth values 
    Returns:
        Tensor: Computed MSE loss 
    """
    return ((predictions - targets) ** 2).mean() 


def rmse_loss(predictions:Tensor, targets:Tensor) -> Tensor:
    """Root Mean Squared Error Loss
    Args:
        predictions (Tensor): Predicted values
        targets (Tensor): Ground truth values
    Returns:
        Tensor: Computed RMSE loss
    """
    return (((predictions - targets) ** 2).mean()).sqrt() 


def cross_entropy_loss(logits:Tensor, targets:Tensor) -> Tensor: 
    """Cross Entropy Loss for multi-class classification
    Args:
        logits (Tensor): Predicted logits (before softmax) 
        targets (Tensor): One-hot encoded ground truth labels 
    Returns:
        Tensor: Computed cross-entropy loss 
    """ 
    probability = logits.softmax(dim=-1)
    ce_loss = - (targets * probability.log()).sum(dim=-1).mean()
    return ce_loss

def binary_cross_entropy_loss(predictions:Tensor, targets:Tensor) -> Tensor:
    """Binary Cross Entropy Loss for binary classification
    Args:
        predictions (Tensor): Predicted probabilities (after sigmoid)
        targets (Tensor): Ground truth labels (0 or 1)
    Returns:
        Tensor: Computed binary cross-entropy loss
    """
    bce_loss = - (targets * predictions.log() + (1 - targets) * (1 - predictions).log()).mean()
    return bce_loss

def logits_binary_cross_entropy_loss(logits:Tensor, targets:Tensor) -> Tensor:
    """Binary Cross Entropy Loss with logits for binary classification
    Args:
        logits (Tensor): Predicted logits (before sigmoid)
        targets (Tensor): Ground truth labels (0 or 1)
    Returns:
        Tensor: Computed binary cross-entropy loss
    """
    probs = logits.sigmoid()
    bce_loss = - (targets * probs.log() + (1 - targets) * (1 - probs).log()).mean()
    return bce_loss

