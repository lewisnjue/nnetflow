from nnetflow.engine import Tensor 

def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor: 
    """Mean Squared Error Loss"""
    return ((predictions - targets) ** 2).mean() 

def rmse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Root Mean Squared Error Loss"""
    return (((predictions - targets) ** 2).mean()).sqrt() 

def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross Entropy Loss for multi-class classification
    Note: Uses log_softmax for numerical stability.
    """
    log_probs = logits.log_softmax(axis=-1)
    ce_loss = - (targets * log_probs).sum(axis=-1).mean()
    return ce_loss

def binary_cross_entropy_loss(predictions: Tensor, targets: Tensor, eps: float = 1e-7) -> Tensor:
    """Binary Cross Entropy Loss for binary classification
    Args:
        predictions (Tensor): Predicted probabilities (after sigmoid)
        targets (Tensor): Ground truth labels (0 or 1)
        eps (float): Epsilon value to prevent log(0)
    """
    predictions = predictions.clip(eps, 1.0 - eps)
    
    bce_loss = - (targets * predictions.log() + (1 - targets) * (1 - predictions).log()).mean()
    return bce_loss

def logits_binary_cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Binary Cross Entropy Loss taking raw logits as input.
    Recommended over binary_cross_entropy_loss for better stability.
    """
    probs = logits.sigmoid()
    
    return binary_cross_entropy_loss(probs, targets)