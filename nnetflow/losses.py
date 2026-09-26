"""Loss functions used throughout the nnetflow training loop.

These utilities follow a very small, functional API: they accept a prediction
Tensor and target Tensor, and return a scalar loss Tensor whose backward pass
propagates gradients to trainable model parameters.
"""

from nnetflow.engine import Tensor 


class MSELoss:
    """Mean squared error loss.

    Computes the average squared error between predictions and targets.
    """
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ((predictions - targets) ** 2).mean()

class RMSELoss:
    """ Root Mean Squared Error Loss Class"""
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return (((predictions - targets) ** 2).mean()).sqrt()

class CrossEntropyLoss:
    """ Cross Entropy Loss Class"""
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return -(targets * predictions.log()).sum() / targets.shape[0]


