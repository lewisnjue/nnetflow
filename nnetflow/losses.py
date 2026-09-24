from nnetflow.engine import Tensor 


class MSELoss:
    """ Mean Squared Error Loss Class"""
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


