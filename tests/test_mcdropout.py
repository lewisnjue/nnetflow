import numpy as np

from nnetflow.engine import Tensor
from nnetflow.layers import MCDropout


def test_mcdropout_returns_a_tensor_in_training_and_eval():
    np.random.seed(11)
    layer = MCDropout(0.5)
    x = Tensor(np.ones((30, 30)), requires_grad=True)
    training_out = layer(x)
    layer.eval()
    eval_out = layer(x)
    assert isinstance(training_out, Tensor)
    assert isinstance(eval_out, Tensor)
    assert np.any(training_out.data == 0.0)
    assert np.any(eval_out.data == 0.0)