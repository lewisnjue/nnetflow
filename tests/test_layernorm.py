import numpy as np
import torch

from nnetflow.engine import Tensor
from nnetflow.layers import LayerNorm


def test_layernorm_forward_and_backward_match_torch():
    np.random.seed(5)
    data = np.random.randn(2, 4, 3)
    layer = LayerNorm(3)
    x = Tensor(data.copy(), requires_grad=True)
    out = layer(x)
    out.sum().backward()

    reference = torch.nn.LayerNorm(3, eps=1e-5, dtype=torch.float64)
    torch_x = torch.tensor(data, requires_grad=True)
    torch_out = reference(torch_x)
    torch_out.sum().backward()

    assert out.shape == data.shape
    assert np.allclose(out.data, torch_out.detach().numpy(), atol=1e-6)
    assert np.allclose(x.grad, torch_x.grad.numpy(), atol=1e-5)
    assert np.allclose(layer.gamma.grad, reference.weight.grad.numpy().reshape(1, 3), atol=1e-5)
    assert np.allclose(layer.beta.grad, reference.bias.grad.numpy().reshape(1, 3), atol=1e-5)


def test_layernorm_normalizes_each_last_dimension():
    layer = LayerNorm(4)
    out = layer(Tensor(np.arange(8, dtype=np.float64).reshape(2, 4)))
    assert np.allclose(out.data.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(out.data.var(axis=-1), 1.0, atol=1e-4)