import pytest
import numpy as np
import torch
import torch.nn as nn

from nnetflow import Tensor
from nnetflow.layers import Conv1d, Conv2d 

conv2d_configs = [
    # (B, C_in, C_out, K, S, P, bias, H, W)
    (1, 1, 1, 3, 1, 0, True, 5, 5),   # Simple case
    (2, 3, 4, 3, 1, 0, True, 8, 8),   # More channels
    (1, 1, 1, 3, 1, 0, False, 5, 5),  # No bias
    (1, 2, 2, 3, 2, 0, True, 7, 7),   # Stride=2
    (1, 1, 2, 3, 1, 1, True, 5, 5),   # Padding=1
    (2, 2, 3, 3, 2, 1, True, 8, 8),   # Stride=2, Padding=1
    (1, 3, 2, 1, 1, 0, True, 5, 5),   # Kernel=1
]

# (B, C_in, C_out, K, S, P, bias, L)
conv1d_configs = [
    # (B, C_in, C_out, K, S, P, bias, L)
    (1, 1, 1, 3, 1, 0, True, 10),  # Simple case
    (2, 3, 4, 3, 1, 0, True, 16),  # More channels
    (1, 1, 1, 3, 1, 0, False, 10), # No bias
    (1, 2, 2, 3, 2, 0, True, 14),  # Stride=2
    (1, 1, 2, 3, 1, 1, True, 10),  # Padding=1
    (2, 2, 3, 3, 2, 1, True, 16),  # Stride=2, Padding=1
    (1, 3, 2, 1, 1, 0, True, 10),  # Kernel=1
]

# Set tolerances for floating point comparison
# Your Tensor class defaults to float64, so we use high precision
RTOL = 1e-5
ATOL = 1e-6


@pytest.mark.parametrize("B, C_in, C_out, K, S, P, bias, H, W", conv2d_configs)
def test_conv2d_forward_backward(B, C_in, C_out, K, S, P, bias, H, W):
    """
    Tests the Conv2d layer's forward and backward pass against PyTorch.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    np_x = np.random.randn(B, C_in, H, W).astype(np.float64)
    np_w = np.random.randn(C_out, C_in, K, K).astype(np.float64)
    if bias:
        np_b = np.random.randn(C_out).astype(np.float64)

    # 2. Initialize custom layer
    my_conv = Conv2d(C_in, C_out, K, S, P, bias)
    my_conv.weight = Tensor(np_w, requires_grad=True)
    if bias:
        my_conv.bias = Tensor(np_b.reshape(1, C_out), requires_grad=True)
    my_x = Tensor(np_x, requires_grad=True)

    # 3. Initialize torch layer
    torch_conv = nn.Conv2d(C_in, C_out, K, S, P, bias=bias)
    torch_conv.weight = nn.Parameter(torch.tensor(np_w, dtype=torch.float64))
    if bias:
        # PyTorch bias is (C_out)
        torch_conv.bias = nn.Parameter(torch.tensor(np_b, dtype=torch.float64))
    torch_x = torch.tensor(np_x, requires_grad=True, dtype=torch.float64)

    # --- 4. Test Forward Pass ---
    my_out = my_conv(my_x)
    torch_out = torch_conv(torch_x)

    np.testing.assert_allclose(
        my_out.data, 
        torch_out.detach().numpy(), 
        rtol=RTOL, 
        atol=ATOL,
        err_msg="Conv2d FORWARD pass mismatch"
    )

    my_loss = my_out.sum()
    torch_loss = torch_out.sum()
    
    my_loss.backward()
    torch_loss.backward()

    np.testing.assert_allclose(
        my_x.grad, 
        torch_x.grad.detach().numpy(), 
        rtol=RTOL, 
        atol=ATOL,
        err_msg="Conv2d INPUT gradient mismatch"
    )
    
    np.testing.assert_allclose(
        my_conv.weight.grad, 
        torch_conv.weight.grad.detach().numpy(), 
        rtol=RTOL, 
        atol=ATOL,
        err_msg="Conv2d WEIGHT gradient mismatch"
    )
    
    if bias:
        np.testing.assert_allclose(
            my_conv.bias.grad, 
            torch_conv.bias.grad.detach().numpy().reshape(my_conv.bias.shape), 
            rtol=RTOL, 
            atol=ATOL,
            err_msg="Conv2d BIAS gradient mismatch"
        )


@pytest.mark.parametrize("B, C_in, C_out, K, S, P, bias, L", conv1d_configs)
def test_conv1d_forward_backward(B, C_in, C_out, K, S, P, bias, L):
    """
    Tests the Conv1d layer's forward and backward pass against PyTorch.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    np_x = np.random.randn(B, C_in, L).astype(np.float64)
    np_w = np.random.randn(C_out, C_in, K).astype(np.float64)
    if bias:
        np_b = np.random.randn(C_out).astype(np.float64)

    my_conv = Conv1d(C_in, C_out, K, S, P, bias)
    my_conv.weight = Tensor(np_w, requires_grad=True)
    if bias:
        my_conv.bias = Tensor(np_b.reshape(1, C_out), requires_grad=True)
    my_x = Tensor(np_x, requires_grad=True)

    torch_conv = nn.Conv1d(C_in, C_out, K, S, P, bias=bias)
    torch_conv.weight = nn.Parameter(torch.tensor(np_w, dtype=torch.float64))
    if bias:
        torch_conv.bias = nn.Parameter(torch.tensor(np_b, dtype=torch.float64))
    torch_x = torch.tensor(np_x, requires_grad=True, dtype=torch.float64)

    my_out = my_conv(my_x)
    torch_out = torch_conv(torch_x)

    np.testing.assert_allclose(
        my_out.data, 
        torch_out.detach().numpy(), 
        rtol=RTOL, 
        atol=ATOL,
        err_msg="Conv1d FORWARD pass mismatch"
    )

    my_loss = my_out.sum()
    torch_loss = torch_out.sum()
    
    my_loss.backward()
    torch_loss.backward()

    np.testing.assert_allclose(
        my_x.grad, 
        torch_x.grad.detach().numpy(), 
        rtol=RTOL, 
        atol=ATOL,
        err_msg="Conv1d INPUT gradient mismatch"
    )
    
    np.testing.assert_allclose(
        my_conv.weight.grad, 
        torch_conv.weight.grad.detach().numpy(), 
        rtol=RTOL, 
        atol=ATOL,
        err_msg="Conv1d WEIGHT gradient mismatch"
    )
    
    if bias:
        np.testing.assert_allclose(
            my_conv.bias.grad, 
            torch_conv.bias.grad.detach().numpy().reshape(my_conv.bias.shape), 
            rtol=RTOL, 
            atol=ATOL,
            err_msg="Conv1d BIAS gradient mismatch"
        )