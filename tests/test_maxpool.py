import pytest
import numpy as np
import torch
import torch.nn as nn
from nnetflow import Tensor
from nnetflow.layers import MaxPool1d, MaxPool2d 

maxpool2d_configs = [
    # (B, C, H, W, K, S, P)
    (1, 1, 6, 6, 2, 2, 0),        # Simple 2x2, non-overlapping
    (2, 3, 7, 7, 3, 2, 0),        # Batched, multi-channel, overlapping stride
    (1, 2, 8, 8, 3, None, 1),     # Padding, stride=None (defaults to kernel_size)
    (1, 1, 7, 5, (2, 3), (1, 2), 0), # Rectangular kernel and stride
    (1, 1, 5, 5, 3, 1, 0),        # Heavy overlap (stride=1)
]

# (B, C, L, K, S, P)
maxpool1d_configs = [
    # (B, C, L, K, S, P)
    (1, 1, 10, 2, 2, 0),       # Simple 2, non-overlapping
    (2, 3, 12, 3, 2, 0),       # Batched, multi-channel, overlapping stride
    (1, 2, 10, 3, None, 1),    # Padding, stride=None (defaults to kernel_size)
    (1, 1, 10, 3, 1, 0),       # Heavy overlap (stride=1)
]


# Set tolerances for floating point comparison
RTOL = 1e-5
ATOL = 1e-6

# -------------------------------------------------------------------
# --- PYTEST FUNCTIONS ---
# -------------------------------------------------------------------

@pytest.mark.parametrize("B, C, H, W, K_size, S_size, P", maxpool2d_configs)
def test_maxpool2d_forward_backward(B, C, H, W, K_size, S_size, P):
    """
    Tests the MaxPool2d layer's forward and backward pass against PyTorch.
    """
    # Set seed for reproducible random numbers
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Create identical data
    # Use .astype(np.float64) to match your Tensor class's default float64
    np_x = np.random.randn(B, C, H, W).astype(np.float64)

    # 2. Initialize custom layer
    # Your layer's __init__ handles int/tuple/None for K and S
    my_pool = MaxPool2d(kernel_size=K_size, stride=S_size, padding=P)
    my_x = Tensor(np_x, requires_grad=True)

    # 3. Initialize torch layer
    # We MUST use return_indices=True to ensure the backward pass
    # is comparable (i.e., based on argmax).
    torch_pool = nn.MaxPool2d(
        kernel_size=K_size, 
        stride=S_size, 
        padding=P, 
        return_indices=True
    )
    torch_x = torch.tensor(np_x, requires_grad=True, dtype=torch.float64)

    # --- 4. Test Forward Pass ---
    my_out = my_pool(my_x)
    
    # PyTorch returns (output, indices)
    torch_out, torch_indices = torch_pool(torch_x)

    np.testing.assert_allclose(
        my_out.data, 
        torch_out.detach().numpy(), 
        rtol=RTOL, 
        atol=ATOL,
        err_msg="MaxPool2d FORWARD pass mismatch"
    )

    # --- 5. Test Backward Pass ---
    # We use .sum() as a simple loss function
    my_loss = my_out.sum()
    torch_loss = torch_out.sum()
    
    my_loss.backward()
    torch_loss.backward()

    # Compare gradients (only for input, no learnable params)
    np.testing.assert_allclose(
        my_x.grad, 
        torch_x.grad.detach().numpy(), 
        rtol=RTOL, 
        atol=ATOL,
        err_msg="MaxPool2d INPUT gradient mismatch"
    )


@pytest.mark.parametrize("B, C, L, K, S, P", maxpool1d_configs)
def test_maxpool1d_forward_backward(B, C, L, K, S, P):
    """
    Tests the MaxPool1d layer's forward and backward pass against PyTorch.
    """
    # Set seed for reproducible random numbers
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Create identical data
    np_x = np.random.randn(B, C, L).astype(np.float64)

    # 2. Initialize custom layer
    my_pool = MaxPool1d(kernel_size=K, stride=S, padding=P)
    my_x = Tensor(np_x, requires_grad=True)

    # 3. Initialize torch layer
    torch_pool = nn.MaxPool1d(
        kernel_size=K, 
        stride=S, 
        padding=P, 
        return_indices=True
    )
    torch_x = torch.tensor(np_x, requires_grad=True, dtype=torch.float64)

    # --- 4. Test Forward Pass ---
    my_out = my_pool(my_x)
    torch_out, torch_indices = torch_pool(torch_x)

    np.testing.assert_allclose(
        my_out.data, 
        torch_out.detach().numpy(), 
        rtol=RTOL, 
        atol=ATOL,
        err_msg="MaxPool1d FORWARD pass mismatch"
    )

    # --- 5. Test Backward Pass ---
    my_loss = my_out.sum()
    torch_loss = torch_out.sum()
    
    my_loss.backward()
    torch_loss.backward()

    # Compare gradients
    np.testing.assert_allclose(
        my_x.grad, 
        torch_x.grad.detach().numpy(), 
        rtol=RTOL, 
        atol=ATOL,
        err_msg="MaxPool1d INPUT gradient mismatch"
    )