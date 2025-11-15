"""Tests that GELU implementations run successfully (SciPy present).

These tests don't directly assert packaging metadata, but they
exercise the code paths that depend on SciPy so that missing
SciPy would show up as an ImportError at test time.
"""

import numpy as np

from nnetflow.engine import Tensor
from nnetflow.experimental import activations as exp_act


def test_engine_gelu_runs():
    x = Tensor(np.array([0.0, 1.0, -1.0]))
    y = x.gelu()

    assert y.shape == x.shape


def test_experimental_gelu_runs():
    x = np.array([0.0, 1.0, -1.0])
    y = exp_act.GELU(x)

    assert y.shape == x.shape
