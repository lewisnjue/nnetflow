import unittest
from nnetflow.nn import MLP
from nnetflow.engine import Tensor

class TestNN(unittest.TestCase):
    def test_mlp_forward(self):
        model = MLP(nin=2, nouts=[4, 2])
        x = [Tensor([1.0]), Tensor([2.0])]
        y = model(x)
        self.assertTrue(isinstance(y, list) and len(y) == 2)

if __name__ == "__main__":
    unittest.main()
