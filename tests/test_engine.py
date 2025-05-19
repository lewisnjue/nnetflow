import unittest
from nnetflow.engine import Tensor

class TestEngine(unittest.TestCase):
    def test_add(self):
        a = Tensor([1.0])
        b = Tensor([2.0])
        c = a + b
        self.assertTrue(c.data == 3.0 or c.data[0] == 3.0)

    def test_backward(self):
        a = Tensor([2.0])
        b = Tensor([3.0])
        c = a * b
        c.backward()
        self.assertTrue(a.grad == 3.0 or a.grad[0] == 3.0)
        self.assertTrue(b.grad == 2.0 or b.grad[0] == 2.0)

if __name__ == "__main__":
    unittest.main()
