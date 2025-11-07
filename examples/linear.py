import numpy as np 
from nnetflow.layers import Linear
from nnetflow.engine import Tensor

def linear_regression():
    # create a simple dataset for regression 
    np.random.seed(0)
    X = np.random.rand(100, 3) 
    true_weights = np.array([[2.0], [-3.0], [1.5]])
    true_bias = np.array([[0.5]])
    y = X @ true_weights + true_bias + 0.1 * np.random.randn(100, 1)  # add some noise 


    linear1 = Linear(in_features=3, out_features=10) 
    linear1.weight.relu() 
    linear2 = Linear(in_features=10, out_features=40) 
    linear2.weight.relu() 
    linear3 = Linear(in_features=40, out_features=1)

    


    # forward pass 
    inputs = Tensor(X, requires_grad=False)
    targets = Tensor(y, requires_grad=False)
    epochs = 100
    for epoch in range(epochs): 
        outputs = linear3(linear2(linear1(inputs))) 
        loss = ((outputs - targets) ** 2).mean()
        loss.backward()
        for layer in [linear1, linear2, linear3]:
            layer.weight.data -= 0.01 * layer.weight.grad
            layer.bias.data -= 0.01 * layer.bias.grad
            layer.weight.zero_grad()
            layer.bias.zero_grad()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        



def classification_task():
    # create a simple dataset for classification 
    np.random.seed(0)
    X = np.random.rand(100, 2) 
    y = ((X[:,0] + X[:,1]) > 1).astype(np.float32).reshape(-1,1)  # simple linearly separable data 

    linear = Linear(in_features=2, out_features=1) 

    # forward pass 
    inputs = Tensor(X, requires_grad=False)
    targets = Tensor(y, requires_grad=False)
    epochs = 100000
    for epoch in range(epochs): 
        outputs = linear(inputs).sigmoid() 
        loss = - (targets * outputs.log() + (1 - targets) * (1 - outputs).log()).mean()  # binary cross-entropy
        loss.backward()
        linear.weight.data -= 0.1 * linear.weight.grad
        linear.bias.data -= 0.1 * linear.bias.grad
        linear.weight.zero_grad()
        linear.bias.zero_grad()
        if (epoch+1) % 10000 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")





linear_regression() 
print("\n \n \n ========================================")

classification_task()