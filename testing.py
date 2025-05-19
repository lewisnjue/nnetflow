import numpy 

from chainflow.nn import MLP, SGD, MSELoss
from chainflow.engine import Tensor 

data = [
    ([Tensor([1.0]), Tensor([2.0])], [Tensor([2.0]), Tensor([4.0])]),
    ([Tensor([3.0]), Tensor([1.0])], [Tensor([6.0]), Tensor([2.0])]),
]

model = MLP(nin=2, nouts=[4, 2])
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.05)

for epoch in range(100):
    total_loss = 0.0

    for x, y_true in data:
        y_pred = model(x)
        if not isinstance(y_pred, list):
            y_pred = [y_pred]
        
        loss = loss_fn(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")




x_test = [Tensor([2.0]), Tensor([3.0])]
y_pred = model(x_test)
print("Prediction:", [p.data for p in y_pred])
