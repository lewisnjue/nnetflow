import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nnetflow.engine import Tensor
from nnetflow.nn import MLP, CrossEntropyLoss, SGD

# Create a challenging two-moons classification dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameters
input_dim = X_train.shape[1]
hidden_layers = [16, 16]
output_dim = 2  # binary classification
learning_rate = 0.1
epochs = 100

# Initialize model, loss, optimizer
model = MLP(nin=input_dim, nouts=hidden_layers + [output_dim])
criterion = CrossEntropyLoss()
optimizer = SGD(params=model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    batch_losses = []

    # Forward pass: compute logits for each sample
    logits = []
    for xi in X_train:
        # convert feature vector to list of Tensors
        t_in = [Tensor([float(val)]) for val in xi]
        out = model(t_in)  # returns list of Tensors (len=output_dim)
        logits.append(out)

    # Compute loss
    loss = criterion(logits, y_train.tolist())
    batch_losses.append(loss.data.item())

    # Backward pass
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == 1:
        avg_loss = np.mean(batch_losses)
        # Evaluate accuracy on test set
        correct = 0
        total = len(X_test)
        for xi, yi in zip(X_test, y_test):
            t_in = [Tensor([float(val)]) for val in xi]
            out = model(t_in)
            # pick class with highest logit
            pred = int(np.argmax([t.data.item() for t in out]))
            if pred == int(yi):
                correct += 1
        acc = correct / total * 100
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")

print("Training complete.")
