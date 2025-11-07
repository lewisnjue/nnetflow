# nnetflow Examples

This directory contains example scripts demonstrating how to use nnetflow for various machine learning tasks.

## Available Examples

### 1. Simple Regression (`simple_regression.py`)

Demonstrates linear regression using a single Linear layer. Shows how to:
- Create a simple model
- Use SGD optimizer
- Train on synthetic regression data
- Evaluate model performance

**Run:**
```bash
python examples/simple_regression.py
```

### 2. Binary Classification (`binary_classification.py`)

Demonstrates binary classification with a multi-layer neural network. Shows how to:
- Build a deeper network with multiple Linear layers
- Use ReLU activations
- Apply binary cross-entropy loss
- Use Adam optimizer
- Evaluate classification accuracy

**Run:**
```bash
python examples/binary_classification.py
```

### 3. Linear Layer Demo (`linear.py`)

Basic demonstration of Linear layer functionality with both regression and classification tasks.

**Run:**
```bash
python examples/linear.py
```

## Key Concepts

All examples demonstrate:

1. **Model Building**: Creating layers and connecting them
2. **Data Preparation**: Converting NumPy arrays to Tensors
3. **Training Loop**: Forward pass, loss computation, backward pass, parameter updates
4. **Optimization**: Using SGD or Adam optimizers
5. **Evaluation**: Computing metrics on test data

## Adding Your Own Examples

When creating new examples:

1. Import required components from nnetflow
2. Generate or load data
3. Define model architecture using Linear layers
4. Choose appropriate loss function
5. Create optimizer with model parameters
6. Implement training loop with evaluation
