# %%
import os
from pathlib import Path
# %%
import nnetflow as nf
from nnetflow import Tensor
from nnetflow.layers import Linear
from nnetflow.optim import Adam
import nnetflow.module as module

# %%
# model

class Model(module.Module):
    def __init__(self,in_features:int,out_features:int,hidden:int = 128):
        super().__init__()
        self.linear1 = Linear(in_features, hidden, dtype=np.float32)
        self.linear2 = Linear(hidden, out_features, dtype=np.float32)
    
    def forward(self,x:Tensor)->Tensor:
        x = self.linear1(x)
        x = x.gelu()
        x = self.linear2(x)
        return x


# %%
# smaple data 
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_regression(n_samples=100000, n_features=10, noise=0.1)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

train_X = Tensor(train_X.astype(np.float32), requires_grad=False)
train_y = Tensor(train_y.astype(np.float32).reshape(-1, 1), requires_grad=False)
test_X = Tensor(test_X.astype(np.float32), requires_grad=False)
test_y = Tensor(test_y.astype(np.float32).reshape(-1, 1), requires_grad=False)

# %%
train_X.dtype

# %%
import matplotlib.pyplot as plt

# plot a few samples of the data
plt.scatter(train_X.data[1:100, 0], train_y.data[1:100, 0])

# %%
class Dataset:
    def __init__(self,X:Tensor,y:Tensor):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
    def __len__(self):
        return self.n_samples
    def __getitem__(self,idx:int):
        return self.X[idx],self.y[idx]

class DataLoader:
    def __init__(self,dataset:Dataset,batch_size:int=32,shuffle:bool=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(dataset)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
    
    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            self.indices = np.random.permutation(self.n_samples)
        else:
            self.indices = np.arange(self.n_samples)
        return self
    
    def __len__(self):
        return self.n_batches
    
    def __next__(self):
        if self.idx >= self.n_samples:
            raise StopIteration
        batch_indices = self.indices[self.idx:self.idx+self.batch_size]
        batch_X, batch_y = [], []
        for i in batch_indices:
            x, y = self.dataset[i]
            batch_X.append(x.data)
            batch_y.append(y.data)
        batch_X = Tensor(np.stack(batch_X), requires_grad=False)
        batch_y = Tensor(np.stack(batch_y), requires_grad=False)
        self.idx += self.batch_size
        return batch_X, batch_y

# %%
dataset = Dataset(train_X, train_y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
# print first batch
for batch_X, batch_y in dataloader:
    print(batch_X.shape, batch_y.shape)
    print(type(batch_X), type(batch_y))
    break

# %%
def train(model:module.Module, dataloader:DataLoader, optimizer:Adam, criterion, epochs:int=10,eval_step:int=1,eval_dataloader:DataLoader=None):
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.data:.4f}")
        if eval_dataloader is not None and (epoch + 1) % eval_step == 0:
            eval_loss = 0.0
            for batch_X, batch_y in eval_dataloader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                eval_loss += loss.data
            eval_loss /= len(eval_dataloader)
            print(f"Evaluation Loss: {eval_loss:.4f}")


# %%
test_X[0].shape

# %%
eval_dataset = Dataset(test_X, test_y)
eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

model = Model(in_features=10, out_features=1, hidden=128)

# %%

optimizer = Adam(model.parameters(), lr=0.001)
class MSELoss:
    def __call__(self, outputs:Tensor, targets:Tensor):
        return ((outputs - targets) ** 2).mean()

criterion = MSELoss()

train(model, dataloader, optimizer, criterion, epochs=10, eval_step=1, eval_dataloader=eval_dataloader)

# %%
model.parameters()

# %%


# %%
print(model.parameters())



