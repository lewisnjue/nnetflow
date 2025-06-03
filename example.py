from nnetflow.optim import Adam 
from nnetflow import engine 
import numpy as np
import pandas as pd
from pathlib import Path
from nnetflow import nn 



class SelfAttention(nn.Module):
    def __init__(self,embed_size,heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads 
        self.head_dim = embed_size // heads 

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"
        self.values = nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.keys = nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.queries = nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0] 
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embeddings int heads 
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)


        # computer attentin scores 
        energy = queries @ keys.permute(0, 1, 3, 2)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = energy.softmax(dim=3)
        out = attention @ values # shape (N, query_len, self.heads, self.head_dim)
        out = out.reshape(N, query_len, self.heads * self.head_dim) # shape (N, query_len, embed_size)
        out = self.fc_out(out) # shape (N, query_len, embed_size)
        return out


class TransformerBlock(nn.Module):
    def __init__(self,embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ffn = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),

            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.ffn(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class GPT2(nn.Module):
    def __init__(self, 
                 vocab_size,
                 embed_size=256,
                 num_layers=6,
                 heads=8,
                 dropout=0.1,
                 forward_expansion=4,
                 max_length=512):
        super().__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.max_length = max_length

    def forward(self, x, mask):
        N, seq_length = x.shape

        positions = np.tile(np.arange(0, seq_length), (N, 1))

        token_embeds = self.token_embedding(x)  # shape (N, seq_length, embed_size)
        position_embeds = self.position_embedding(positions)  # shape (N, seq_length, embed_size)
        x = self.dropout(token_embeds + position_embeds)  # shape (N, seq_length, embed_size)

        for layer in self.layers:
            x = layer(x, x, x, mask) # self attention, key, query, value are all the same
        out = self.fc_out(x)  # shape (N, seq_length, vocab_size)
        return out




# example of training this model code 



data_path = Path("data.txt")

with open(data_path, "r") as f:
    data = f.read()


class TextDataset:
    def __init__(self, text, seq_length=64, pad_token='[PAD]'):
        self.text = text
        self.seq_length = seq_length
        self.vocab = sorted(set(text) | set([pad_token]))
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.pad_token = pad_token
        self.pad_idx = self.char_to_idx[pad_token]
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return np.array(x, dtype=np.int64), np.int64(y)


if __name__ == "__main__":
    import tqdm
    import random
    
    # Hyperparameters
    seq_length = 64
    batch_size = 32
    num_epochs = 2  # Reduced to 2
    lr = 3e-4
    model_save_path = "gpt2_char_model.nnetflow.pt"

    # Prepare dataset
    dataset = TextDataset(data, seq_length=seq_length)
    vocab = dataset.vocab
    vocab_size = dataset.vocab_size
    char_to_idx = dataset.char_to_idx
    idx_to_char = dataset.idx_to_char
    pad_idx = dataset.pad_idx

    # Limit dataset size to avoid OOM
    max_train_samples = 1000  # Reduce further for safety
    # Only sample a random subset of indices for each epoch
    def get_batches(dataset, batch_size, max_samples=None):
        indices = list(range(len(dataset)))
        if max_samples is not None and len(indices) > max_samples:
            indices = random.sample(indices, max_samples)
        random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = []
            batch_y = []
            for idx in batch_idx:
                x, y = dataset[idx]
                batch_x.append(x)  # x is already a numpy array
                batch_y.append(y)
            yield np.stack(batch_x), np.array(batch_y)

    # Model
    model = GPT2(vocab_size=vocab_size, max_length=seq_length, embed_size=128, num_layers=2, heads=2, dropout=0.1, forward_expansion=2)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, num_epochs + 1):
        nn.train()
        total_loss = 0
        num_samples = 0
        for i, (x, y) in enumerate(get_batches(dataset, batch_size, max_samples=max_train_samples)):
            logits = model(x, mask=None)  # (batch, seq_length, vocab_size)
            logits = logits[:, -1, :]  # (batch, vocab_size)
            y_tensor = engine.Tensor(y, shape=y.shape)
            loss = criterion(logits, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Use float() to extract scalar from loss.data if .item() is not available
            total_loss += float(loss.data) * x.shape[0]
            num_samples += x.shape[0]
            if (i+1) % 5 == 0:
                print(f"Batch {i+1} - Loss: {float(loss.data):.4f}")
        avg_loss = total_loss / num_samples
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

        # Evaluation: generate text
        nn.eval()
        # Use the underlying data for text generation
        def get_base_data(ds):
            if hasattr(ds, 'data'):
                return ds.data
            elif hasattr(ds, 'base'):
                return get_base_data(ds.base)
            else:
                raise AttributeError('No data attribute found for dataset')
        base_data = get_base_data(dataset)
        start_idx = random.randint(0, len(base_data) - seq_length)
        input_seq = base_data[start_idx:start_idx+seq_length]
        generated = input_seq[:]
        for _ in range(100):  # Generate 100 characters
            x_gen = np.array(generated[-seq_length:])[None, :]  # (1, seq_length)
            logits = model(x_gen, mask=None)  # (1, seq_length, vocab_size)
            logits = logits[:, -1, :]
            # Use the softmax function defined in nnetflow.nn
            probs = nn.__dict__["softmax"](logits, dim=1).data[0]
            next_idx = np.random.choice(len(probs), p=probs/probs.sum())
            generated.append(next_idx)
        gen_text = ''.join([idx_to_char[i] for i in generated])
        print(f"Sample generated text after epoch {epoch}:\n{gen_text}\n")