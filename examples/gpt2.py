import os
import numpy as np
import time
import gc
from nnetflow.engine import Tensor
from nnetflow import layers
from nnetflow import losses as nf_losses
from nnetflow import optim as nf_optim
import tiktoken
import kagglehub

GPT_CONFIG_TINY = {
    "vocab_size": 50257,
    "context_length": 4,  
    "emb_dim": 4,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class FeedForward:
    def __init__(self, cfg):
        self.layers = [
            layers.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            layers.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        ]

    def __call__(self, x):
        return self.layers[1](self.layers[0](x).gelu())

    def parameters(self):
        params = []
        params.extend(self.layers[0].parameters())
        params.extend(self.layers[1].parameters())
        return params


class MultiHeadAttention:
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = layers.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = layers.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = layers.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = layers.Linear(d_out, d_out)
        self.dropout = layers.Dropout(dropout)
        mask = np.triu(np.ones((context_length, context_length)), k=1)
        self.mask = Tensor(mask, requires_grad=False)

    def __call__(self, x):
        B, T, _ = x.shape
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        K = K.reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        V = V.reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))

        attn_scores = (Q @ K.transpose((0, 1, 3, 2))) / (self.head_dim ** 0.5)
        mask = self.mask[:T, :T].bool()
        attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float('-inf'))

        attn_weights = attn_scores.softmax(axis=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ V
        context = context.transpose((0, 2, 1, 3)).reshape(B, T, self.d_out)
        return self.out_proj(context)

    def parameters(self):
        params = []
        params.extend(self.W_query.parameters())
        params.extend(self.W_key.parameters())
        params.extend(self.W_value.parameters())
        params.extend(self.out_proj.parameters())
        return params


class TransformerBlock:
    def __init__(self, config):
        self.att = MultiHeadAttention(
            d_in=config['emb_dim'],
            d_out=config['emb_dim'],
            context_length=config['context_length'],
            num_heads=config['n_heads'],
            dropout=config['drop_rate'],
            qkv_bias=config['qkv_bias']
        )
        self.ff = FeedForward(config)
        self.norm1 = layers.LayerNorm(dim=config['emb_dim'])
        self.norm2 = layers.LayerNorm(dim=config['emb_dim'])
        self.drop_shortcut = layers.Dropout(config['drop_rate'])

    def __call__(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x += shortcut
        return x

    def parameters(self):
        params = []
        params.extend(self.att.parameters())
        params.extend(self.ff.parameters())
        params.extend(self.norm1.parameters())
        params.extend(self.norm2.parameters())
        return params


class GPT2:
    def __init__(self, config):
        self.config = config
        self.tok_emb = layers.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emb = layers.Embedding(config['context_length'], config['emb_dim'])
        self.drop_emb = layers.Dropout(config['drop_rate'])
        self.trf_blocks = [TransformerBlock(config) for _ in range(config['n_layers'])]
        self.final_norm = layers.LayerNorm(dim=config['emb_dim'])
        self.out_head = layers.Linear(config['emb_dim'], config['vocab_size'], bias=False)

    def parameters(self):
        params = []
        params.extend(self.tok_emb.parameters())
        params.extend(self.pos_emb.parameters())
        params.extend(self.final_norm.parameters())
        params.extend(self.out_head.parameters())
        for block in self.trf_blocks:
            params.extend(block.parameters())
        return params

    def __call__(self, in_idx):
        if isinstance(in_idx, Tensor):
            in_idx = in_idx.data
        in_idx = np.asarray(in_idx, dtype=np.int64)

        B, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        positions = np.arange(seq_len)
        pos_embeds = self.pos_emb(positions)[None, :, :]  # (1, seq_len, emb_dim)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ----------------------------- Model & Data -----------------------------
model = GPT2(GPT_CONFIG_TINY)
print(f"Model parameter count: {sum(p.data.size for p in model.parameters()):,}")

path = kagglehub.dataset_download("rakibulhasanshaon69/the-verdict-txt")
with open(os.path.join(path, 'the-verdict.txt'), 'r', encoding='utf-8') as f:
    raw_text = f.read()

enc = tiktoken.get_encoding("gpt2")
ids = np.array(enc.encode(raw_text), dtype=np.int64)
split_idx = int(0.9 * len(ids))
train_ids = ids[:split_idx]
val_ids = ids[split_idx:]
print(f"Train tokens: {len(train_ids)}, Val tokens: {len(val_ids)}")

block_size = GPT_CONFIG_TINY["context_length"]  # training uses full model context
batch_size = 4


def get_batch(split='train'):
    data = train_ids if split == 'train' else val_ids
    ix = np.random.randint(0, len(data) - block_size, batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def to_one_hot(targets_np, vocab_size):
    B, T = targets_np.shape
    oh = np.zeros((B, T, vocab_size), dtype=np.float32)
    oh[np.arange(B)[:, None], np.arange(T)[None, :], targets_np] = 1.0
    return Tensor(oh, requires_grad=False)


def generate_text(model, start_tokens, max_tokens=100, temperature=0.8):
    context_length = GPT_CONFIG_TINY["context_length"]
    model_input = list(start_tokens)[-context_length:]  # ← FIX 1: truncate initial context
    generated = []

    for _ in range(max_tokens):
        x = np.array(model_input, dtype=np.int64)[None, :]
        logits = model(x)
        next_logits = logits[0, -1, :].data / temperature
        next_logits -= np.max(next_logits)
        probs = np.exp(next_logits)
        probs /= probs.sum()
        next_token = np.random.choice(GPT_CONFIG_TINY["vocab_size"], p=probs)

        generated.append(next_token)
        model_input.append(next_token)

        if len(model_input) > context_length:  # ← FIX 2: sliding window never exceeds context_length
            model_input = model_input[-context_length:]

    return generated


# ----------------------------- Training Loop -----------------------------
lr = 1e-3
grad_clip = 1.0
max_epochs = 1000

optimizer = nf_optim.Adam(model.parameters(), lr=lr)


def clip_grad_norm(params, max_norm):
    total_norm = np.sqrt(sum(np.sum(p.grad ** 2) for p in params if p.grad is not None))
    if total_norm > max_norm:
        coef = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad *= coef


print("Starting training...")
epoch = 0
step = 0
running_loss = 0.0
samples_since_print = 0
val_context = val_ids[:128].tolist()  # safe starting point

while epoch < max_epochs:
    xb, yb = get_batch('train')

    optimizer.zero_grad()
    logits = model(xb)
    targets_oh = to_one_hot(yb, GPT_CONFIG_TINY["vocab_size"])
    loss = nf_losses.cross_entropy_loss(logits, targets_oh)
    loss.backward()
    clip_grad_norm(model.parameters(), grad_clip)
    optimizer.step()

    running_loss += loss.item()
    samples_since_print += batch_size

    step += 1
    if step * batch_size * block_size > len(train_ids):  # approx epoch boundary
        avg_loss = running_loss / samples_since_print
        epoch += 1
        print(f"\n>>> Completed epoch {epoch}/{max_epochs}\n")
        if epoch % 10 == 0:
            gen_tokens = generate_text(model, val_context, max_tokens=100, temperature=0.8)
            sample = enc.decode(val_context + gen_tokens)

            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
            print("Generated:")
            print("-" * 50)
            print(sample)
            print("-" * 50)
        running_loss = 0.0
        samples_since_print = 0
        step = 0
        gc.collect()

print("Training finished!")