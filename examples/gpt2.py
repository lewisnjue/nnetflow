import os
import numpy as np
import time
import gc
from nnetflow.engine import Tensor
from nnetflow.module import Module
from nnetflow import layers
from nnetflow import losses as nf_losses
from nnetflow import optim as nf_optim
from nnetflow.device import set_device, get_array_module, is_gpu_available, gpu_supports_dtype
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

# Set device - use GPU if available
if is_gpu_available():
    set_device('cuda', device_id=0)
    print("Using GPU for training")
    # Check if GPU supports float16 for faster training
    use_fp16 = gpu_supports_dtype(np.float16)
    if use_fp16:
        print("GPU supports float16 - using float16 for faster training")
    else:
        print("GPU does not support float16 - using float32")
        use_fp16 = False
else:
    set_device('cpu')
    print("Using CPU for training")
    use_fp16 = False  # CPU training typically uses float32

class FeedForward(Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = [
            layers.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            layers.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        ]

    def forward(self, x):
        return self.layers[1](self.layers[0](x).gelu())


class MultiHeadAttention(Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = layers.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = layers.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = layers.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = layers.Linear(d_out, d_out)
        self.dropout = layers.Dropout(dropout)
        xp = get_array_module()
        mask = xp.triu(xp.ones((context_length, context_length)), k=1)
        self.mask = Tensor(mask, requires_grad=False)

    def forward(self, x):
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


class TransformerBlock(Module):

    def __init__(self, config):
        super().__init__() 
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

    def forward(self, x):
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

class GPT2(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = layers.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emb = layers.Embedding(config['context_length'], config['emb_dim'])
        self.drop_emb = layers.Dropout(config['drop_rate'])
        self.trf_blocks = [TransformerBlock(config) for _ in range(config['n_layers'])]
        self.final_norm = layers.LayerNorm(dim=config['emb_dim'])
        self.out_head = layers.Linear(config['emb_dim'], config['vocab_size'], bias=False)

    def forward(self, in_idx):
        xp = get_array_module()
        if isinstance(in_idx, Tensor):
            in_idx = in_idx.data
        # Convert to device array (works for both numpy and cupy)
        if hasattr(in_idx, 'dtype'):
            in_idx = xp.asarray(in_idx, dtype=xp.int64)
        else:
            in_idx = xp.asarray(in_idx, dtype=xp.int64)

        B, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        positions = xp.arange(seq_len)
        pos_embeds = self.pos_emb(positions)[None, :, :]  # (1, seq_len, emb_dim)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# Create model with appropriate dtype
if use_fp16:
    model = GPT2(GPT_CONFIG_TINY).to(np.float16)  # Use float16 on GPU for faster computation
else:
    model = GPT2(GPT_CONFIG_TINY).to(np.float32)  # Use float32 on CPU or if GPU doesn't support float16
print(f"Model parameter count: {sum(p.data.size for p in model.parameters()):,}")

path = kagglehub.dataset_download("rakibulhasanshaon69/the-verdict-txt")
with open(os.path.join(path, 'the-verdict.txt'), 'r', encoding='utf-8') as f:
    raw_text = f.read()

enc = tiktoken.get_encoding("gpt2")
xp = get_array_module()
ids = xp.array(enc.encode(raw_text), dtype=xp.int64)
split_idx = int(0.9 * len(ids))
train_ids = ids[:split_idx]
val_ids = ids[split_idx:]
print(f"Train tokens: {len(train_ids)}, Val tokens: {len(val_ids)}")

block_size = GPT_CONFIG_TINY["context_length"]  # training uses full model context
batch_size = 4


def get_batch(split='train'):
    xp = get_array_module()
    data = train_ids if split == 'train' else val_ids
    ix = xp.random.randint(0, len(data) - block_size, batch_size)
    x = xp.stack([data[i:i + block_size] for i in ix])
    y = xp.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def to_one_hot(targets_np, vocab_size):
    xp = get_array_module()
    B, T = targets_np.shape
    dtype = np.float16 if use_fp16 else np.float32
    oh = xp.zeros((B, T, vocab_size), dtype=dtype)
    oh[xp.arange(B)[:, None], xp.arange(T)[None, :], targets_np] = 1.0
    return Tensor(oh, requires_grad=False, dtype=dtype)


def generate_text(model, start_tokens, max_tokens=100, temperature=0.8):
    xp = get_array_module()
    context_length = GPT_CONFIG_TINY["context_length"]
    model_input = list(start_tokens)[-context_length:] 
    generated = []

    for _ in range(max_tokens):
        x = xp.array(model_input, dtype=xp.int16)[None, :]
        logits = model(x)
        next_logits = logits[0, -1, :].data / temperature
        next_logits -= xp.max(next_logits)
        probs = xp.exp(next_logits)
        probs /= probs.sum()
        # For random choice, convert to numpy if using cupy
        if xp is not np:
            probs_np = np.asarray(probs)
            next_token = int(np.random.choice(GPT_CONFIG_TINY["vocab_size"], p=probs_np))
        else:
            next_token = int(xp.random.choice(GPT_CONFIG_TINY["vocab_size"], p=probs))

        generated.append(next_token)
        model_input.append(next_token)

        if len(model_input) > context_length:  # ← FIX 2: sliding window never exceeds context_length
            model_input = model_input[-context_length:]

    return generated


lr = 1e-3
grad_clip = 1.0
max_epochs = 1000

optimizer = nf_optim.Adam(model.parameters(), lr=lr)


def clip_grad_norm(params, max_norm):
    xp = get_array_module()
    total_norm = xp.sqrt(sum(xp.sum(p.grad ** 2) for p in params if p.grad is not None))
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
# Convert val_context to list for generation (needs to be Python list)
val_context_np = val_ids[:128]
if hasattr(val_context_np, 'tolist'):
    val_context = val_context_np.tolist()
else:
    val_context = list(val_context_np)

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