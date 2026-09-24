"""Train a small character-level GPT on Shakespeare with nnetflow.

Run a short demonstration with::

    uv run python examples/gpt.py --steps 200

The script downloads the Tiny Shakespeare corpus on the first run. If the
network is unavailable, pass ``--text path/to/corpus.txt`` or use the built-in
short fallback corpus.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from nnetflow import Tensor
from nnetflow.layers import Embedding, LayerNorm, Linear, MultiHeadAttention
from nnetflow.module import Module
from nnetflow.optim import Adam


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
FALLBACK_TEXT = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles.\n"
) * 32


class Dataset:
    """Character windows and their one-step-shifted targets."""

    def __init__(self, token_ids: np.ndarray, block_size: int) -> None:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional array")
        if len(token_ids) <= block_size:
            raise ValueError("The corpus must be longer than block_size")
        self.token_ids = token_ids.astype(np.int64, copy=False)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        start = int(index)
        x = self.token_ids[start : start + self.block_size]
        target = int(self.token_ids[start + self.block_size])
        return x, target


class DataLoader:
    """Small NumPy DataLoader matching the regression example's interface."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(dataset))
        self.position = 0

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        self.position = 0
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(self.indices)
        return self

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __next__(self) -> tuple[Tensor, Tensor]:
        if self.position >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.position : self.position + self.batch_size]
        inputs = []
        targets = []
        for index in batch_indices:
            input_ids, target = self.dataset[int(index)]
            inputs.append(input_ids)
            targets.append(target)
        self.position += self.batch_size
        return (
            Tensor(np.stack(inputs).astype(np.int64), requires_grad=False),
            Tensor(np.asarray(targets, dtype=np.int64), requires_grad=False),
        )


class GPTBlock(Module):
    """A pre-normalized transformer block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        block_size: int,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(
            embedding_dim,
            embedding_dim,
            num_heads,
            dropout=dropout,
            causal=True,
            max_seq_len=block_size,
            dtype=np.float32,
        )
        self.norm2 = LayerNorm(embedding_dim)
        self.feed_forward1 = Linear(embedding_dim, 4 * embedding_dim, dtype=np.float32)
        self.feed_forward2 = Linear(4 * embedding_dim, embedding_dim, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        feed_forward = self.feed_forward2(self.feed_forward1(self.norm2(x)).gelu())
        return x + feed_forward


class CharacterGPT(Module):
    """A small decoder-only character language model."""

    def __init__(
        self,
        vocabulary_size: int,
        block_size: int = 64,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.embedding = Embedding(vocabulary_size, embedding_dim, dtype=np.float32)
        self.position_embedding = Embedding(block_size, embedding_dim, dtype=np.float32)
        self.blocks = [
            GPTBlock(embedding_dim, num_heads, dropout, block_size)
            for _ in range(num_layers)
        ]
        self.final_norm = LayerNorm(embedding_dim)
        self.lm_head = Linear(embedding_dim, vocabulary_size, dtype=np.float32)

    def forward(self, token_ids: Tensor) -> Tensor:
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.block_size:
            raise ValueError("Input sequence is longer than block_size")
        positions = np.arange(sequence_length, dtype=np.int64)
        token_vectors = self.embedding(token_ids.data)
        position_vectors = self.position_embedding(positions)
        x = token_vectors + position_vectors
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def load_text(path: Optional[str]) -> str:
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    try:
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=10) as response:
            return response.read().decode("utf-8")
    except Exception as error:
        print(f"Could not download Shakespeare corpus ({error}); using fallback text.")
        return FALLBACK_TEXT


def make_vocabulary(text: str) -> tuple[dict[str, int], dict[int, str], np.ndarray]:
    characters = sorted(set(text))
    encode = {character: index for index, character in enumerate(characters)}
    decode = {index: character for character, index in encode.items()}
    token_ids = np.asarray([encode[character] for character in text], dtype=np.int64)
    return encode, decode, token_ids


def cross_entropy(logits: Tensor, targets: Tensor, vocabulary_size: int) -> Tensor:
    """Cross entropy for logits shaped (batch, time, vocabulary)."""
    target_data = np.zeros((targets.shape[0], vocabulary_size), dtype=np.float32)
    target_data[np.arange(targets.shape[0]), targets.data] = 1.0
    one_hot = Tensor(target_data, requires_grad=False, dtype=np.float32)
    final_logits = logits[:, -1, :]
    return -(one_hot * final_logits.log_softmax(axis=-1)).sum(axis=-1).mean()


def train(
    model: CharacterGPT,
    dataloader: DataLoader,
    optimizer: Adam,
    vocabulary_size: int,
    steps: int,
    report_every: int,
) -> None:
    batches = iter(dataloader)
    for step in range(1, steps + 1):
        try:
            inputs, targets = next(batches)
        except StopIteration:
            batches = iter(dataloader)
            inputs, targets = next(batches)
        optimizer.zero_grad()
        loss = cross_entropy(model(inputs), targets, vocabulary_size)
        loss.backward()
        optimizer.step()
        if step == 1 or step % report_every == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}")


def generate(
    model: CharacterGPT,
    prompt: str,
    encode: dict[str, int],
    decode: dict[int, str],
    max_new_tokens: int,
    temperature: float,
) -> str:
    if not prompt:
        prompt = "\n"
    unknown = set(prompt) - set(encode)
    if unknown:
        raise ValueError(f"Prompt contains characters outside the vocabulary: {unknown}")
    token_ids = [encode[character] for character in prompt]
    for _ in range(max_new_tokens):
        context = np.asarray(token_ids[-model.block_size :], dtype=np.int64)[None, :]
        logits = model(Tensor(context, requires_grad=False)).data[0, -1]
        scaled = logits / max(temperature, 1e-6)
        probabilities = np.exp(scaled - np.max(scaled))
        probabilities /= probabilities.sum()
        token_ids.append(int(np.random.choice(len(probabilities), p=probabilities)))
    return "".join(decode[index] for index in token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Path to a UTF-8 Shakespeare-style corpus")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    np.random.seed(args.seed)
    text = load_text(args.text)
    encode, decode, token_ids = make_vocabulary(text)
    dataset = Dataset(token_ids, args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, seed=args.seed)
    model = CharacterGPT(
        len(encode),
        block_size=args.block_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.heads,
        num_layers=args.layers,
    )
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    print(f"corpus: {len(text):,} characters | vocabulary: {len(encode)} characters")
    print(f"parameters: {sum(parameter.data.size for parameter in model.parameters()):,}")
    train(model, dataloader, optimizer, len(encode), args.steps, report_every=max(1, args.steps // 10))
    model.eval()
    print("\n" + generate(model, args.prompt, encode, decode, args.new_tokens, args.temperature))


if __name__ == "__main__":
    main()
