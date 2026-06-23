"""
char-level tokenizer + batching.

going char-level keeps everything dependency-free and easy to follow: the
"vocab" is just every unique character in the text. each char -> an int, and the
model learns to guess the next char.
"""

import torch


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}   # char -> id
        self.itos = {i: c for i, c in enumerate(chars)}   # id -> char
        self.vocab_size = len(chars)

    def encode(self, s: str):
        # skip chars we never saw at training time instead of crashing
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def state(self):
        return {"itos": self.itos, "stoi": self.stoi, "vocab_size": self.vocab_size}

    @classmethod
    def from_state(cls, state):
        # rebuild a tokenizer from a saved checkpoint without re-reading the text
        tok = cls.__new__(cls)
        tok.stoi = {k: int(v) for k, v in state["stoi"].items()}
        tok.itos = {int(k): v for k, v in state["itos"].items()}
        tok.vocab_size = state["vocab_size"]
        return tok


def load_data(path: str, train_frac: float = 0.9):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    return tok, data[:n], data[n:]   # tokenizer, train split, val split


def get_batch(data, block_size, batch_size, device):
    """grab batch_size random chunks of length block_size.

    x is the chunk, y is the same chunk shifted right by one - so y is just "the
    next char" for every position in x. that's the whole training signal.
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
