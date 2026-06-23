"""
minigpt - a small gpt (decoder-only transformer) i wrote from scratch to really
understand how attention works.

no nn.Transformer, no nn.MultiheadAttention - the whole attention block is done
by hand below so nothing is hidden. it's basically the gpt recipe: token + pos
embeddings -> a few transformer blocks -> a linear head that predicts the next
token.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 256      # filled in from the tokenizer
    block_size: int = 128      # context length - how far back the model can look
    n_layer: int = 4           # how many transformer blocks we stack
    n_head: int = 4            # attention heads per block
    n_embd: int = 128          # embedding / hidden size
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """multi-head self attention, written out the long way.

    causal = each position can only look at itself and the past, never the
    future. that masking is the whole reason this works as a language model.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd has to divide evenly by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        # one big linear gives us q, k and v at once (cheaper than 3 separate ones)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # lower triangular matrix = the causal mask. 1 means "you can look here"
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

        # keep the last attention map around so we can plot it later (viz.py).
        # purely for inspection, not used in the forward maths.
        self.last_attn = None

    def forward(self, x):
        B, T, C = x.shape  # batch, time (how many tokens), channels (n_embd)

        # project then chop into q, k, v - each is (B, T, C)
        q, k, v = self.qkv(x).split(C, dim=2)

        # reshape so each head gets its own slice -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # how much each token cares about every other token -> (B, n_head, T, T)
        # the /sqrt(head_dim) keeps the numbers small, otherwise softmax saturates
        # and the gradients basically vanish
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # block out the future: set those scores to -inf before softmax
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        self.last_attn = att.detach()
        att = self.attn_drop(att)

        # weighted sum of the values, then glue the heads back together
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """plain feed-forward net applied at every position. this is where most of
    the model's 'thinking' capacity actually sits."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)   # blow it up 4x
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)  # bring it back down
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """one transformer block. pre-norm style: normalize first, then add the
    residual back. attention to mix info across tokens, mlp to process it."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # tie the input embedding and the output head together. saves a bunch of
        # params and tends to help a little. standard gpt trick.
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence is longer than block_size"

        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)   # token id + where it sits
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross entropy over the vocab, flattened across batch * time
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """sample one token at a time, feeding our own output back in. classic
        autoregressive decoding."""
        self.eval()
        for _ in range(max_new_tokens):
            # never feed more than block_size tokens or the pos embeddings break
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # only the last step matters
            if top_k is not None:
                # keep only the top_k options, kill the rest
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
