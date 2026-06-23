"""
generate text from a trained checkpoint.

    python sample.py --ckpt out/ckpt.pt --prompt "Candide" --tokens 500
"""

import argparse

import torch

from data import CharTokenizer
from model import GPT, GPTConfig


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    args = p.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    tok = CharTokenizer.from_state(ckpt["tokenizer"])

    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = tok.encode(args.prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
