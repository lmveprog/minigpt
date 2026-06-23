"""
training loop.

    python train.py --data data/corpus.txt --steps 3000

grabs whatever device is around (cuda > apple mps > cpu), trains with adamw,
checks train/val loss every once in a while, dumps the curve to out/loss.csv and
saves the final checkpoint (weights + tokenizer + config) to out/ckpt.pt.
"""

import argparse
import json
import os
import time

import torch

from data import load_data, get_batch
from model import GPT, GPTConfig


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, batch_size, device, iters=50):
    # average loss over a handful of batches so the number isn't pure noise
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, cfg.block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/corpus.txt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--out", default="out")
    args = p.parse_args()

    device = pick_device()
    os.makedirs(args.out, exist_ok=True)
    print(f"device: {device}")

    tok, train_data, val_data = load_data(args.data)
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(cfg).to(device)
    print(f"vocab_size={tok.vocab_size}  params={model.num_params()/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_log = [("step", "train_loss", "val_loss")]
    t0 = time.time()

    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            m = estimate_loss(model, train_data, val_data, cfg, args.batch_size, device)
            dt = time.time() - t0
            print(f"step {step:5d} | train {m['train']:.4f} | val {m['val']:.4f} | {dt:.0f}s")
            loss_log.append((step, round(m["train"], 4), round(m["val"], 4)))

        x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stops the odd exploding grad
        opt.step()

    # save the weights + the loss curve so we can plot it later
    ckpt = {
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "tokenizer": tok.state(),
    }
    torch.save(ckpt, os.path.join(args.out, "ckpt.pt"))
    with open(os.path.join(args.out, "loss.csv"), "w") as f:
        for row in loss_log:
            f.write(",".join(map(str, row)) + "\n")
    print(f"saved checkpoint -> {os.path.join(args.out, 'ckpt.pt')}")


if __name__ == "__main__":
    main()
