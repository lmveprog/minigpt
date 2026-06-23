"""
make the two pictures that go in the readme:

  1. the training loss curve (from out/loss.csv)
  2. an attention heatmap - shows which previous chars each char looked at,
     pulled straight out of the first attention head after a forward pass

    python viz.py

writes assets/loss_curve.png and assets/attention.png
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")  # no gui, just save files
import matplotlib.pyplot as plt
import torch

from data import CharTokenizer
from model import GPT, GPTConfig


def plot_loss(csv_path="out/loss.csv", out="assets/loss_curve.png"):
    steps, train, val = [], [], []
    with open(csv_path) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            steps.append(int(row[0]))
            train.append(float(row[1]))
            val.append(float(row[2]))

    plt.figure(figsize=(7, 4.2))
    plt.plot(steps, train, label="train", marker="o", ms=3)
    plt.plot(steps, val, label="val", marker="o", ms=3)
    plt.axhline(4.57, ls="--", c="grey", lw=1, label="random guess (ln 97)")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title("minigpt training on Candide")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print("wrote", out)


def plot_attention(ckpt_path="out/ckpt.pt", prompt="Candide etait", out="assets/attention.png"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    tok = CharTokenizer.from_state(ckpt["tokenizer"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = torch.tensor([tok.encode(prompt)])
    with torch.no_grad():
        model(ids)  # run it once just to fill in last_attn

    # first block, first head. shape is (T, T)
    att = model.blocks[0].attn.last_attn[0, 0].numpy()
    chars = [c if c != " " else "·" for c in prompt]

    plt.figure(figsize=(5.5, 5))
    plt.imshow(att, cmap="viridis")
    plt.xticks(range(len(chars)), chars)
    plt.yticks(range(len(chars)), chars)
    plt.xlabel("attends to")
    plt.ylabel("current char")
    plt.title("attention - block 0, head 0")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    plot_loss()
    plot_attention()
