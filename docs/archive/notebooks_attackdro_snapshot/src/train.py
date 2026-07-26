"""Minimal PyTorch training loop.

Purpose: a self-contained, runnable starting point that proves the full remote
pipeline works (Mac -> SSH -> WSL2 -> GPU). It trains a small MLP on synthetic
data. Replace the model and dataset with your paper's real ones.

    python src/train.py --config configs/default.yaml
"""

import argparse
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_data(cfg: dict) -> TensorDataset:
    g = torch.Generator().manual_seed(cfg["seed"])
    x = torch.randn(cfg["num_samples"], cfg["input_dim"], generator=g)
    # A learnable synthetic target: linear projection + argmax into class labels.
    w = torch.randn(cfg["input_dim"], cfg["output_dim"], generator=g)
    y = (x @ w).argmax(dim=1)
    return TensorDataset(x, y)


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["input_dim"], cfg["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["hidden_dim"], cfg["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["hidden_dim"], cfg["output_dim"]),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    torch.manual_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    loader = DataLoader(
        make_data(cfg), batch_size=cfg["batch_size"], shuffle=True, drop_last=True
    )
    model = MLP(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        total, correct, running = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

            running += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)

        print(
            f"epoch {epoch:02d} | loss {running / total:.4f} | "
            f"acc {correct / total:.3f} | {time.time() - t0:.1f}s"
        )

    print("done.")


if __name__ == "__main__":
    main()
