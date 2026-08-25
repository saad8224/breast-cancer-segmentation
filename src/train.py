"""
train.py
--------
Train the 3-tier U-Net on the Kaggle breast-cancer H&E dataset.

Reproduces the Capstone 2 training setup:
    - 46 train / 12 test split, deterministic seed 42
    - Adam optimiser, lr = 1e-4
    - Combined BCE + Dice loss
    - Batch size 4, up to 100 epochs, early-stopping on val Dice
    - Best model saved to reports/model_best.pt

Usage:
    python train.py --data data --outdir reports --epochs 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import make_loaders
from losses import BCEDiceLoss
from model import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def _dice(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.4) -> float:
    """Hard-threshold Dice for monitoring."""
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    inter = (preds * targets).sum().item()
    denom = preds.sum().item() + targets.sum().item()
    return (2 * inter + 1) / (denom + 1)


def train_epoch(net, loader, criterion, optim) -> float:
    net.train()
    total = 0.0
    for img, mask in loader:
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        optim.zero_grad()
        logits = net(img)
        loss = criterion(logits, mask)
        loss.backward()
        optim.step()
        total += loss.item() * img.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(net, loader, criterion) -> tuple[float, float]:
    net.eval()
    total_loss = 0.0
    total_dice = 0.0
    for img, mask in loader:
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        logits = net(img)
        total_loss += criterion(logits, mask).item() * img.size(0)
        total_dice += _dice(logits, mask) * img.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_dice / n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("reports"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    args = p.parse_args()

    torch.manual_seed(SEED)
    args.outdir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = make_loaders(str(args.data), batch_size=args.batch_size)
    print(f"Train: {len(train_loader.dataset)} · Test: {len(test_loader.dataset)}")
    print(f"Device: {DEVICE}")

    net = UNet().to(DEVICE)
    criterion = BCEDiceLoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    history = []
    best_dice = -1.0
    since_best = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(net, train_loader, criterion, optim)
        va_loss, va_dice = eval_epoch(net, test_loader, criterion)
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "val_loss": va_loss, "val_dice": va_dice})
        print(f"Epoch {epoch:3d} | train {tr_loss:.4f} | val {va_loss:.4f} | dice {va_dice:.4f}")

        if va_dice > best_dice:
            best_dice = va_dice
            since_best = 0
            torch.save(net.state_dict(), args.outdir / "model_best.pt")
        else:
            since_best += 1
            if since_best >= args.patience:
                print(f"Early stop after {epoch} epochs (best dice {best_dice:.4f}).")
                break

    (args.outdir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nDone. Best val Dice = {best_dice:.4f}. Model → {args.outdir/'model_best.pt'}")


if __name__ == "__main__":
    main()
