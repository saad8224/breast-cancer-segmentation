"""
evaluate.py
-----------
Run the trained U-Net on the test split and report every metric that
appears in the Capstone 2 write-up: pixel accuracy, Dice, mean IoU,
Precision, Recall, F1. Also renders OpenCV contour overlays so you can
eyeball the segmentation quality.

Usage:
    python evaluate.py --data data --model reports/model_best.pt \
                       --outdir reports --threshold 0.40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from data import make_loaders
from model import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Binary segmentation metrics on flattened masks."""
    pred = pred.astype(bool).ravel()
    gt = gt.astype(bool).ravel()

    tp = int(np.logical_and(pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    pixel_acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    iou_pos = tp / max(tp + fp + fn, 1)
    iou_neg = tn / max(tn + fp + fn, 1)
    miou = (iou_pos + iou_neg) / 2
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)

    return {
        "pixel_accuracy": pixel_acc,
        "dice": dice,
        "mIoU": miou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _overlay(img_rgb: np.ndarray, mask: np.ndarray, colour=(0, 255, 0)) -> np.ndarray:
    """Draw contours of `mask` on top of `img_rgb` (BGR-in for cv2)."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, contours, -1, colour, 1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("reports"))
    p.add_argument("--threshold", type=float, default=0.40)
    args = p.parse_args()

    figdir = args.outdir / "predictions"
    figdir.mkdir(parents=True, exist_ok=True)

    _, test_loader = make_loaders(str(args.data), batch_size=1)
    net = UNet().to(DEVICE)
    net.load_state_dict(torch.load(args.model, map_location=DEVICE))
    net.eval()

    per_image = []
    with torch.no_grad():
        for i, (img_t, mask_t) in enumerate(test_loader):
            img_t, mask_t = img_t.to(DEVICE), mask_t.to(DEVICE)
            logits = net(img_t)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (probs >= args.threshold).astype(np.uint8)
            gt = mask_t[0, 0].cpu().numpy().astype(np.uint8)
            per_image.append(_metrics(pred, gt))

            img_np = (img_t[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            gt_overlay = _overlay(img_np, gt, colour=(255, 0, 0))     # ground truth in red
            pred_overlay = _overlay(gt_overlay, pred, colour=(0, 255, 0))  # prediction in green
            cv2.imwrite(str(figdir / f"pred_{i:02d}.png"),
                        cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))

    # Aggregate — the paper reports means across the test set.
    keys = per_image[0].keys()
    summary = {k: float(np.mean([m[k] for m in per_image])) for k in keys}
    summary["threshold"] = args.threshold
    summary["test_size"] = len(per_image)

    (args.outdir / "test_metrics.json").write_text(json.dumps(summary, indent=2))

    print("\nTest-set metrics")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:>16s}: {v:.4f}")
        else:
            print(f"  {k:>16s}: {v}")


if __name__ == "__main__":
    main()
