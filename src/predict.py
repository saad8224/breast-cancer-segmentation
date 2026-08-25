"""
predict.py
----------
Segment a single H&E image with the trained U-Net and save an overlay
showing the predicted nucleus contours. Handy for the demo screenshots
in the LinkedIn post.

Usage:
    python predict.py --model reports/model_best.pt \
                      --image path/to/tile.tif \
                      --out prediction.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from model import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("prediction.png"))
    p.add_argument("--threshold", type=float, default=0.40)
    args = p.parse_args()

    net = UNet().to(DEVICE)
    net.load_state_dict(torch.load(args.model, map_location=DEVICE))
    net.eval()

    img = Image.open(args.image).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_t = T.ToTensor()(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(net(img_t))[0, 0].cpu().numpy()
    mask = (probs >= args.threshold).astype(np.uint8)

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, contours, -1, (0, 255, 0), 1)

    cv2.imwrite(str(args.out), bgr)
    print(f"Saved overlay → {args.out}  ({len(contours)} nuclei detected)")


if __name__ == "__main__":
    main()
