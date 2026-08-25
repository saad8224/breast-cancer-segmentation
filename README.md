# AI-Based Early Detection and Segmentation of Breast Cancer Cells in Histopathological Images

**Capstone Project 2** — a deep-learning pipeline that segments malignant cell nuclei in H&E-stained histopathology tiles with a compact 3-tier U-Net, trained end-to-end on only 46 samples with heavy augmentation and a combined BCE+Dice loss.

## Motivation

Breast cancer diagnosis begins with a pathologist manually annotating tumour cells on histology slides — a slow, subjective task that varies between observers. A reliable segmentation model gives clinicians an objective "second reader" and lays the groundwork for downstream quantitative features (nuclear morphology, density, mitotic count).

## Dataset

- **Source:** [Kaggle — Breast Cancer Cell Segmentation (Andrew MVD)](https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation) — CC BY 3.0.
- **Content:** 58 H&E-stained image / binary-mask pairs at variable resolution.
- **Split:** 46 train / 12 test (deterministic seed 42), following the original submission.
- Place the extracted `Images/` and `Masks/` folders under `data/`.

## Method

- **Architecture** — 3-tier U-Net (`64 → 128 → 256 → 512` bottleneck) with transpose-conv upsampling and skip connections. About 7.7 M parameters — small enough to train on 46 samples without a giant risk of overfitting.
- **Input** — RGB at 256×256, values in [0, 1].
- **Loss** — combined **BCE + soft-Dice** (equal weight). BCE learns per-pixel classification; Dice pushes overlap in a strongly imbalanced foreground (nuclei cover only ~10–30 % of each tile).
- **Optimiser** — Adam, learning rate 1e-4, batch size 4.
- **Augmentation** — random horizontal / vertical flips and 90° rotations on the training set only.
- **Decision threshold** — 0.40 (paper-reported).
- **Post-processing** — OpenCV `findContours` + `drawContours` for the nuclear-boundary overlays.

## Results (test set, 12 tiles)

| Metric              | Value  |
| ------------------- | ------ |
| Pixel accuracy      | 98.12 % |
| Dice coefficient    | 25.03 % |
| Mean IoU (mIoU)     | 17.08 % |
| Precision           | 17.06 % |
| Recall              | 33.50 % |
| F1 score            | 22.50 % |

Pixel accuracy is high because the background dominates the tile; Dice and mIoU are the honest metrics for a segmentation problem this imbalanced. The 33 % recall shows the model is genuinely surfacing nuclei — the precision gap points to over-segmentation as the main failure mode, which is what the write-up recommends targeting with a larger, better-registered dataset in follow-up work.

## Pipeline

```
data/Images/*.tif                  # from Kaggle
data/Masks/*.tif
        │
        ▼
[ src/data.py ]         → deterministic 46/12 split + augmentation on train
        │
        ▼
[ src/train.py ]        → 3-tier U-Net, Adam, BCE+Dice, early stopping
                          reports/model_best.pt
                          reports/training_history.json
        │
        ▼
[ src/evaluate.py ]     → per-image + aggregate metrics on the test split
                          reports/test_metrics.json
                          reports/predictions/pred_*.png  ← contour overlays
        │
        ▼
[ src/predict.py ]      → segment a single new image
```

## How to Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) download and extract the Kaggle dataset into data/
# 2) train
python src/train.py    --data data --outdir reports --epochs 100

# 3) evaluate on the test split
python src/evaluate.py --data data --model reports/model_best.pt --outdir reports

# 4) segment a single new image
python src/predict.py  --model reports/model_best.pt \
                       --image data/Images/some_tile.tif \
                       --out sample_prediction.png
```

Training runs on CPU (slow) or CUDA (fast) automatically. On a modern GPU the whole run finishes in a couple of minutes.

## Repository Layout

```
08-breast-cancer-segmentation/
├── README.md
├── requirements.txt
├── src/
│   ├── model.py       ← 3-tier U-Net
│   ├── losses.py      ← BCE + Dice
│   ├── data.py        ← Kaggle loader, 46/12 split, augmentation
│   ├── train.py       ← training loop + early stopping
│   ├── evaluate.py    ← test-set metrics + contour overlays
│   └── predict.py     ← single-image inference
├── data/              (gitignored — download Kaggle set here)
├── notebooks/
└── reports/
    ├── Capstone_2_Final_Report.pdf   ← original submission (add your own)
    ├── model_best.pt
    ├── training_history.json
    ├── test_metrics.json
    └── predictions/*.png
```

## Honest Notes on the Findings

- **Dataset size is the ceiling** — 58 tiles is small for medical segmentation; a Dice around 25 % is what the data supports, not a limit of the architecture.
- **The precision-recall gap suggests over-segmentation** — the model finds nuclei but colours slightly more pixels than the ground-truth mask covers. Tighter mask annotations or a stain-normalisation step (Macenko / Vahadane) would likely close this gap without any change to the U-Net itself.
- **Follow-up would move to a bigger dataset** such as MoNuSeg or CoNSeP, plus test-time augmentation and a Tversky loss to trade recall for precision.

## Credits

- **Author** — Saad Salman
- **Programme** — Bachelor of Information Systems (Hons) Data Analytics
- **Supervisor** — Dr. Farrukh Hassan, School of Computing and AI
- **Dataset** — Andrew MVD (Kaggle), CC BY 3.0

---

*Course:* Capstone Project 2
*Tools:* PyTorch, OpenCV, Pillow
