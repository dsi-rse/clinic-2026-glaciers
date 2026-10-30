#!/usr/bin/env python
"""Score a checkpoint without relying on a fixed 0.5 cut-off.

Runs the network on every validation patch and reports:
  * pooled ROC AUC: can the network rank crevasse pixels above background pixels?
  * within-patch ROC AUC: does it place crevasses correctly INSIDE each patch?
  * precision, recall and IoU at several cut-offs, next to the random-guess precision.
It also saves a figure with the ROC curve and the training/validation loss curves.

AUC is 0.5 for random guessing and 1.0 for a perfect ranking.

Usage (from the folder that contains run_gpu100/ and patches_2021/):
    python evaluate_roc.py --run-dir run_gpu100 --patch-dir patches_2021

Requires: numpy, torch, matplotlib (and unet.py in the same folder, or --unet-dir)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save files; no window needed (works in WSL and on the cluster)
import matplotlib.pyplot as plt
import numpy as np
import torch

# A pixel counts as "crevasse" when the authors' score is above this value.
CREVASSE_SCORE = 0.5
# Number of bins used to summarise predicted probabilities over all patches.
N_BINS = 2000
# A patch needs at least this many pixels of each class for a within-patch AUC.
MIN_CLASS_PIXELS = 100
# Cut-offs at which to report precision, recall, and IoU.
CUTOFFS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
# AUC of a classifier that guesses at random.
CHANCE_AUC = 0.5


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return ROC AUC from ranks (Mann-Whitney), or NaN if a class is missing."""
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg_rank = (
        np.cumsum(counts) - (counts - 1) / 2.0
    )  # tied scores share an average rank
    ranks = avg_rank[inverse]
    u_stat = ranks[labels].sum() - n_pos * (n_pos + 1) / 2
    return float(u_stat / (n_pos * n_neg))


def hist_roc(
    pos_hist: np.ndarray, neg_hist: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the ROC curve (fpr, tpr) and its AUC from score histograms."""
    tp = np.cumsum(pos_hist[::-1])  # sweep the cut-off from high scores to low
    fp = np.cumsum(neg_hist[::-1])
    tpr = np.concatenate([[0.0], tp / tp[-1]])
    fpr = np.concatenate([[0.0], fp / fp[-1]])
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return fpr, tpr, float(integrate(tpr, fpr))


def metrics_at(
    pos_hist: np.ndarray, neg_hist: np.ndarray, cutoff: float
) -> tuple[float, float, float]:
    """Return (precision, recall, IoU) when calling scores >= cutoff crevasse."""
    idx = min(int(cutoff * N_BINS), N_BINS - 1)
    tp = float(pos_hist[idx:].sum())
    fp = float(neg_hist[idx:].sum())
    fn = float(pos_hist[:idx].sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    return precision, recall, iou


def read_log(path: Path) -> dict[str, list[float]]:
    """Read log.csv into one list of numbers per column."""
    cols: dict[str, list[float]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            for key, val in row.items():
                cols.setdefault(key, []).append(float(val))
    return cols


def main() -> None:
    """Parse the command line, score the checkpoint, and save a figure."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run-dir", required=True, help="Folder with the checkpoint and split.json"
    )
    ap.add_argument("--patch-dir", required=True, help="Folder made by make_patches.py")
    ap.add_argument(
        "--ckpt", default="best.pt", help="Checkpoint file inside --run-dir"
    )
    ap.add_argument("--unet-dir", default=None, help="Folder containing unet.py")
    args = ap.parse_args()

    if args.unet_dir:
        sys.path.insert(0, args.unet_dir)
    from unet import UNet

    run, pdir = Path(args.run_dir), Path(args.patch_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(run / args.ckpt, map_location="cpu")
    model = UNet(1, 2)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    print(f"loaded {run / args.ckpt} (epoch {ck.get('epoch', '?')}) on {device}")

    with (run / "split.json").open() as fh:
        val_ids = json.load(fh)["val_patches"]

    pos_hist = np.zeros(N_BINS)
    neg_hist = np.zeros(N_BINS)
    patch_aucs = []
    for pid in val_ids:
        z = np.load(pdir / "patches" / f"{pid}.npz")
        img = z["image"].astype(np.float32)
        lab = z["label"].astype(np.float32)
        valid = z["valid"]
        with torch.no_grad():
            x = torch.from_numpy((img - 0.5) / 0.5)[None].to(device)
            prob = torch.softmax(model(x), dim=1)[0, 1].cpu().numpy()

        is_pos = valid & (lab > CREVASSE_SCORE)
        is_neg = valid & (lab <= CREVASSE_SCORE)
        bins = np.minimum((prob * N_BINS).astype(int), N_BINS - 1)
        pos_hist += np.bincount(bins[is_pos], minlength=N_BINS)
        neg_hist += np.bincount(bins[is_neg], minlength=N_BINS)
        if is_pos.sum() >= MIN_CLASS_PIXELS and is_neg.sum() >= MIN_CLASS_PIXELS:
            patch_aucs.append(rank_auc(prob[valid], is_pos[valid]))

    fpr, tpr, pooled_auc = hist_roc(pos_hist, neg_hist)
    pos_rate = pos_hist.sum() / (pos_hist.sum() + neg_hist.sum())
    print(
        f"\nvalidation patches: {len(val_ids)}   crevasse pixels: {pos_rate * 100:.1f}% of valid"
    )
    print(f"pooled ROC AUC:       {pooled_auc:.3f}   (random guessing = {CHANCE_AUC})")
    if patch_aucs:
        print(
            f"within-patch AUC:     mean {np.mean(patch_aucs):.3f}, "
            f"median {np.median(patch_aucs):.3f}, over {len(patch_aucs)} patches "
            f"with both classes"
        )
    print(f"\nrandom-guess precision would be {pos_rate:.3f}")
    print("cut-off   precision   recall    IoU")
    for cutoff in CUTOFFS:
        p, r, iou = metrics_at(pos_hist, neg_hist, cutoff)
        print(f"  {cutoff:4.2f}      {p:6.3f}    {r:6.3f}   {iou:6.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(fpr, tpr, label=f"network (AUC {pooled_auc:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="gray", label="random guessing")
    ax[0].set_xlabel("false positive rate")
    ax[0].set_ylabel("true positive rate")
    ax[0].set_title("ROC curve (validation pixels)")
    ax[0].legend()
    log_path = run / "log.csv"
    if log_path.exists():
        log = read_log(log_path)
        ax[1].plot(log["epoch"], log["train_loss"], label="training loss")
        ax[1].plot(log["epoch"], log["val_loss"], label="validation loss")
        best = int(np.argmin(log["val_loss"]))
        ax[1].axvline(
            log["epoch"][best], color="gray", linestyle=":", label="best epoch"
        )
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("loss")
        ax[1].set_title("Training vs validation loss")
        ax[1].legend()
    fig.tight_layout()
    out_png = Path(f"{run}_eval.png")
    fig.savefig(out_png, dpi=120)
    print(f"\nfigure saved as {out_png}")


if __name__ == "__main__":
    main()
