#!/usr/bin/env python
"""
Look at what a trained checkpoint actually predicts.

For several validation patches it saves a picture with four panels:
  1. backscatter (the network's input)
  2. the authors' fracture map (the target)
  3. your network's crevasse probability, on a fixed 0-1 scale
  4. the same prediction, stretched to its own min/max (reveals faint structure)

It also prints numbers that help diagnose a network that "predicts nothing":
mean/max predicted probability, and the mean prediction on true-crevasse pixels
versus background pixels. A network that learned anything should score higher on
crevasse pixels than on background pixels.

Run from the folder that contains run2/ and patches_2021/:
  python plot_predictions.py --run-dir run2 --patch-dir patches_2021

Requires: numpy, torch, matplotlib   (and unet.py in the same folder, or --unet-dir)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # save files; no window needed (works in WSL)
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Folder with the checkpoint and split.json, e.g. run2")
    ap.add_argument("--patch-dir", required=True, help="Folder made by make_patches.py, e.g. patches_2021")
    ap.add_argument("--ckpt", default="best.pt", help="Checkpoint file inside --run-dir")
    ap.add_argument("--n", type=int, default=6, help="How many crevasse-rich patches to plot")
    ap.add_argument("--n-empty", type=int, default=2, help="How many crevasse-free patches to also plot")
    ap.add_argument("--unet-dir", default=None, help="Folder containing unet.py")
    args = ap.parse_args()

    if args.unet_dir:
        sys.path.insert(0, args.unet_dir)
    from unet import UNet

    run, pdir = Path(args.run_dir), Path(args.patch_dir)
    out_dir = Path(str(run) + "_plots")
    out_dir.mkdir(exist_ok=True)

    # model
    ck = torch.load(run / args.ckpt, map_location="cpu")
    model = UNet(1, 2)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"loaded {run / args.ckpt} (epoch {ck.get('epoch', '?')})")

    # pick validation patches: most crevasse-rich first, plus a few empty ones
    val_ids = set(json.load(open(run / "split.json"))["val_patches"])
    with open(pdir / "index.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["patch_id"] in val_ids]
    rows.sort(key=lambda r: float(r["pos_frac"]), reverse=True)
    chosen = rows[: args.n] + [r for r in rows[::-1] if float(r["pos_frac"]) < 0.001][: args.n_empty]

    all_pos, all_neg = [], []
    for r in chosen:
        z = np.load(pdir / "patches" / f"{r['patch_id']}.npz")
        img = z["image"].astype(np.float32)        # (1,P,P) in [0,1]
        lab = z["label"].astype(np.float32)        # (P,P)
        valid = z["valid"]

        with torch.no_grad():
            x = torch.from_numpy((img - 0.5) / 0.5)[None]
            prob = torch.softmax(model(x), dim=1)[0, 1].numpy()

        pos, neg = valid & (lab > 0.5), valid & (lab <= 0.5)
        if pos.any():
            all_pos.append(prob[pos].mean())
        if neg.any():
            all_neg.append(prob[neg].mean())
        print(f"{r['patch_id']:>14}  crevasse share {float(r['pos_frac']) * 100:5.1f}%   "
              f"pred mean {prob[valid].mean():.3f} max {prob[valid].max():.3f}   "
              f"mean on crevasse px {prob[pos].mean() if pos.any() else float('nan'):.3f}   "
              f"mean on background px {prob[neg].mean() if neg.any() else float('nan'):.3f}")

        fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
        ax[0].imshow(img[0], cmap="gray", vmin=0, vmax=1); ax[0].set_title("backscatter (input)")
        ax[1].imshow(np.where(valid, lab, np.nan), vmin=0, vmax=1); ax[1].set_title("authors' fracture map")
        ax[2].imshow(np.where(valid, prob, np.nan), vmin=0, vmax=1); ax[2].set_title("prediction (0-1 scale)")
        ax[3].imshow(np.where(valid, prob, np.nan)); ax[3].set_title(f"prediction (stretched {prob.min():.2f}-{prob.max():.2f})")
        for a in ax:
            a.axis("off")
        fig.suptitle(r["patch_id"])
        fig.tight_layout()
        fig.savefig(out_dir / f"check_{r['patch_id']}.png", dpi=100)
        plt.close(fig)

    if all_pos and all_neg:
        print(f"\naverage prediction on crevasse pixels:   {np.mean(all_pos):.3f}")
        print(f"average prediction on background pixels: {np.mean(all_neg):.3f}")
        print("(if these are about equal, the network has not learned to separate them)")
    print(f"\npictures saved in {out_dir}/")


if __name__ == "__main__":
    main()