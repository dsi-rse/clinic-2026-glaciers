#!/usr/bin/env python
"""
Train the paper's shallow U-Net to reproduce the fracture map from the backscatter image,
using patches made by make_patches.py.  No bootstrapping: the released fracture map is the
training target directly.

  * Network: UNet(1, 2) from the paper's unet.py (2 outputs = "no crevasse" / "crevasse")
  * Target: the fracture score y in [0,1] used as a SOFT label, i.e. the network is asked
    to predict probabilities [1-y, y].  Use --binary-threshold 0.5 to harden it instead.
  * Split: by geographic block (default 500 km squares) so neighbouring patches never end
    up on both sides of the train/validation split.
  * Checkpoints keep the paper's layout: torch.load(path)['model_state_dict'].

Usage:
  python train_unet.py --patch-dir patches_2021 --out-dir run1 --unet-dir C:/path/to/pub_modules

Requires: numpy, torch
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def read_index(patch_dir):
    with open(Path(patch_dir) / "index.csv") as fh:
        return list(csv.DictReader(fh))


def split_by_block(rows, val_frac, block_km, seed):
    key = lambda r: f"{int(float(r['x_m']) // (block_km * 1000))}_{int(float(r['y_m']) // (block_km * 1000))}"
    groups = sorted({key(r) for r in rows})
    if len(groups) < 2:
        raise SystemExit("Only one geographic block found; lower --block-km to get a split.")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(val_frac * len(groups))))
    val_groups = set(groups[:n_val])
    train = [r for r in rows if key(r) not in val_groups]
    val = [r for r in rows if key(r) in val_groups]
    return train, val, sorted(val_groups)


class PatchDataset(Dataset):
    def __init__(self, patch_dir, rows, augment, binary_threshold):
        self.dir = Path(patch_dir) / "patches"
        self.ids = [r["patch_id"] for r in rows]
        self.augment = augment
        self.thr = binary_threshold

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        z = np.load(self.dir / f"{self.ids[i]}.npz")
        img = z["image"].astype(np.float32)            # (1,P,P) in [0,1]
        y = z["label"].astype(np.float32)              # (P,P) in [0,1]
        valid = z["valid"].astype(np.float32)          # (P,P)
        if self.thr is not None:
            y = (y > self.thr).astype(np.float32)
        if self.augment:
            # flips change apparent radar look direction; they are optional (--augment none)
            if np.random.rand() < 0.5:
                img, y, valid = img[:, :, ::-1], y[:, ::-1], valid[:, ::-1]
            if np.random.rand() < 0.5:
                img, y, valid = img[:, ::-1, :], y[::-1, :], valid[::-1, :]
        img = (img - 0.5) / 0.5                        # same rescaling as the paper's script
        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in (img, y, valid))


def soft_ce(logits, y, valid, class_w):
    """Cross-entropy against soft targets [1-y, y], ignoring invalid pixels."""
    target = torch.stack([1 - y, y], dim=1)
    ce = F.cross_entropy(logits, target, weight=class_w, reduction="none")      # (N,H,W)
    denom = (valid * (class_w[0] * (1 - y) + class_w[1] * y)).sum().clamp(min=1e-6)
    return (ce * valid).sum() / denom


@torch.no_grad()
def evaluate(model, loader, device, class_w, thresh):
    model.eval()
    loss_sum, n = 0.0, 0
    tp = fp = fn = 0
    abs_err, n_pix = 0.0, 0
    for img, y, valid in loader:
        img, y, valid = img.to(device), y.to(device), valid.to(device)
        logits = model(img)
        loss_sum += soft_ce(logits, y, valid, class_w).item()
        n += 1
        prob = torch.softmax(logits, dim=1)[:, 1]
        known = valid > 0
        pred, truth = prob > thresh, y > 0.5
        tp += (pred & truth & known).sum().item()
        fp += (pred & ~truth & known).sum().item()
        fn += (~pred & truth & known).sum().item()
        abs_err += ((prob - y).abs() * valid).sum().item()
        n_pix += valid.sum().item()
    safe = lambda a, b: a / b if b else float("nan")
    p, r = safe(tp, tp + fp), safe(tp, tp + fn)
    return {"val_loss": loss_sum / max(n, 1), "precision": p, "recall": r,
            "iou": safe(tp, tp + fp + fn), "mean_abs_err": safe(abs_err, n_pix)}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--unet-dir", default=None, help="Folder containing the paper's unet.py")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--block-km", type=float, default=500.0, help="Size of geographic blocks for the split")
    ap.add_argument("--pos-weight", type=float, default=3.0, help="Class weight for 'crevasse'")
    ap.add_argument("--binary-threshold", type=float, default=None,
                    help="If set (e.g. 0.5), harden labels to 0/1 instead of using soft scores")
    ap.add_argument("--augment", choices=["none", "flip"], default="flip")
    ap.add_argument("--init", default=None, help="Checkpoint to start from")
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability cut-off for metrics")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.unet_dir:
        sys.path.insert(0, args.unet_dir)
    from unet import UNet

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_index(args.patch_dir)
    train_rows, val_rows, val_groups = split_by_block(rows, args.val_frac, args.block_km, args.seed)
    json.dump({"val_blocks": val_groups, "train_patches": [r["patch_id"] for r in train_rows],
               "val_patches": [r["patch_id"] for r in val_rows]}, open(out / "split.json", "w"), indent=1)
    print(f"patches: {len(train_rows)} train / {len(val_rows)} val   (val blocks: {len(val_groups)})")

    w = np.array([float(r["valid_frac"]) for r in train_rows])
    pos_share = np.average([float(r["pos_frac"]) for r in train_rows], weights=w)
    print(f"share of valid train pixels with score > 0.5: {100 * pos_share:.2f}%  (guide for --pos-weight)")

    train_ds = PatchDataset(args.patch_dir, train_rows, args.augment == "flip", args.binary_threshold)
    val_ds = PatchDataset(args.patch_dir, val_rows, False, args.binary_threshold)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    patch = train_ds[0][0].shape[-1]
    if patch % 8:
        raise SystemExit(f"Patch size {patch} must be divisible by 8.")
    device = pick_device()
    print(f"device: {device}")
    model = UNet(1, 2)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model_state_dict"])
        print(f"initialised from {args.init}")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_w = torch.tensor([1.0, args.pos_weight], device=device)

    best, fields = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for img, y, valid in train_dl:
            img, y, valid = img.to(device), y.to(device), valid.to(device)
            opt.zero_grad()
            loss = soft_ce(model(img), y, valid, class_w)
            loss.backward()
            opt.step()
            run += loss.item()
            n += 1
        m = evaluate(model, val_dl, device, class_w, args.threshold)
        rec = {"epoch": epoch, "train_loss": run / max(n, 1), **m, "seconds": round(time.time() - t0, 1)}
        print(f"epoch {epoch:02d}  train {rec['train_loss']:.4f}  val {rec['val_loss']:.4f}  "
              f"P/R/IoU {rec['precision']:.3f}/{rec['recall']:.3f}/{rec['iou']:.3f}  MAE {rec['mean_abs_err']:.3f}")
        if fields is None:
            fields = list(rec)
            with open(out / "log.csv", "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=fields).writeheader()
        with open(out / "log.csv", "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(rec)

        ck = {"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
              "epoch": epoch, "args": vars(args)}
        torch.save(ck, out / f"ckpt_e{epoch:02d}.pt")
        if rec["val_loss"] < best:
            best = rec["val_loss"]
            torch.save(ck, out / "best.pt")
    print(f"done. best validation loss {best:.4f} (saved as best.pt)")


if __name__ == "__main__":
    main()
