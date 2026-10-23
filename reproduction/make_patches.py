#!/usr/bin/env python
"""
Cut training patches from the paper's released GeoTIFFs:
    202106_backscatter_map.tiff  (Sentinel-1 backscatter, dB, 100 m)
    202106_fracture_map.tiff     (fracture score 0..1, same grid)

The files are continent-sized (~45,000 x 55,000 pixels), so this reads one strip of
rows at a time and never loads a whole file.

Per patch it saves (in <out-dir>/patches/<id>.npz):
    image  (1,P,P) float16  backscatter scaled to 0..1 (no-data -> 0)
    label  (P,P)   float16  fracture score 0..1 (soft target; no-data -> 0)
    valid  (P,P)   bool     True where BOTH files have data
and one row per patch in <out-dir>/index.csv (including map position, used later
to split train/validation by geographic block).

Usage:
  python make_patches.py \
      --backscatter ../data/202106_backscatter_map.tiff \
      --fracture    ../data/202106_fracture_map.tiff \
      --out-dir patches_2021

Requires: numpy, rasterio
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backscatter", required=True)
    ap.add_argument("--fracture", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256, help="256 = no overlap; 128 = 4x more patches")
    ap.add_argument("--db-min", type=float, default=-25.0, help="dB value mapped to 0")
    ap.add_argument("--db-max", type=float, default=5.0, help="dB value mapped to 1")
    ap.add_argument("--min-valid-frac", type=float, default=0.9,
                    help="Skip patches with less than this fraction of pixels having data in both files")
    ap.add_argument("--empty-thresh", type=float, default=0.001,
                    help="A patch is 'empty' if fewer than this fraction of valid pixels have score > 0.5")
    ap.add_argument("--empty-keep-prob", type=float, default=1.0,
                    help="Chance of keeping an empty patch (lower it to rebalance toward crevassed patches)")
    ap.add_argument("--keep-prob", type=float, default=1.0,
                    help="Randomly keep only this fraction of all patches (to shrink the dataset)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    (out / "patches").mkdir(parents=True, exist_ok=True)
    P, S = args.patch_size, args.stride

    rows = []
    n_seen = n_lowvalid = n_empty_dropped = 0
    with rasterio.open(args.backscatter) as sb, rasterio.open(args.fracture) as sf:
        # the two files must sit on the same grid
        if sb.crs != sf.crs or sb.res != sf.res:
            raise SystemExit("CRS or pixel size differ between the two files.")
        if (sb.transform.c, sb.transform.f) != (sf.transform.c, sf.transform.f):
            raise SystemExit("The two files have different top-left corners; they would need reprojecting.")

        H = min(sb.height, sf.height)
        W = min(sb.width, sf.width)          # fracture map is narrower: crop to the common area
        res = sb.res[0]
        x_left, y_top = sb.transform.c, sb.transform.f
        print(f"common area: {H} x {W} pixels, pixel size {res} m")

        for r0 in range(0, H - P + 1, S):
            img_strip = sb.read(1, window=Window(0, r0, W, P)).astype(np.float32)
            lab_strip = sf.read(1, window=Window(0, r0, W, P)).astype(np.float32)
            valid_strip = np.isfinite(img_strip) & np.isfinite(lab_strip)
            if not valid_strip.any():
                continue

            for c0 in range(0, W - P + 1, S):
                v = valid_strip[:, c0:c0 + P]
                n_valid = int(v.sum())
                n_seen += 1
                if n_valid / (P * P) < max(args.min_valid_frac, 1e-6):
                    n_lowvalid += 1
                    continue

                lab = np.where(v, np.clip(lab_strip[:, c0:c0 + P], 0, 1), 0.0)
                pos_frac = float(((lab > 0.5) & v).sum() / n_valid)
                if pos_frac < args.empty_thresh and rng.random() > args.empty_keep_prob:
                    n_empty_dropped += 1
                    continue
                if rng.random() > args.keep_prob:
                    continue

                img = np.where(v, img_strip[:, c0:c0 + P], args.db_min)
                img = np.clip((img - args.db_min) / (args.db_max - args.db_min), 0, 1)

                pid = f"r{r0}_c{c0}"
                np.savez_compressed(out / "patches" / f"{pid}.npz",
                                    image=img[None].astype(np.float16),
                                    label=lab.astype(np.float16), valid=v)
                rows.append({
                    "patch_id": pid, "row_px": r0, "col_px": c0,
                    "x_m": x_left + c0 * res, "y_m": y_top - r0 * res,
                    "valid_frac": round(n_valid / (P * P), 4),
                    "mean_label": round(float(lab[v].mean()), 4),
                    "pos_frac": round(pos_frac, 4),
                })
            if len(rows) and (r0 // S) % 20 == 0:
                print(f"  row {r0}/{H}: {len(rows)} patches so far")

    if not rows:
        raise SystemExit("No patches written. Try lowering --min-valid-frac.")
    with open(out / "index.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    pos = np.array([r["pos_frac"] for r in rows])
    print(f"\nwindows examined: {n_seen}   skipped (too little data): {n_lowvalid}   "
          f"empty patches dropped: {n_empty_dropped}")
    print(f"patches written: {len(rows)}")
    print(f"patches with >0.1% crevasse pixels (score>0.5): {(pos >= 0.001).mean() * 100:.1f}%")
    print(f"average share of valid pixels with score>0.5: {pos.mean() * 100:.2f}%")
    print(f"index: {out / 'index.csv'}")


if __name__ == "__main__":
    main()
