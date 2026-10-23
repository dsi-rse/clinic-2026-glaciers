# Surawy-Stepney crevasse mapping: small-scale reproduction

An early experiment following Surawy-Stepney et al. (2023), "Mapping Antarctic crevasses and their evolution with deep learning applied to satellite radar imagery," *The Cryosphere* 17, 4421–4445, https://doi.org/10.5194/tc-17-4421-2023.

The goal was to learn how the method works before our NISAR data is available. A small U-Net is trained to predict the authors' released fracture map from the released radar backscatter image.

## What this does

1. `make_patches.py` cuts the two continent-sized GeoTIFFs into matching 256×256 patches (radar image plus the authors' fracture score for the same pixels) and records each patch's map position in `index.csv`.
2. `train_unet.py` trains the paper's shallow U-Net on those patches from random weights, with a train/validation split by 500 km geographic block.
3. `plot_predictions.py` saves side-by-side pictures (input, authors' map, prediction) for validation patches, plus a few numbers on how well the prediction separates crevasse pixels from background.

## Data

- `202106_backscatter_map.tiff`: Sentinel-1 backscatter in dB, 100 m pixels (about 45,000 × 56,000), a composite for June 2021.
- `202106_fracture_map.tiff`: the authors' fracture score from 0 to 1 on the same grid. It is narrower than the backscatter image, so the scripts crop to the shared area.
- Source: Surawy-Stepney, T. and Hogg, A. E., "Data accompanying the article 'Mapping Antarctic Crevasses and their Evolution with Deep Learning Applied to Satellite Radar Imagery'," Zenodo, https://doi.org/10.5281/zenodo.8296992 (Creative Commons Attribution license). The record describes the fracture and backscatter maps as June 2021 at 100 m, matching the file names. (The paper's data-availability statement says June 2022, which appears to be a typo.) The data files are not in this repository because of their size.

## Run order

```bash
pip install numpy rasterio torch matplotlib

python make_patches.py \
    --backscatter data/202106_backscatter_map.tiff \
    --fracture    data/202106_fracture_map.tiff \
    --out-dir patches_2021

python train_unet.py --patch-dir patches_2021 --out-dir run1 \
    --epochs 4 --batch-size 4 --pos-weight 1

python plot_predictions.py --run-dir run1 --patch-dir patches_2021
```

`train_unet.py` needs the paper's `unet.py` in the same folder (or `--unet-dir`). It comes from the paper's Zenodo code archive, https://doi.org/10.5281/zenodo.8410599, in the `pub_modules` folder. The only change here was removing two unused imports (`torchvision` and `random`) so that torchvision isn't required.

## What I found

- On CPU, one epoch took roughly 20 to 35 minutes with batch size 4.
- With `--pos-weight 3`, the loss stayed flat over 3 epochs and the mean error rose.
- With `--pos-weight 1`, validation loss fell every epoch over 4 epochs (0.253 to 0.230) and mean error fell (0.117 to 0.086). Recall at the 0.5 threshold stayed at zero, because no predicted probability reached 0.5.
- In the prediction plots (8 validation patches, judged by eye), the network traced some visible cracks and stayed near zero on crevasse-free patches. It also responded to non-crevasse features such as sea ice and bright blocks, and missed fine crevasse fields.

## Limitations

- The labels are the authors' model output, not hand-drawn ground truth. A good score would mean "imitates their pipeline," not "finds real crevasses."
- This does not use the paper's bootstrapping, its separate type-A and type-B networks, or 50 m individual acquisitions. It uses one network on a 100 m composite.
- Training was only a few short epochs from random weights, so these results are preliminary.
- Everything is one month of one sensor. Whether it generalizes to other months is untested; the public record only has June 2021, so that check would need more data from the authors.
- Next steps: a threshold-free score (such as ROC AUC) and a longer run.
