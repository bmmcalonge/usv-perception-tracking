"""
Near-duplicate check across train.py's random train/val split of MaSTr1325.

Rebuilds the same seeded split as train.make_split(), then compares every val image
to every train image using 32x24 grayscale thumbnails (Pearson correlation). A high
count means val images have near-identical twins in train, so random-split val mIoU
overstates generalization. Thumbnail correlation is coarse: treat counts as an upper
bound and inspect the top pairs (written with --pairs-out).

    python analysis/split_leakage.py --img-dir data/MaSTr1325_images_512x384
"""

import os
import glob
import argparse
import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs-out", default=None, help="optional image of the 3 closest val/train pairs")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg"))
                   + glob.glob(os.path.join(args.img_dir, "*.png")))
    n = len(paths)
    perm = np.random.default_rng(args.seed).permutation(n)       # same split as train.make_split
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    thumbs = np.stack([cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (32, 24),
                                  interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
                       for p in paths])
    thumbs -= thumbs.mean(axis=1, keepdims=True)
    thumbs /= np.linalg.norm(thumbs, axis=1, keepdims=True) + 1e-9
    corr = thumbs[val_idx] @ thumbs[train_idx].T
    best = corr.max(axis=1)

    print(f"images {n}: train {len(train_idx)}, val {len(val_idx)} (seed {args.seed})")
    for t in (0.90, 0.95, 0.98):
        print(f"val images with a train image at thumbnail corr >= {t:.2f}: "
              f"{(best >= t).sum():3d} / {len(val_idx)}")
    top = np.argsort(-best)[:3]
    for i in top:
        j = train_idx[corr[i].argmax()]
        print(f"  closest pair: val {os.path.basename(paths[val_idx[i]])} <-> "
              f"train {os.path.basename(paths[j])}  corr {best[i]:.4f}")
    if args.pairs_out:
        rows = [np.hstack([cv2.imread(paths[val_idx[i]]), cv2.imread(paths[train_idx[corr[i].argmax()]])])
                for i in top]
        cv2.imwrite(args.pairs_out, np.vstack(rows))
        print(f"wrote {args.pairs_out}")


if __name__ == "__main__":
    main()
