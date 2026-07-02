"""
Train a maritime semantic segmentation model on MaSTr1325.

Example:
    python train.py \
        --img-dir  data/MaSTr1325_images_512x384 \
        --mask-dir data/MaSTr1325_masks_512x384 \
        --arch unet --encoder resnet34 \
        --epochs 40 --batch-size 8 --lr 3e-4 \
        --out runs/unet_r34

Outputs:
    runs/.../best.pt        best checkpoint (by val mIoU)
    runs/.../last.pt        final checkpoint
    runs/.../metrics.csv    per-epoch loss + mIoU
"""

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from data import MaSTr1325, NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES
from model import build_model, IMAGENET_MEAN, IMAGENET_STD


def build_transforms(height, width, train=True):
    norm = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        return A.Compose([
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15,
                                 val_shift_limit=10, p=0.2),
            A.GaussNoise(p=0.15),
            norm, ToTensorV2(),
        ])
    return A.Compose([A.Resize(height, width), norm, ToTensorV2()])


class ConfusionMatrix:
    """Streaming confusion matrix -> per-class IoU and mIoU (ignores IGNORE_INDEX)."""

    def __init__(self, num_classes):
        self.n = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        k = (target >= 0) & (target < self.n)  # drops IGNORE_INDEX (255)
        idx = self.n * target[k].astype(np.int64) + pred[k].astype(np.int64)
        self.mat += np.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self):
        m = self.mat.astype(np.float64)
        inter = np.diag(m)
        union = m.sum(1) + m.sum(0) - inter
        iou = inter / np.maximum(union, 1e-9)
        return iou, float(np.nanmean(iou))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES)
    for imgs, masks in loader:
        imgs = imgs.to(device)
        pred = model(imgs).argmax(1).cpu().numpy()
        cm.update(pred.ravel(), masks.numpy().ravel())
    return cm.iou()


def make_split(img_dir, mask_dir, mask_suffix, height, width, val_frac, seed):
    """Two dataset views (train-aug / val-clean) over one shuffled index split."""
    train_ds = MaSTr1325(img_dir, mask_dir,
                         transform=build_transforms(height, width, True),
                         mask_suffix=mask_suffix)
    val_ds = MaSTr1325(img_dir, mask_dir,
                       transform=build_transforms(height, width, False),
                       mask_suffix=mask_suffix)
    n = len(train_ds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    return Subset(train_ds, train_idx), Subset(val_ds, val_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--mask-suffix", default="m")
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")
    if device == "cpu":
        print("[warn] no CUDA detected - training will be slow. "
              "This project is meant to run on your GPU machine.")

    train_set, val_set = make_split(args.img_dir, args.mask_dir, args.mask_suffix,
                                    args.height, args.width, args.val_frac, args.seed)
    print(f"[info] train={len(train_set)}  val={len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model(args.arch, args.encoder, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_miou"] +
                               [f"iou_{c}" for c in CLASS_NAMES])

    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        scheduler.step()
        train_loss = running / max(1, len(train_set))

        per_class_iou, miou = evaluate(model, val_loader, device)
        print(f"  loss={train_loss:.4f}  mIoU={miou:.4f}  "
              + "  ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, per_class_iou)))

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{miou:.4f}"]
                                   + [f"{v:.4f}" for v in per_class_iou])

        ckpt = {"model": model.state_dict(), "arch": args.arch,
                "encoder": args.encoder, "num_classes": NUM_CLASSES,
                "height": args.height, "width": args.width}
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if miou > best_miou:
            best_miou = miou
            torch.save(ckpt, os.path.join(args.out, "best.pt"))
            print(f"  ** new best mIoU {best_miou:.4f} -> best.pt")

    print(f"[done] best val mIoU = {best_miou:.4f}  ({args.out}/best.pt)")


if __name__ == "__main__":
    main()
