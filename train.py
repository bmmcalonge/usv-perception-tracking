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
import shutil
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from data import MaSTr1325, NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES
from model import build_model, IMAGENET_MEAN, IMAGENET_STD


def build_transforms(height, width, train=True, preset="default"):
    norm = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train and preset == "ewasr":
        # eWaSR's training augmentation (github.com/tersekmatija/eWaSR, datasets/transforms.py).
        # Padded borders from the rotation are ignored in the loss, and GaussNoise uses the older
        # albumentations default strength (variance 10-50 on 0-255) that eWaSR was trained with.
        return A.Compose([
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=(0, 0.3), rotate_limit=15, border_mode=cv2.BORDER_CONSTANT,
                               fill_mask=IGNORE_INDEX, p=0.7),
            A.Compose([A.ColorJitter(hue=0.05, p=0.7),
                       A.RandomGamma(gamma_limit=(70, 120), p=1.0)], p=0.5),
            A.Compose([A.GaussNoise(std_range=(10 ** 0.5 / 255, 50 ** 0.5 / 255), p=0.5),
                       A.ISONoise(p=0.5)], p=0.3),
            norm, ToTensorV2(),
        ])
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


def focal_loss(logits, target, gamma=2.0):
    """Focal loss adapted from eWaSR's wasr/loss.py (Apache-2.0, see licenses/eWaSR-LICENSE), changed to
    index labels: -(1 - p_t)^gamma * log(p_t), ignored pixels contribute 0, averaged over all pixels."""
    valid = target != IGNORE_INDEX
    t = torch.where(valid, target, torch.zeros_like(target))
    logp = torch.log_softmax(logits, 1).gather(1, t.unsqueeze(1)).squeeze(1)
    return (-((1 - logp.exp()) ** gamma) * logp * valid).mean()


def water_obstacle_separation_loss(features, target):
    """WaSR/eWaSR water-obstacle separation loss, adapted from github.com/tersekmatija/eWaSR
    wasr/loss.py (Apache-2.0, see licenses/eWaSR-LICENSE) and changed to index labels. Pushes encoder features of obstacle pixels away from
    the water feature distribution: obstacles x water variance / (obstacle-to-water-mean distance)."""
    onehot = F.one_hot(torch.where(target == IGNORE_INDEX, 3, target), 4)[..., :3]
    gt = F.interpolate(onehot.permute(0, 3, 1, 2).float(), size=features.shape[2:], mode="area")
    mask_water, mask_obstacles = gt[:, 1:2], gt[:, 0:1]
    n_water = mask_water.sum((0, 2, 3), keepdim=True).clamp(min=1.0)
    n_obstacles = mask_obstacles.sum((0, 2, 3), keepdim=True)
    if n_obstacles.squeeze() < 5 or n_water.squeeze() < 5:
        return features.new_zeros(())
    water, obstacles = mask_water * features, mask_obstacles * features
    mean_water = water.sum((0, 2, 3), keepdim=True) / n_water
    var_water = (water - mean_water * mask_water).pow(2).sum((0, 2, 3), keepdim=True) / n_water
    dist = (obstacles - mean_water * mask_obstacles).pow(2).sum((0, 2, 3), keepdim=True)
    return (n_obstacles * var_water / (dist + 0.01)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES)
    for imgs, masks in loader:
        imgs = imgs.to(device)
        pred = model(imgs).argmax(1).cpu().numpy()
        cm.update(pred.ravel(), masks.numpy().ravel())
    return cm.iou()


def make_split(img_dir, mask_dir, mask_suffix, height, width, val_frac, seed, aug="default"):
    """Two dataset views (train-aug / val-clean) over one shuffled index split.
    val_frac <= 0 trains on every image and returns no validation set."""
    train_ds = MaSTr1325(img_dir, mask_dir,
                         transform=build_transforms(height, width, True, aug),
                         mask_suffix=mask_suffix)
    if val_frac <= 0:
        return Subset(train_ds, list(range(len(train_ds)))), None
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
    ap.add_argument("--aug", choices=["default", "ewasr"], default="default",
                    help="'ewasr' = eWaSR's augmentation (flip, scale/rotate, color/gamma, noise)")
    ap.add_argument("--loss", choices=["ce", "focal_sep"], default="ce",
                    help="'focal_sep' = eWaSR's focal loss + water-obstacle separation loss")
    ap.add_argument("--sep-lambda", type=float, default=0.01,
                    help="weight of the separation loss (eWaSR default 0.01)")
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
                                    args.height, args.width, args.val_frac, args.seed, args.aug)
    print(f"[info] train={len(train_set)}  val={len(val_set) if val_set else 0}  "
          f"aug={args.aug}  loss={args.loss}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = (DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True) if val_set else None)

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
            if args.loss == "focal_sep":
                feats = model.encoder(imgs)
                logits = model.segmentation_head(model.decoder(feats))
                # separation loss on the stride-16 encoder features (ResNet layer3), as in eWaSR
                loss = (focal_loss(logits, masks)
                        + args.sep_lambda * water_obstacle_separation_loss(feats[-2], masks))
            else:
                loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        scheduler.step()
        train_loss = running / max(1, len(train_set))

        if val_loader is None:
            per_class_iou, miou = np.full(NUM_CLASSES, np.nan), float("nan")
        else:
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
        if val_loader is not None and miou > best_miou:
            best_miou = miou
            torch.save(ckpt, os.path.join(args.out, "best.pt"))
            print(f"  ** new best mIoU {best_miou:.4f} -> best.pt")

    if val_loader is None:
        shutil.copyfile(os.path.join(args.out, "last.pt"), os.path.join(args.out, "best.pt"))
        print(f"[done] no validation split: best.pt is the final epoch ({args.out}/best.pt)")
    else:
        print(f"[done] best val mIoU = {best_miou:.4f}  ({args.out}/best.pt)")


if __name__ == "__main__":
    main()
