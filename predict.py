"""
Run the trained segmenter on a folder of images and save colorized masks
(quick visual check of segmentation quality, independent of the tracker).

    python predict.py --ckpt runs/unet_r34/best.pt \
        --img-dir some/images --out-dir preds
"""

import os
import glob
import argparse
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import build_model, IMAGENET_MEAN, IMAGENET_STD
from data import colorize_mask


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["arch"], ckpt["encoder"], ckpt["num_classes"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt.get("height", 384), ckpt.get("width", 512)


@torch.no_grad()
def segment(model, img_rgb, height, width, device, obstacle_conf=0.5):
    """Returns a full-resolution train-id mask for an RGB image.

    obstacle_conf: softmax probability threshold for the obstacle class (0=obstacle).
    Pixels predicted as obstacle but with confidence below this threshold are
    reclassified as water, suppressing uncertain detections (glint, wakes).
    Default 0.5 is equivalent to plain argmax.
    """
    h0, w0 = img_rgb.shape[:2]
    tf = A.Compose([A.Resize(height, width),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2()])
    x = tf(image=img_rgb)["image"].unsqueeze(0).to(device)
    logits = model(x)
    if obstacle_conf > 0.5:
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        obs_prob = probs[0, 0].cpu().numpy()
        pred[(pred == 0) & (obs_prob < obstacle_conf)] = 1  # reclassify uncertain obstacles as water
    else:
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out-dir", default="preds")
    ap.add_argument("--alpha", type=float, default=0.5, help="overlay opacity")
    ap.add_argument("--infer-height", type=int, default=None,
                    help="inference height (default: model train height)")
    ap.add_argument("--infer-width", type=int, default=None,
                    help="inference width (default: model train width)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, H, W = load_model(args.ckpt, device)
    IH = args.infer_height or H
    IW = args.infer_width or W
    if (IH, IW) != (H, W):
        print(f"[info] inference resolution overridden: {W}x{H} -> {IW}x{IH}")

    paths = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg"))
                   + glob.glob(os.path.join(args.img_dir, "*.png")))
    print(f"[info] {len(paths)} images on {device}")
    for p in paths:
        bgr = cv2.imread(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = segment(model, rgb, IH, IW, device)
        overlay = cv2.addWeighted(bgr, 1 - args.alpha,
                                  colorize_mask(mask), args.alpha, 0)
        out = os.path.join(args.out_dir, os.path.basename(p))
        cv2.imwrite(out, overlay)
    print(f"[done] overlays -> {args.out_dir}")


if __name__ == "__main__":
    main()
