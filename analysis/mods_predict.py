"""
Run the segmenter on the MODS benchmark and write masks for the official toolkit.

MODS (Bovcon et al., T-ITS 2021) scores what a USV needs from segmentation:
water-edge RMSE (px) on water/obstacle and water/land boundaries, and obstacle
detection TP/FP/FN/F1, overall and inside a 15 m danger zone. Evaluation runs
locally with https://github.com/bborja/mods_evaluation.

This script writes <out>/<method>/<sequence>/<frame>.png for every annotated frame
(RGB codes the toolkit expects: obstacle black, water red, sky green) plus a
toolkit config <out>/mods_config.yaml with absolute paths. Then:

    python analysis/mods_predict.py --ckpt runs/unet_r34/best.pt --mods-root data/mods
    cd data/mods/mods_evaluation
    python modb_evaluation.py unet_r34 --config-file ../predictions/mods_config.yaml --workers 8
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import load_model, segment

# train id -> RGB color expected by the MODS toolkit (configs/mods.yaml INPUT_COLORS)
MODS_COLORS = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}   # obstacle, water, sky


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mods-root", default="data/mods", help="folder with mods.json and sequences/")
    ap.add_argument("--out", default=None, help="default: <mods-root>/predictions")
    ap.add_argument("--method", default="unet_r34")
    args = ap.parse_args()

    mods_root = os.path.abspath(args.mods_root)
    out = os.path.abspath(args.out or os.path.join(mods_root, "predictions"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, H, W = load_model(args.ckpt, device)

    with open(os.path.join(mods_root, "mods.json")) as f:
        sequences = json.load(f)["dataset"]["sequences"]
    jobs = []
    for seq in sequences:
        seq_name = seq["path"].strip("/").split("/")[0]
        for frame in seq["frames"]:
            jobs.append((seq_name, frame["image_file_name"]))
    print(f"[info] {len(sequences)} sequences, {len(jobs)} annotated frames on {device}")

    for seq_name, img_name in tqdm(jobs, desc="MODS"):
        bgr = cv2.imread(os.path.join(mods_root, "sequences", seq_name, "frames", img_name))
        mask = segment(model, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), H, W, device)
        rgb = np.zeros((*mask.shape, 3), np.uint8)
        for cid, color in MODS_COLORS.items():
            rgb[mask == cid] = color
        dst = os.path.join(out, args.method, seq_name, os.path.splitext(img_name)[0] + ".png")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    cfg_path = os.path.join(out, "mods_config.yaml")
    with open(cfg_path, "w") as f:
        f.write(
            "PATHS:\n"
            f"  RESULTS: '{os.path.join(out, 'results')}'\n"
            f"  DATASET: '{mods_root}'\n"
            f"  DATASET_CALIB: '{os.path.join(mods_root, 'calibration')}'\n"
            f"  SEGMENTATIONS: '{out}'\n"
            "SEGMENTATIONS:\n"
            "  SEQ_FIRST: False\n"
            "  INPUT_COLORS:\n"
            + "".join(f"    - [{c[0]}, {c[1]}, {c[2]}]\n" for c in MODS_COLORS.values()))
    print(f"[done] masks -> {os.path.join(out, args.method)}\n[done] toolkit config -> {cfg_path}")


if __name__ == "__main__":
    main()
