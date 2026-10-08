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

The same script writes eWaSR's masks for a like-for-like comparison (run in eWaSR's
environment, see analysis/ewasr_model.py):

    ~/projects/ewasr/venv/bin/python analysis/mods_predict.py --model ewasr --mods-root data/mods
    ~/projects/ewasr/venv/bin/python analysis/mods_predict.py --model ewasr_imu --mods-root data/mods \
        --ewasr-weights ~/projects/ewasr/weights/ewasr_resnet18_imu.pth
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

# train id -> RGB color expected by the MODS toolkit (configs/mods.yaml INPUT_COLORS)
MODS_COLORS = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}   # obstacle, water, sky


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ours", "ewasr", "ewasr_imu"], default="ours")
    ap.add_argument("--ckpt", default="runs/unet_r34/best.pt")
    ap.add_argument("--ewasr-repo", default=os.path.expanduser("~/projects/ewasr/eWaSR"))
    ap.add_argument("--ewasr-weights", default=os.path.expanduser("~/projects/ewasr/weights/ewasr_resnet18.pth"))
    ap.add_argument("--mods-root", default="data/mods", help="folder with mods.json and sequences/")
    ap.add_argument("--out", default=None, help="default: <mods-root>/predictions")
    ap.add_argument("--method", default=None,
                    help="name of the prediction folder (default: unet_r34 / ewasr_resnet18[_imu])")
    ap.add_argument("--obstacle-conf", type=float, default=0.5,
                    help="this repo's model only: obstacle confidence threshold, as in run_video.py "
                         "(0.5 = plain argmax)")
    args = ap.parse_args()

    mods_root = os.path.abspath(args.mods_root)
    out = os.path.abspath(args.out or os.path.join(mods_root, "predictions"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "ours":
        from predict import load_model, segment
        model, H, W = load_model(args.ckpt, device)
        run = lambda rgb, imu_path: segment(model, rgb, H, W, device, obstacle_conf=args.obstacle_conf)
        args.method = args.method or ("unet_r34" if args.obstacle_conf == 0.5
                                      else f"unet_r34_conf{args.obstacle_conf:g}")
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ewasr_model import load_ewasr, segment as ewasr_segment
        variant = "ewasr_resnet18_imu" if args.model == "ewasr_imu" else "ewasr_resnet18"
        model = load_ewasr(args.ewasr_repo, args.ewasr_weights, variant, device)
        run = lambda rgb, imu_path: ewasr_segment(
            model, rgb, device,
            cv2.imread(imu_path, cv2.IMREAD_GRAYSCALE) if args.model == "ewasr_imu" else None)
        args.method = args.method or variant

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
        imu_path = os.path.join(mods_root, "sequences", seq_name, "imus", os.path.splitext(img_name)[0] + ".png")
        mask = run(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), imu_path)
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
