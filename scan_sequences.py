"""
Scan all MODD2 sequences and rank them by how many floating obstacles the
'enclosed' detector finds -- so you can pick a good tracking-demo sequence
instead of guessing folder names.

For each sequence it samples a few left-camera frames, runs segment -> enclosed
detection, and reports the max obstacle-box count seen (and which frame). Loads
the model once, so scanning all ~28 sequences is quick on a GPU.

    python scan_sequences.py --ckpt runs/unet_r34/best.pt \
        --video-root data/video_data --samples 6 \
        --min-area 300 --max-width-frac 0.6

Sequences are printed best-first. Pick one with a steady count of 1-3 and run
run_video.py on it.
"""

import os
import glob
import argparse
import cv2
import torch

from predict import load_model, segment
from detect import mask_to_boxes


def sample_frames(frames_dir, n, pattern="*L.jpg"):
    frames = sorted(glob.glob(os.path.join(frames_dir, pattern)))
    if not frames:
        return []
    if len(frames) <= n:
        return frames
    step = len(frames) / float(n)
    return [frames[int(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video-root", default="data/video_data")
    ap.add_argument("--pattern", default="*L.jpg")
    ap.add_argument("--samples", type=int, default=6, help="frames sampled per sequence")
    ap.add_argument("--mode", default="enclosed", choices=["all", "adjacent", "enclosed"])
    ap.add_argument("--min-area", type=int, default=300)
    ap.add_argument("--max-width-frac", type=float, default=0.6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, H, W = load_model(args.ckpt, device)

    seqs = sorted(d for d in glob.glob(os.path.join(args.video_root, "*"))
                  if os.path.isdir(d))
    if not seqs:
        raise FileNotFoundError(f"No sequence folders under {args.video_root}")

    results = []
    for seq in seqs:
        frames_dir = os.path.join(seq, "frames")
        if not os.path.isdir(frames_dir):
            frames_dir = seq  # some layouts put frames directly in the seq dir
        frames = sample_frames(frames_dir, args.samples, args.pattern)
        if not frames:
            continue
        counts = []
        for fp in frames:
            bgr = cv2.imread(fp)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = segment(model, rgb, H, W, device)
            boxes = mask_to_boxes(mask, min_area=args.min_area, mode=args.mode,
                                  max_width_frac=args.max_width_frac)
            counts.append(len(boxes))
        if not counts:
            continue
        name = os.path.basename(seq)
        max_c = max(counts)
        avg_c = sum(counts) / len(counts)
        best_frame = frames[counts.index(max_c)]
        results.append((max_c, avg_c, name, os.path.basename(best_frame)))

    results.sort(reverse=True)  # best (most obstacles) first
    print(f"\n{'seq':40s} {'max':>4s} {'avg':>5s}  best-frame")
    print("-" * 72)
    for max_c, avg_c, name, bf in results:
        flag = "  <-- candidate" if 1 <= max_c <= 4 and avg_c >= 0.5 else ""
        print(f"{name:40s} {max_c:>4d} {avg_c:>5.1f}  {bf}{flag}")
    print("\nPick a '<-- candidate' sequence (steady 1-4 obstacles) for run_video.py.")


if __name__ == "__main__":
    main()
