"""
Visualize the detection bridge on a single image: segment -> obstacle boxes.

Runs the trained segmenter, converts the obstacle mask to bounding boxes
(exactly what run_video.py feeds the tracker), and draws them. Useful for
eyeballing segmentation quality AND tuning --min-area / the water-adjacency
filter before running the full video pipeline.

    python viz_boxes.py --ckpt runs/unet_r34/best.pt --image sample_imgs/0001.jpg
    python viz_boxes.py --ckpt runs/unet_r34/best.pt --image frame.jpg \
        --min-area 300 --out boxes_demo.jpg          # bigger min-area for hi-res frames

Add --overlay to also blend the colorized segmentation mask underneath the boxes.
"""

import argparse
import cv2
import torch

from predict import load_model, segment
from detect import mask_to_boxes
from data import colorize_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="boxes_demo.jpg")
    ap.add_argument("--min-area", type=int, default=80,
                    help="min obstacle area in px; raise for high-res frames (e.g. 300+)")
    ap.add_argument("--mode", default="adjacent",
                    choices=["all", "adjacent", "enclosed"],
                    help="'enclosed' keeps only floating obstacles (water above them), "
                         "dropping shoreline; 'adjacent' keeps water-bordering blobs; "
                         "'all' keeps everything.")
    ap.add_argument("--close-size", type=int, default=0,
                    help="morphological closing kernel size (0=off). Try 15-25 to merge "
                         "fragmented boat hulls.")
    ap.add_argument("--max-width-frac", type=float, default=None,
                    help="drop boxes wider than this fraction of image width (e.g. 0.6)")
    ap.add_argument("--max-aspect", type=float, default=None,
                    help="drop boxes with width/height ratio above this (e.g. 5 suppresses "
                         "mooring chains/ropes that are very wide and thin)")
    ap.add_argument("--overlay", action="store_true",
                    help="blend the segmentation mask under the boxes")
    ap.add_argument("--infer-height", type=int, default=None,
                    help="inference height override (default: model train height)")
    ap.add_argument("--infer-width", type=int, default=None,
                    help="inference width override (default: model train width)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, H, W = load_model(args.ckpt, device)
    IH = args.infer_height or H
    IW = args.infer_width or W
    if (IH, IW) != (H, W):
        print(f"[info] inference resolution overridden: {W}x{H} -> {IW}x{IH}")

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    mask = segment(model, rgb, IH, IW, device)
    boxes = mask_to_boxes(mask, min_area=args.min_area, mode=args.mode,
                          max_width_frac=args.max_width_frac,
                          max_aspect=args.max_aspect,
                          close_size=args.close_size)
    print(f"{len(boxes)} obstacle boxes (min_area={args.min_area}, "
          f"mode={args.mode}, max_width_frac={args.max_width_frac}, "
          f"max_aspect={args.max_aspect})")

    if args.overlay:
        bgr = cv2.addWeighted(bgr, 0.6, colorize_mask(mask), 0.4, 0)
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(args.out, bgr)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
