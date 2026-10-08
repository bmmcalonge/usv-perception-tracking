"""
Throughput benchmark: this repo's U-Net/ResNet-34 vs eWaSR, under identical conditions.

Two measurements, same GPU, fp32, batch 1, 512x384 network input:

  1. Segmentation only: the network's forward pass on one preprocessed frame, timed with
     CUDA events after 10 warm-up passes; median over --iters runs. Same protocol as
     eWaSR's own tools/benchmark.py.
  2. Full pipeline: for every frame of a MODD2 sequence, JPEG decode -> segment (resize,
     normalize, network, argmax, resize back) -> obstacle boxes -> Kalman/SORT update.
     No drawing or video encoding. Median FPS over --repeats passes, after a warm-up pass
     on the first 30 frames. Per-stage times are reported from the median pass.

The two models need different environments (eWaSR pins timm==0.6.5), so run once in each,
with the same torch build:

    .venv/bin/python analysis/benchmark_speed.py --model ours --ckpt runs/unet_r34/best.pt
    ~/projects/ewasr/venv/bin/python analysis/benchmark_speed.py --model ewasr \
        --ewasr-repo ~/projects/ewasr/eWaSR --ewasr-weights ~/projects/ewasr/weights/ewasr_resnet18.pth
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect import mask_to_boxes
from tracker import Sort

DEFAULT_SEQ = "data/video_data/kope81-00-00000560-00001080/frames"
# detector/tracker settings of the documented kope81 run (README, "Run")
DETECT_KW = dict(min_area=300, mode="enclosed", max_width_frac=0.6)
SORT_KW = dict(max_age=20, min_hits=3, iou_thr=0.2)


def build(args, device):
    """Return (name, network_forward(x), segment(rgb) -> mask, n_params, example_input)."""
    if args.model == "ours":
        from predict import load_model, segment
        from model import IMAGENET_MEAN, IMAGENET_STD
        net, H, W = load_model(args.ckpt, device)
        name = "U-Net / ResNet-34 (this repo)"
        forward = lambda x: net(x)
        seg = lambda rgb: segment(net, rgb, H, W, device)
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ewasr_model import load_ewasr, segment, IMAGENET_MEAN, IMAGENET_STD, HEIGHT as H, WIDTH as W
        net = load_ewasr(args.ewasr_repo, args.ewasr_weights, "ewasr_resnet18", device)
        name = "eWaSR ResNet-18 (camera only)"
        imu = torch.zeros((1, H, W), dtype=torch.bool, device=device)
        forward = lambda x: net({"image": x, "imu_mask": imu})["out"]
        seg = lambda rgb: segment(net, rgb, device)
    n_params = sum(p.numel() for p in net.parameters())
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    tf = A.Compose([A.Resize(H, W), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
    return name, forward, seg, n_params, tf, (H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ours", "ewasr"], required=True)
    ap.add_argument("--ckpt", default="runs/unet_r34/best.pt")
    ap.add_argument("--ewasr-repo", default=os.path.expanduser("~/projects/ewasr/eWaSR"))
    ap.add_argument("--ewasr-weights", default=os.path.expanduser("~/projects/ewasr/weights/ewasr_resnet18.pth"))
    ap.add_argument("--frames", default=DEFAULT_SEQ)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cudnn.benchmark = False                       # same default for both models
    name, forward, seg, n_params, tf, (H, W) = build(args, device)
    frames = sorted(glob.glob(os.path.join(args.frames, "*L.jpg")))

    print(f"model   : {name}  ({n_params / 1e6:.1f} M parameters)")
    print(f"system  : {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuDNN "
          f"{torch.backends.cudnn.version()} | fp32, batch 1, input {W}x{H}")

    # 1) segmentation only: network forward pass, CUDA events, median of --iters
    rgb0 = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)
    x = tf(image=rgb0)["image"].unsqueeze(0).to(device)
    lat = np.zeros(args.iters)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(10):
            forward(x)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for i in range(args.iters):
            start.record(); forward(x); end.record()
            torch.cuda.synchronize()
            lat[i] = start.elapsed_time(end)
    mem = torch.cuda.max_memory_allocated() / 2**20
    print(f"\n[segmentation only] median {np.median(lat):.2f} ms  (p95 {np.percentile(lat, 95):.2f} ms)"
          f"  -> {1000 / np.median(lat):.0f} FPS | peak GPU memory {mem:.0f} MiB")

    # 2) full pipeline on the sequence: decode -> segment -> boxes -> tracker
    def run(paths):
        tracker = Sort(**SORT_KW)
        t = np.zeros(3)
        t0 = time.perf_counter()
        for p in paths:
            a = time.perf_counter()
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            b = time.perf_counter()
            mask = seg(rgb)                                      # .cpu() inside syncs the GPU
            c = time.perf_counter()
            tracker.update(mask_to_boxes(mask, **DETECT_KW))
            d = time.perf_counter()
            t += (b - a, c - b, d - c)
        return len(paths) / (time.perf_counter() - t0), t / len(paths) * 1000

    run(frames[:30])                                             # warm-up
    passes = sorted((run(frames) for _ in range(args.repeats)), key=lambda r: r[0])
    fps, stage_ms = passes[len(passes) // 2]
    print(f"[full pipeline]     {fps:.1f} FPS on {len(frames)} frames ({os.path.basename(os.path.dirname(args.frames))}),"
          f" median of {args.repeats} passes")
    print(f"                    per frame: decode {stage_ms[0]:.2f} ms | segmentation {stage_ms[1]:.2f} ms"
          f" | boxes + tracking {stage_ms[2]:.2f} ms")


if __name__ == "__main__":
    main()
