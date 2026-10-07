"""
Full pipeline on a video / image sequence (e.g. a MODS sequence):
    segment -> obstacle boxes -> Kalman/SORT tracking -> annotated video.

    python run_video.py --ckpt runs/unet_r34/best.pt \
        --frames data/MODS/seq01/frames \
        --out out_seq01.mp4 --fps 15

For MODD2 (stereo frames interleaved L/R in one folder), point --frames at the
sequence's frames/ folder and use --pattern to take a single camera:

    python run_video.py --ckpt runs/unet_r34/best.pt \
        --frames data/video_data/kope67-00-00040950-00041190/frames \
        --pattern '*L.jpg' --min-area 300 --out out_kope67.mp4

If --gt (MOT-format ground truth) is supplied AND `motmetrics` is installed,
also prints MOTA / IDF1.  Always prints throughput (FPS) and track counts.
"""

import os
import glob
import time
import argparse
import cv2
import numpy as np
import torch

from predict import load_model, segment
from detect import mask_to_boxes
from tracker import Sort

# distinct BGR colors for track ids
_PALETTE = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
            (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255)]


def color_for(tid):
    return _PALETTE[int(tid) % len(_PALETTE)]


def list_frames(frames_dir, pattern=None):
    """Sorted frame paths. If pattern is given (e.g. '*L.jpg'), use it verbatim;
    otherwise default to all .jpg/.png in the folder."""
    if pattern:
        return sorted(glob.glob(os.path.join(frames_dir, pattern)))
    return sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))
                  + glob.glob(os.path.join(frames_dir, "*.png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True, help="folder of sequential frames")
    ap.add_argument("--pattern", default=None,
                    help="glob to select frames in --frames, e.g. '*L.jpg' for MODD2 "
                         "left-camera only. Default: all .jpg/.png.")
    ap.add_argument("--out", default="tracked.mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--min-area", type=int, default=80)
    ap.add_argument("--mode", default="adjacent",
                    choices=["all", "adjacent", "enclosed"],
                    help="'enclosed' tracks only floating obstacles (water above them), "
                         "dropping shoreline; 'adjacent' keeps water-bordering blobs; "
                         "'all' keeps everything.")
    ap.add_argument("--close-size", type=int, default=0,
                    help="morphological closing kernel size before component labeling "
                         "(0=off). Try 15-25 to merge fragmented boat hulls into one box.")
    ap.add_argument("--max-width-frac", type=float, default=None,
                    help="drop boxes wider than this fraction of image width (e.g. 0.6)")
    ap.add_argument("--max-aspect", type=float, default=None,
                    help="drop boxes with width/height ratio above this (e.g. 5 suppresses "
                         "mooring chains/ropes that produce thin horizontal strips)")
    ap.add_argument("--max-age", type=int, default=8)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--obstacle-conf", type=float, default=0.5,
                    help="softmax confidence threshold for obstacle class (default 0.5 = argmax). "
                         "Raise to 0.7-0.9 to suppress uncertain detections like glint/wakes.")
    ap.add_argument("--infer-height", type=int, default=None,
                    help="inference height override (default: model train height). "
                         "Higher values improve detection of distant/small obstacles.")
    ap.add_argument("--infer-width", type=int, default=None,
                    help="inference width override (default: model train width).")
    ap.add_argument("--no-tracker", action="store_true",
                    help="skip Kalman/SORT tracker; draw raw per-frame detector boxes "
                         "with no IDs and no temporal coasting (useful for comparison).")
    ap.add_argument("--imu", default=None,
                    help="path to MODD2 sequence imu/ folder. Enables IMU-based camera "
                         "motion compensation: predicted track positions are rotated by "
                         "the between-frame IMU delta before IoU matching, reducing ID "
                         "swaps during waves or maneuvers. calibration.yaml must be in "
                         "the parent folder of --imu.")
    ap.add_argument("--cam-imu-calib", default=None,
                    help="path to R_cam_imu.npy (3x3 rotation from IMU frame to camera "
                         "frame, computed by calibrate_imu.py). Required for correct "
                         "axis mapping — without it --imu has no effect.")
    ap.add_argument("--gt", default=None, help="optional MOT-format gt.txt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, H, W = load_model(args.ckpt, device)

    imu_angles, K_cam, R_cam_imu = None, None, None
    if args.imu:
        from imu import (load_euler_angles, load_calibration_K,
                         delta_homography, unwrap_euler_sequence)
        imu_angles = load_euler_angles(args.imu)
        imu_angles = unwrap_euler_sequence(imu_angles)
        calib = os.path.join(os.path.dirname(args.imu.rstrip("/\\")), "calibration.yaml")
        K_cam = load_calibration_K(calib)
        R_cam_imu = np.load(args.cam_imu_calib) if args.cam_imu_calib else None
        status = f"  R_cam_imu: {args.cam_imu_calib}" if R_cam_imu is not None else "  (no --cam-imu-calib: compensation inactive)"
        print(f"[imu] loaded {len(imu_angles)} frames  calib: {calib}{status}")
    angles_prev = None
    IH = args.infer_height or H
    IW = args.infer_width or W
    if (IH, IW) != (H, W):
        print(f"[info] inference resolution overridden: {W}x{H} -> {IW}x{IH}")
    tracker = None if args.no_tracker else Sort(max_age=args.max_age, min_hits=args.min_hits, iou_thr=args.iou)

    frames = list_frames(args.frames, args.pattern)
    if not frames:
        raise FileNotFoundError(
            f"No frames in {args.frames}" +
            (f" matching pattern '{args.pattern}'" if args.pattern else "") +
            ". For MODD2, point --frames at a sequence's frames/ folder and pass "
            "--pattern '*L.jpg'.")
    h0, w0 = cv2.imread(frames[0]).shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w0, h0))

    acc = None
    if args.gt:
        try:
            import motmetrics as mm
            acc = mm.MOTAccumulator(auto_id=True)
            gt = _load_mot_gt(args.gt)
        except ImportError:
            print("[warn] motmetrics not installed -> skipping MOTA/IDF1")
            args.gt = None

    n_frames, total_tracks, t0 = 0, 0, time.time()
    for fi, fp in enumerate(frames):
        bgr = cv2.imread(fp)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = segment(model, rgb, IH, IW, device, obstacle_conf=args.obstacle_conf)
        boxes = mask_to_boxes(mask, min_area=args.min_area, mode=args.mode,
                              max_width_frac=args.max_width_frac,
                              max_aspect=args.max_aspect,
                              close_size=args.close_size)

        H_imu = None
        if imu_angles is not None:
            basename = os.path.splitext(os.path.basename(fp))[0].rstrip("LRlr")
            frame_id = int(basename)
            if frame_id in imu_angles:
                angles_curr = imu_angles[frame_id]
                if angles_prev is not None:
                    H_imu = delta_homography(angles_prev, angles_curr, K_cam, R_cam_imu=R_cam_imu)
                angles_prev = angles_curr

        if tracker is not None:
            active = tracker.update(boxes, homography=H_imu)
            for x1, y1, x2, y2, tid in active:
                c = color_for(tid)
                cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
                cv2.putText(bgr, f"ID {int(tid)}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        else:
            active = boxes
            for x1, y1, x2, y2 in boxes.astype(int):
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        total_tracks += len(active)
        n_frames += 1
        cv2.putText(bgr, f"frame {fi}  boxes {len(active)}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(bgr)

        if acc is not None and fi in gt:
            _accumulate(acc, gt[fi], active)

    writer.release()
    dt = time.time() - t0
    fps = n_frames / dt if dt > 0 else 0.0
    print(f"[done] {n_frames} frames -> {args.out}")
    print(f"[perf] {fps:.1f} FPS end-to-end on {device} "
          f"(seg+detect+track)   avg tracks/frame={total_tracks/max(1,n_frames):.2f}")

    if acc is not None:
        import motmetrics as mm
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=["mota", "idf1", "num_switches"],
                             name="seq")
        print(summary)


# ---- optional MOT-format ground-truth helpers (frame,id,x,y,w,h,...) ----

def _load_mot_gt(path):
    gt = {}
    for line in open(path):
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        f, i, x, y, w, h = (int(float(p[0])), int(float(p[1])),
                            float(p[2]), float(p[3]), float(p[4]), float(p[5]))
        gt.setdefault(f - 1, []).append((i, [x, y, x + w, y + h]))  # 1-indexed -> 0
    return gt


def _accumulate(acc, gt_objs, tracks):
    import motmetrics as mm
    gids = [g[0] for g in gt_objs]
    gboxes = [g[1] for g in gt_objs]
    tids = [int(t[4]) for t in tracks]
    tboxes = [t[:4] for t in tracks]
    if gboxes and tboxes:
        from tracker import iou_matrix
        dist = 1.0 - iou_matrix(gboxes, tboxes)
        dist[dist > 0.5] = np.nan  # gate at IoU 0.5
    else:
        dist = np.empty((len(gboxes), len(tboxes)))
    acc.update(gids, tids, dist)


if __name__ == "__main__":
    main()
