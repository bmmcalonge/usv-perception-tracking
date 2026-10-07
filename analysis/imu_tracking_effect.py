"""
Effect of IMU motion compensation on tracking, without ground truth.

MODD2 has no track-ID annotations, so MOTA/IDF1/HOTA can't be computed. Instead,
the same per-frame detections are tracked with and without the IMU homography and
we report:
  - prediction quality: IoU between each track's predicted box (after compensation)
    and the best-overlapping detection in that frame (median and mean)
  - reported IDs and mean frames per reported ID (a fragmentation proxy)

    python analysis/imu_tracking_effect.py --ckpt runs/unet_r34/best.pt
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import load_model, segment
from detect import mask_to_boxes
import tracker as trk
from imu import load_euler_angles, load_calibration_K, unwrap_euler_sequence, delta_homography

# (sequence, detector settings, tracker settings) -- the configurations used in the report
ADJACENT = (dict(min_area=800, mode="adjacent", max_width_frac=0.3, max_aspect=5),
            dict(max_age=20, min_hits=2, iou_thr=0.2))
ENCLOSED = (dict(min_area=300, mode="enclosed", max_width_frac=0.6),
            dict(max_age=20, min_hits=3, iou_thr=0.2))
RUNS = [("kope82-00-00011177-00011797", *ADJACENT),
        ("kope81-00-00000560-00001080", *ENCLOSED),
        ("kope67-00-00062671-00063461", *ADJACENT)]


def track(ids, boxes, sort_kw, homographies):
    trk.KalmanBoxTracker.count = 0
    sort = trk.Sort(**sort_kw)
    ious = []
    associate = sort._associate

    def recording_associate(dets, preds):            # log prediction-vs-detection overlap
        if len(dets) and len(preds):
            ious.extend(trk.iou_matrix(preds, dets).max(axis=1).tolist())
        return associate(dets, preds)

    sort._associate = recording_associate
    frames_per_id = {}
    for fid, b in zip(ids, boxes):
        for row in sort.update(b, homography=homographies.get(fid)):
            frames_per_id[int(row[4])] = frames_per_id.get(int(row[4]), 0) + 1
    return np.array(ious), frames_per_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video-root", default="data/video_data")
    ap.add_argument("--cam-imu-calib", default="data/calibration_sequences/R_cam_imu.npy")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, H, W = load_model(args.ckpt, device)
    R_cam_imu = np.load(args.cam_imu_calib)

    for seq, det_kw, sort_kw in RUNS:
        S = os.path.join(args.video_root, seq)
        frames = sorted(glob.glob(os.path.join(S, "frames", "*L.jpg")))
        ids = [int(os.path.splitext(os.path.basename(p))[0].rstrip("LRlr")) for p in frames]
        boxes = [mask_to_boxes(segment(model, cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB),
                                       H, W, device), **det_kw) for p in frames]
        angles = unwrap_euler_sequence(load_euler_angles(os.path.join(S, "imu")))
        K = load_calibration_K(os.path.join(S, "calibration.yaml"))
        homs = {b: delta_homography(angles[a], angles[b], K, R_cam_imu=R_cam_imu)
                for a, b in zip(ids, ids[1:]) if a in angles and b in angles}
        print(f"\n{seq} ({det_kw['mode']} mode): {len(frames)} frames, "
              f"{sum(len(b) for b in boxes)} detections")
        for name, h in [("no IMU", {}), ("IMU compensation", homs)]:
            ious, fpi = track(ids, boxes, sort_kw, h)
            print(f"  {name:17s} pred-vs-detection IoU median {np.median(ious):.3f} "
                  f"mean {ious.mean():.3f} | reported IDs {len(fpi):3d} | "
                  f"mean frames/ID {np.mean(list(fpi.values())) if fpi else 0:6.1f}")


if __name__ == "__main__":
    main()
