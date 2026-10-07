"""
Cross-check the IMU against camera rotation measured from the images.

For each MODD2 sequence:
  vision : frame-to-frame camera rotation from LK-tracked far-field features
           (water masked out with the segmenter, bottom of frame dropped), solved
           on bearing vectors with RANSAC + Kabsch. Pure rotation is a good model
           for distant scenery.
  IMU    : frame-to-frame body rotation D = R_curr^T R_prev (imu.py convention).
  fit    : per-sequence camera<-IMU rotation R = argmin sum |w_cam - R w_imu|^2,
           i.e. a rotation-only camera-IMU calibration from that sequence alone.

Reported per IMU time offset (frames):
  - median image-motion residual in px: no compensation / calibrate_imu.py's
    R_cam_imu / the per-sequence fit
  - correlation between vision and calibrated-IMU rotation about each camera axis
    (x = pitch, y = pan, z = roll)
  - angle between the per-sequence fit and R_cam_imu

If the IMU is trustworthy, the fit lands close to R_cam_imu and compensation cuts
the residual; if no rotation explains the IMU, its readings don't follow the camera.

    python analysis/imu_vs_vision.py --ckpt runs/unet_r34/best.pt
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import load_model, segment
from imu import load_euler_angles, load_calibration_K, unwrap_euler_sequence

WATER = 1
DEFAULT_SEQS = ["kope67-00-00062671-00063461", "kope67-00-00025200-00025670",
                "kope71-01-00011520-00011800", "kope75-00-00021500-00022160",
                "kope81-00-00000560-00001080", "kope82-00-00011177-00011797"]


def kabsch(A, B):
    """Rotation R minimising sum |B_i - R A_i|^2 (A, B: N x 3)."""
    U, _, Vt = np.linalg.svd(B.T @ A)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def bearings(pts, K_inv):
    b = (K_inv @ np.c_[pts, np.ones(len(pts))].T).T
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def vision_rotation(p0, p1, K_inv, thr_rad, rng, iters=200):
    """Rotation mapping bearings of p0 onto p1, RANSAC over 2-point Kabsch fits."""
    b0, b1 = bearings(p0, K_inv), bearings(p1, K_inv)
    best = None
    for _ in range(iters):
        i = rng.choice(len(b0), 2, replace=False)
        R = kabsch(b0[i], b1[i])
        err = np.arccos(np.clip(np.sum((R @ b0.T).T * b1, axis=1), -1, 1))
        inl = err < thr_rad
        if best is None or inl.sum() > best.sum():
            best = inl
    if best.sum() < 30:
        return None
    return kabsch(b0[best], b1[best])


def camera_rotations(frames, model, H, W, device, K, rng):
    """Frame-to-frame camera rotation vectors from far-field features."""
    K_inv, f = np.linalg.inv(K), K[0, 0]
    out = {}
    prev = cv2.imread(frames[0])
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_mask = segment(model, cv2.cvtColor(prev, cv2.COLOR_BGR2RGB), H, W, device)
    h = prev_g.shape[0]
    for k in range(1, len(frames)):
        cur = cv2.imread(frames[k])
        cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        water = cv2.dilate((prev_mask == WATER).astype(np.uint8), np.ones((15, 15), np.uint8))
        far = (water == 0).astype(np.uint8) * 255
        far[int(0.85 * h):] = 0                          # own-ship parts at the bottom
        p0 = cv2.goodFeaturesToTrack(prev_g, 600, 0.005, 8, mask=far)
        if p0 is not None and len(p0) >= 30:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, cur_g, p0, None)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(cur_g, prev_g, p1, None)
            fb = np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1)
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5)   # forward-backward check
            if good.sum() >= 30:
                R = vision_rotation(p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good],
                                    K_inv, 1.0 / f, rng)
                if R is not None:
                    out[k] = Rotation.from_matrix(R).as_rotvec()
        prev_g = cur_g
        prev_mask = segment(model, cv2.cvtColor(cur, cv2.COLOR_BGR2RGB), H, W, device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video-root", default="data/video_data")
    ap.add_argument("--cam-imu-calib", default="data/calibration_sequences/R_cam_imu.npy")
    ap.add_argument("--seqs", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument("--lags", nargs="*", type=int, default=[-1, 0, 1],
                    help="IMU time offsets to test, in frames")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, H, W = load_model(args.ckpt, device)
    R_cam_imu = np.load(args.cam_imu_calib)
    rng = np.random.default_rng(0)

    for seq in args.seqs:
        S = os.path.join(args.video_root, seq)
        frames = sorted(glob.glob(os.path.join(S, "frames", "*L.jpg")))
        ids = [int(os.path.splitext(os.path.basename(p))[0].rstrip("LRlr")) for p in frames]
        angles = unwrap_euler_sequence(load_euler_angles(os.path.join(S, "imu")))
        K = load_calibration_K(os.path.join(S, "calibration.yaml"))
        f = K[0, 0]
        w_vis = camera_rotations(frames, model, H, W, device, K, rng)
        med = np.degrees(np.median([np.linalg.norm(v) for v in w_vis.values()]))
        print(f"\n{seq}: {len(w_vis)}/{len(frames) - 1} frame pairs, "
              f"median camera rotation {med:.3f} deg/frame")
        for lag in args.lags:
            wi, wc = [], []
            for k, wv in w_vis.items():
                a, b = ids[k - 1] + lag, ids[k] + lag
                if a in angles and b in angles:
                    D = (Rotation.from_euler("xyz", angles[b]).as_matrix().T
                         @ Rotation.from_euler("xyz", angles[a]).as_matrix())
                    wi.append(Rotation.from_matrix(D).as_rotvec())
                    wc.append(wv)
            wi, wc = np.array(wi), np.array(wc)
            R_fit = kabsch(wi, wc)
            pred = (R_cam_imu @ wi.T).T
            res = [np.median(np.linalg.norm(wc - p, axis=1)) * f
                   for p in (np.zeros_like(wc), pred, (R_fit @ wi.T).T)]
            corr = [np.corrcoef(wc[:, j], pred[:, j])[0, 1] for j in range(3)]
            gap = np.degrees(Rotation.from_matrix(R_fit @ R_cam_imu.T).magnitude())
            print(f"  IMU offset {lag:+d}: residual px none {res[0]:5.2f} | calibrated {res[1]:5.2f} | "
                  f"per-seq fit {res[2]:5.2f} || corr pitch {corr[0]:+.2f} pan {corr[1]:+.2f} "
                  f"roll {corr[2]:+.2f} || fit vs calibration {gap:5.1f} deg")


if __name__ == "__main__":
    main()
