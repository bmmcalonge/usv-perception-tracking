"""
Compute R_cam_imu — the 3×3 rotation from IMU frame to camera frame.

Uses MODD2 calibration sequences (Bovcon et al. 2017, Section IV):
  1. Ground-plane normal from PCA of stereo point clouds (USV on flat ground)
     → tells us how the camera's vertical axis is oriented
  2. IMU Euler angles during same flat-ground session
     → tells us how the IMU's axes are oriented relative to world

Assumptions:
  - IMU X axis = bow direction = camera optical axis (forward)
  - IMU frame is body-to-world: v_world = R_imu @ v_body
  - Camera convention: X=right, Y=down, Z=forward (OpenCV standard)

Usage:
    python calibrate_imu.py \
        --calib-dir data/calibration_sequences \
        --out data/calibration_sequences/R_cam_imu.npy

Then pass --cam-imu-calib data/calibration_sequences/R_cam_imu.npy to run_video.py.
"""

import os
import glob
import argparse
import numpy as np
from scipy.spatial.transform import Rotation


def load_pcd_xyz(path):
    with open(path, "rb") as f:
        n_points, fields, sizes = 0, [], []
        while True:
            line = f.readline().decode("ascii", errors="ignore")
            k = line.split()
            if not k:
                continue
            if k[0] == "FIELDS":
                fields = k[1:]
            elif k[0] == "SIZE":
                sizes = [int(s) for s in k[1:]]
            elif k[0] == "POINTS":
                n_points = int(k[1])
            elif k[0].startswith("DATA"):
                break
        data = np.frombuffer(f.read(n_points * sum(sizes)), dtype=np.float32)
    pts = data.reshape(n_points, len(fields))
    xi = fields.index("x")
    yi = fields.index("y")
    zi = fields.index("z")
    xyz = pts[:, [xi, yi, zi]]
    return xyz[np.isfinite(xyz).all(axis=1)]


def load_euler_angles(imu_dir):
    angles = {}
    for path in sorted(glob.glob(os.path.join(imu_dir, "*.txt"))):
        fid = int(os.path.splitext(os.path.basename(path))[0])
        angles[fid] = np.loadtxt(path)
    return angles


def ground_plane_normal(calib_dirs):
    """PCA of stereo ground-plane point clouds -> plane normal in camera frame."""
    all_pts = []
    for d in calib_dirs:
        for pcd in sorted(glob.glob(os.path.join(d, "pcl-pointcloud", "*.pcd"))):
            pts = load_pcd_xyz(pcd)
            # keep points on the water/ground surface visible to the camera
            mask = (pts[:, 1] > 100) & (pts[:, 1] < 3000) & \
                   (pts[:, 2] > 3000) & (pts[:, 2] < 25000)
            all_pts.append(pts[mask])
    pts = np.concatenate(all_pts)
    print(f"  [pcd] {len(pts)} ground points from {len(calib_dirs)} sequences")
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = Vt[-1]
    if normal[1] > 0:   # ensure it points up (-Y in camera = up)
        normal = -normal
    return normal


def mean_imu_angles(calib_dirs):
    """Average IMU Euler angles while USV was on flat ground."""
    all_vals = []
    for d in calib_dirs:
        a = load_euler_angles(os.path.join(d, "imu"))
        all_vals.extend(a.values())
    arr = np.array(all_vals)
    return arr.mean(axis=0)


def compute_R_cam_imu(n_cam, angles_cal):
    """
    Build R_cam_imu from:
      n_cam      : ground-plane normal in camera frame (= camera's 'up' direction)
      angles_cal : mean IMU Euler angles [roll, pitch, yaw] on flat ground
    """
    R_imu_cal = Rotation.from_euler("xyz", angles_cal).as_matrix()

    # IMU 'up' in IMU body frame when USV is on flat ground
    world_up = np.array([0.0, 0.0, 1.0])
    imu_up = R_imu_cal.T @ world_up

    # IMU X = bow = same direction as camera Z (optical axis, forward)
    imu_fwd = np.array([1.0, 0.0, 0.0])
    cam_fwd = np.array([0.0, 0.0, 1.0])

    # orthonormal IMU basis (fwd, up, right)
    imu_fwd /= np.linalg.norm(imu_fwd)
    imu_up  /= np.linalg.norm(imu_up)
    imu_right = np.cross(imu_fwd, imu_up)
    imu_right /= np.linalg.norm(imu_right)
    imu_up = np.cross(imu_right, imu_fwd)  # re-orthogonalize

    # corresponding camera-frame directions
    cam_up    = n_cam / np.linalg.norm(n_cam)
    cam_right = np.cross(cam_fwd, cam_up)
    cam_right /= np.linalg.norm(cam_right)
    cam_up    = np.cross(cam_right, cam_fwd)

    # R_cam_imu maps IMU basis to camera basis
    A = np.column_stack([imu_fwd, imu_up, imu_right])  # IMU columns
    B = np.column_stack([cam_fwd, cam_up, cam_right])   # camera columns
    R_raw = B @ A.T
    U, _, Vt = np.linalg.svd(R_raw)        # project onto SO(3)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", default="data/calibration_sequences",
                    help="root folder containing scal0-.../scal1-... subdirectories")
    ap.add_argument("--out", default="data/calibration_sequences/R_cam_imu.npy",
                    help="output path for R_cam_imu.npy")
    args = ap.parse_args()

    calib_dirs = sorted(glob.glob(os.path.join(args.calib_dir, "scal*")))
    if not calib_dirs:
        raise FileNotFoundError(f"No scal* directories found in {args.calib_dir}")
    print(f"Calibration sequences: {[os.path.basename(d) for d in calib_dirs]}")

    print("Computing ground-plane normal from PCD point clouds...")
    n_cam = ground_plane_normal(calib_dirs)
    print(f"  n_cam = {n_cam}  (tilt: pitch={np.degrees(np.arctan2(-n_cam[2], -n_cam[1])):.2f}°  roll={np.degrees(np.arctan2(n_cam[0], -n_cam[1])):.2f}°)")

    print("Computing mean IMU angles on flat ground...")
    angles_cal = mean_imu_angles(calib_dirs)
    print(f"  roll={angles_cal[0]:.4f}  pitch={angles_cal[1]:.4f}  yaw={angles_cal[2]:.4f} rad")

    print("Building R_cam_imu...")
    R = compute_R_cam_imu(n_cam, angles_cal)
    print(f"  R_cam_imu =\n{np.round(R, 4)}")

    np.save(args.out, R)
    print(f"Saved: {args.out}")
    print()
    print("Usage:")
    print(f"  python run_video.py ... --imu <seq>/imu --cam-imu-calib {args.out}")


if __name__ == "__main__":
    main()
