"""
IMU utilities for MODD2 sequences.

Per-frame .txt files each contain 3 Euler angles (radians) on separate lines:
    roll   — rotation around X
    pitch  — rotation around Y
    yaw    — rotation around Z (heading)
These are the IMU's absolute orientation relative to the water surface.
(Bovcon et al. 2017, Section III-A: "readout of Euler angles from the IMU")

Usage in run_video.py:
    angles = load_euler_angles("path/to/sequence/imu/")
    angles = unwrap_euler_sequence(angles)
    K = load_calibration_K("path/to/sequence/calibration.yaml")
    H = delta_homography(angles[frame_id_t_minus_1], angles[frame_id_t], K)
    active = tracker.update(boxes, homography=H)  # H may be None (skipped frame)
"""

import os
import glob
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


def load_euler_angles(imu_dir):
    """
    Load per-frame Euler angles from a MODD2 imu/ directory.
    Returns dict: int(frame_id) -> np.array([roll, pitch, yaw]) in radians.
    """
    angles = {}
    for path in sorted(glob.glob(os.path.join(imu_dir, "*.txt"))):
        frame_id = int(os.path.splitext(os.path.basename(path))[0])
        angles[frame_id] = np.loadtxt(path)
    return angles


def load_calibration_K(calibration_yaml):
    """Parse left-camera intrinsic matrix K from a MODD2 calibration.yaml (OpenCV format)."""
    fs = cv2.FileStorage(calibration_yaml, cv2.FILE_STORAGE_READ)
    K = fs.getNode("M1").mat()
    fs.release()
    if K is None or K.shape != (3, 3):
        raise ValueError(f"Could not parse M1 from {calibration_yaml}")
    return K


def unwrap_euler_sequence(angles_dict):
    """
    Remove 2π wrapping discontinuities from a sequence of Euler angles.
    angles_dict: {frame_id: np.array([roll, pitch, yaw])}
    Returns new dict with unwrapped angles (same keys, continuous values).
    """
    ids = sorted(angles_dict.keys())
    arr = np.array([angles_dict[i] for i in ids])  # N×3
    arr = np.unwrap(arr, axis=0)                    # unwrap each channel independently
    return {fid: arr[k] for k, fid in enumerate(ids)}


def euler_to_R(angles):
    """[roll, pitch, yaw] in radians (XYZ convention) -> 3×3 rotation matrix."""
    return Rotation.from_euler('xyz', angles).as_matrix()


def delta_homography(angles_prev, angles_curr, K,
                     R_cam_imu=None, max_angle=0.05):
    """
    IMU-based rotation homography from frame t-1 to frame t.

    angles_prev, angles_curr: np.array([roll, pitch, yaw]) in radians
    K: 3×3 camera intrinsic matrix (left camera M1 from calibration.yaml)
    R_cam_imu: 3×3 rotation from IMU frame to camera frame — computed by
               calibrate_imu.py from the MODD2 calibration sequences (PCD
               ground-plane PCA + IMU reading on flat ground, per Bovcon 2017 §IV).
               Load with: R_cam_imu = np.load('data/calibration_sequences/R_cam_imu.npy')
               Returns None if not provided.
    max_angle: skip frames where the estimated camera rotation exceeds this (radians).

    Returns None if R_cam_imu is not provided or if rotation exceeds max_angle.
    """
    if R_cam_imu is None:
        return None

    # R_imu is body-to-world (v_world = R_imu @ v_body).
    # Camera world-to-cam: R_cam_world = R_cam_imu @ R_imu.T
    # Forward homography (t-1 → t):
    #   dR_cam = R_cam_world_t @ R_cam_world_{t-1}.T
    #          = R_cam_imu @ R_curr.T @ R_prev @ R_cam_imu.T
    R_prev = Rotation.from_euler('xyz', angles_prev).as_matrix()
    R_curr = Rotation.from_euler('xyz', angles_curr).as_matrix()
    dR_cam = R_cam_imu @ (R_curr.T @ R_prev) @ R_cam_imu.T

    angle = np.arccos(np.clip((np.trace(dR_cam) - 1.0) / 2.0, -1.0, 1.0))
    if angle > max_angle:
        return None
    return K @ dR_cam @ np.linalg.inv(K)
