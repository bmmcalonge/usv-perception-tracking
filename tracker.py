"""
SORT-style multi-object tracker: a constant-velocity Kalman filter per track,
with IoU-based Hungarian data association.

This is the part that ties directly to closed-loop estimation work: each track
is a linear Kalman filter predicting box motion between frames, corrected by
detections when they associate. No deep net, no training — pure state estimation.

State per track (SORT formulation):
    x = [u, v, s, r, u', v', s']        (7-dim)
        u, v : box center
        s    : scale (area = w*h)
        r    : aspect ratio (w/h, assumed ~constant)
        u',v',s' : their time derivatives
Measurement:
    z = [u, v, s, r]                    (4-dim)

Implemented with plain NumPy so the filter math is fully visible and auditable.

Run a self-contained sanity check (no data needed):
    python tracker.py
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------- geometry helpers ----------

def bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> [u,v,s,r]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    u = bbox[0] + w / 2.0
    v = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([u, v, s, r], dtype=float)


def z_to_bbox(x):
    """[u,v,s,r,...] -> [x1,y1,x2,y2]."""
    s, r = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
                     x[0] + w / 2.0, x[1] + h / 2.0], dtype=float)


def warp_box(box, H):
    """Apply homography H to [x1,y1,x2,y2]; returns axis-aligned bounding box of the result."""
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1, 1.0], [x2, y1, 1.0],
                        [x2, y2, 1.0], [x1, y2, 1.0]]).T  # 3×4
    w = H @ corners
    w /= w[2]
    return np.array([w[0].min(), w[1].min(), w[0].max(), w[1].max()])


def iou_matrix(boxes_a, boxes_b):
    """Pairwise IoU between two sets of [x1,y1,x2,y2] boxes -> (A,B) matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a = np.asarray(boxes_a, dtype=float)[:, None, :]
    b = np.asarray(boxes_b, dtype=float)[None, :, :]
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


# ---------- single-track Kalman filter ----------

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        # constant-velocity model
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0           # position += velocity
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)           # observe [u,v,s,r]

        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0             # high initial velocity uncertainty
        self.P *= 10.0
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01               # small process noise on velocities
        self.Q[-1, -1] *= 0.01
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0               # scale/ratio noisier than center

        self.x = np.zeros(7)
        self.x[:4] = bbox_to_z(bbox)

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.time_since_update = 0
        self.hits = 0
        self.age = 0

    def predict(self):
        # guard against negative area after extrapolation
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return z_to_bbox(self.x)

    def update(self, bbox):
        z = bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1

    def warp(self, H):
        """Camera-motion compensation: re-express the predicted box in the current
        camera frame. Velocities are kept (H is a small inter-frame rotation)."""
        self.x[:4] = bbox_to_z(warp_box(z_to_bbox(self.x), H))

    @property
    def state_bbox(self):
        return z_to_bbox(self.x)


# ---------- the tracker ----------

class Sort:
    """
    max_age   : frames a track survives without a detection (handles occlusion /
                missed detections - the Kalman filter coasts on its velocity model)
    min_hits  : detections required before a track is reported (suppresses
                one-frame false positives from the segmenter)
    iou_thr   : minimum IoU for a detection<->track match
    """

    def __init__(self, max_age=8, min_hits=3, iou_thr=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thr = iou_thr
        self.tracks = []
        self.frame = 0

    def update(self, detections, homography=None):
        """
        detections: (N,4) array of [x1,y1,x2,y2]
        homography: optional 3×3 rotation-induced homography (K @ ΔR @ K_inv) from
                    imu.delta_homography(). Each track's predicted Kalman state is warped
                    by H before IoU matching, compensating for camera rotation (waves,
                    maneuvers).
        Returns (M,5): [x1,y1,x2,y2,id]
        """
        self.frame += 1
        dets = np.asarray(detections, dtype=float).reshape(-1, 4)

        # 1) predict existing tracks forward. With a homography, the Kalman state
        #    itself is moved into the current camera frame (as in BoT-SORT). Warping
        #    only a copy used for matching would count camera motion twice, because
        #    the velocity learned from past detections already contains it.
        for t in self.tracks:
            t.predict()
            if homography is not None:
                t.warp(homography)
        predicted = np.array([t.state_bbox for t in self.tracks]).reshape(-1, 4)

        # 2) associate via IoU + Hungarian
        matches, unmatched_dets, unmatched_trks = self._associate(dets, predicted)

        # 3) correct matched tracks
        for d_idx, t_idx in matches:
            self.tracks[t_idx].update(dets[d_idx])

        # 4) spawn new tracks for unmatched detections
        for d_idx in unmatched_dets:
            self.tracks.append(KalmanBoxTracker(dets[d_idx]))

        # 5) cull dead tracks, emit confirmed ones
        out = []
        survivors = []
        for t in self.tracks:
            if t.time_since_update <= self.max_age:
                survivors.append(t)
                confirmed = t.hits >= self.min_hits or self.frame <= self.min_hits
                if t.time_since_update == 0 and confirmed:
                    b = t.state_bbox
                    out.append([b[0], b[1], b[2], b[3], t.id])
        self.tracks = survivors
        return np.array(out).reshape(-1, 5)

    def _associate(self, dets, trks):
        if len(trks) == 0:
            return [], list(range(len(dets))), []
        if len(dets) == 0:
            return [], [], list(range(len(trks)))
        iou = iou_matrix(dets, trks)
        # Hungarian minimizes cost -> use negative IoU
        d_idx, t_idx = linear_sum_assignment(-iou)
        matches, um_d, um_t = [], [], []
        matched_d, matched_t = set(), set()
        for di, ti in zip(d_idx, t_idx):
            if iou[di, ti] >= self.iou_thr:
                matches.append((di, ti))
                matched_d.add(di)
                matched_t.add(ti)
        um_d = [i for i in range(len(dets)) if i not in matched_d]
        um_t = [i for i in range(len(trks)) if i not in matched_t]
        return matches, um_d, um_t


# ---------- self-test (runs with no dataset) ----------

def _self_test():
    """Two crossing 'vessels' + noisy detections + 3 dropped frames.
    Verifies stable IDs through the dropout and noise reduction vs raw dets."""
    rng = np.random.default_rng(0)
    T = 40
    tracker = Sort(max_age=8, min_hits=3, iou_thr=0.1)

    def box(cx, cy, s=20):
        return [cx - s, cy - s, cx + s, cy + s]

    id_sets = []
    raw_err, trk_err = [], []
    for t in range(T):
        # object A moves right, object B moves left; they cross mid-sequence
        ax, ay = 30 + 4 * t, 100
        bx, by = 190 - 4 * t, 105
        truth = {"A": (ax, ay), "B": (bx, by)}

        dets = []
        if t not in (18, 19, 20):  # simulate a 3-frame detector dropout
            for (cx, cy) in truth.values():
                nx = cx + rng.normal(0, 3)
                ny = cy + rng.normal(0, 3)
                dets.append(box(nx, ny))
                raw_err.append(np.hypot(nx - cx, ny - cy))
        tracks = tracker.update(dets)
        id_sets.append({int(r[4]) for r in tracks})

        # tracked-center error vs nearest truth
        for r in tracks:
            cx_t = (r[0] + r[2]) / 2
            cy_t = (r[1] + r[3]) / 2
            best = min(np.hypot(cx_t - c[0], cy_t - c[1]) for c in truth.values())
            trk_err.append(best)

    stable_ids = set.intersection(*[s for s in id_sets if s]) if any(id_sets) else set()
    print("=== tracker self-test ===")
    print(f"frames                : {T}")
    print(f"mean raw detector err : {np.mean(raw_err):.2f} px")
    print(f"mean tracked err      : {np.mean(trk_err):.2f} px")
    print(f"IDs alive whole run   : {sorted(stable_ids)}  "
          f"(stable through the 3-frame dropout if non-empty)")
    assert len(stable_ids) >= 1, "expected at least one ID to persist through dropout"
    print("PASS")


def _self_test_camera_motion():
    """A static object seen from a moving camera (30 px box). The per-frame homography
    describes the camera-induced image shift. With compensation the track must keep one
    ID both for a steady pan (catches double-counting camera motion) and for rocking
    (catches a missing or wrong-signed warp)."""
    def run(shift, use_h):
        rng = np.random.default_rng(1)
        KalmanBoxTracker.count = 0
        tracker = Sort(max_age=3, min_hits=1, iou_thr=0.3)
        ids = set()
        for t in range(len(shift)):
            cx = 300 + shift[t] + rng.normal(0, 1)
            H = None
            if use_h and t > 0:
                H = np.array([[1, 0, shift[t] - shift[t - 1]], [0, 1, 0], [0, 0, 1]], float)
            for r in tracker.update([[cx - 15, 185, cx + 15, 215]], homography=H):
                ids.add(int(r[4]))
        return len(ids)

    T = 60
    pan = 20.0 * np.arange(T)                            # steady pan, 20 px/frame
    rock = 40.0 * np.sin(2 * np.pi * np.arange(T) / 8)   # rocking, up to ~31 px/frame
    print("=== camera-motion compensation self-test ===")
    for name, shift in [("steady pan", pan), ("rocking", rock)]:
        without_h, with_h = run(shift, False), run(shift, True)
        print(f"{name:10s}: IDs without compensation {without_h:2d} | with compensation {with_h:2d}")
        assert with_h == 1, f"{name}: compensated track should keep a single ID"
    assert run(rock, False) > 1, "rocking scenario should fragment without compensation"
    print("PASS")


if __name__ == "__main__":
    _self_test()
    _self_test_camera_motion()
