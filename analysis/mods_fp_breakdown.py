"""
Where do a method's MODS false positives come from? Compare methods on the same axes.

Reads the toolkit's per-frame results (results_<method>.json) and, for every false-positive
blob, records:
  - position: distance of the blob's top edge below the IMU horizon (MODS 'imus/' mask),
    and whether it touches the bottom image border (water right in front of the camera)
  - size class (blob area in pixels)
  - sequence
Only frames the toolkit scores for false positives (exhaustively annotated) contribute.

    python analysis/mods_fp_breakdown.py --methods unet_r34 ewasr_resnet18 \
        --sheet-method unet_r34 --sheet-out fp_examples.jpg
"""

import os
import json
import argparse
from collections import Counter, defaultdict
import numpy as np
import cv2

POSITION_BINS = [("touching horizon (0-20 px below)", 0, 20),
                 ("near horizon (20-100 px below)", 20, 100),
                 ("open water (>100 px below)", 100, 10**6)]
SIZE_BINS = [("25-225 px", 0, 225), ("225-900 px", 225, 900),
             ("900-2500 px", 900, 2500), (">2500 px", 2500, 10**9)]


def horizon_row(mods_root, seq, hor_name, x, cache):
    key = (seq, hor_name)
    if key not in cache:
        m = cv2.imread(os.path.join(mods_root, "sequences", seq, "imus", hor_name), cv2.IMREAD_GRAYSCALE)
        cache.clear()
        cache[key] = m
    m = cache[key]
    col = m[:, int(np.clip(x, 0, m.shape[1] - 1))]
    return int(np.argmax(col > 0)) if col.any() else m.shape[0]


def collect(mods_root, method, seq_names):
    with open(os.path.join(mods_root, "predictions", "results", f"results_{method}.json")) as f:
        res = json.load(f)
    fps, cache = [], {}
    for s in res["sequences"]:
        if not s.get("evaluated"):
            continue
        seq = seq_names[s["id"] - 1]
        for fr in s["frames"]:
            danger = {tuple(d["bbox"]) for d in fr["obstacles_danger"]["fp_list"]}
            for fp in fr["obstacles"]["fp_list"]:
                x1, y1, x2, y2 = fp["bbox"]
                h = horizon_row(mods_root, seq, fr["hor_name"], (x1 + x2) / 2, cache)
                fps.append(dict(seq=seq, img=fr["img_name"], bbox=(x1, y1, x2, y2), area=fp["area"],
                                below_h=y1 - h, bottom=y2 >= 950, danger=tuple(fp["bbox"]) in danger))
    return fps


def table(fps_by_method):
    names = list(fps_by_method)
    rows = [("all false positives", lambda f: True),
            ("inside the danger zone", lambda f: f["danger"])]
    rows += [(n, (lambda lo, hi: lambda f: not f["bottom"] and lo <= f["below_h"] < hi)(lo, hi))
             for n, lo, hi in POSITION_BINS]
    rows += [("touching the bottom edge (in front of the camera)", lambda f: f["bottom"])]
    rows += [(f"size {n}", (lambda lo, hi: lambda f: lo <= f["area"] < hi)(lo, hi)) for n, lo, hi in SIZE_BINS]
    w = max(len(r[0]) for r in rows) + 2
    print(f"{'':{w}s}" + "".join(f"{n:>22s}" for n in names))
    for label, cond in rows:
        print(f"{label:{w}s}" + "".join(f"{sum(cond(f) for f in fps_by_method[n]):>22d}" for n in names))
    for n in names:
        per_seq = Counter(f["seq"] for f in fps_by_method[n])
        top = per_seq.most_common(5)
        share = sum(c for _, c in top) / max(1, len(fps_by_method[n]))
        print(f"\n{n}: top-5 sequences hold {100 * share:.0f}% of false positives: "
              + ", ".join(f"{s} ({c})" for s, c in top))


def sheet(mods_root, fps, out, n=12, seed=0):
    rng = np.random.default_rng(seed)
    pick = [fps[i] for i in rng.choice(len(fps), size=min(n, len(fps)), replace=False)]
    tiles = []
    for f in pick:
        img = cv2.imread(os.path.join(mods_root, "sequences", f["seq"], "frames", f["img"]))
        x1, y1, x2, y2 = f["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half_w, half_h = max(120, (x2 - x1)), max(90, (y2 - y1))
        X1, Y1 = max(0, cx - half_w), max(0, cy - half_h)
        X2, Y2 = min(img.shape[1], cx + half_w), min(img.shape[0], cy + half_h)
        crop = img[Y1:Y2, X1:X2].copy()
        cv2.rectangle(crop, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), (0, 0, 255), 2)
        crop = cv2.resize(crop, (320, 240))
        cv2.putText(crop, f"{f['seq'][:7]} {f['area']}px", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        tiles.append(crop)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(out, np.vstack(rows))
    print(f"\nwrote {out} ({len(pick)} random false positives)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mods-root", default="data/mods")
    ap.add_argument("--methods", nargs="+", default=["unet_r34", "ewasr_resnet18"])
    ap.add_argument("--sheet-method", default=None)
    ap.add_argument("--sheet-out", default="fp_examples.jpg")
    ap.add_argument("--sheet-min-area", type=int, default=0, help="only sample blobs at least this large")
    args = ap.parse_args()

    with open(os.path.join(args.mods_root, "mods.json")) as f:
        seq_names = [s["path"].strip("/").split("/")[0] for s in json.load(f)["dataset"]["sequences"]]
    fps = {m: collect(args.mods_root, m, seq_names) for m in args.methods}
    table(fps)
    if args.sheet_method:
        pool = [f for f in fps[args.sheet_method] if f["area"] >= args.sheet_min_area]
        sheet(args.mods_root, pool, args.sheet_out)


if __name__ == "__main__":
    main()
