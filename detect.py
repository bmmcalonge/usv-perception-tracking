"""
Turn a per-pixel obstacle mask into discrete obstacle bounding boxes.

The segmenter labels every non-water/non-sky pixel as "obstacle" (one class),
and MaSTr1325's obstacle class is "obstacles AND environment" -- so shoreline,
piers, and buildings are obstacle-colored too. For USV navigation what matters
is the small obstacles floating below the sea edge (boats, buoys), not the coast.

mask_to_boxes() takes connected components of the obstacle class and filters them
according to `mode`:

  "all"       every obstacle blob >= min_area.
  "adjacent"  blobs that border water (drops sky-only regions).      [default]
  "enclosed"  floating obstacles: border water AND have water directly ABOVE them
              -- a boat below the sea edge has sea above it, whereas shoreline has
              sky above it. This drops the coast and isolates discrete obstacles,
              mirroring MODD2's small-obstacle (below sea-edge) definition.

max_width_frac (optional, any mode): also drop blobs wider than this fraction of
the image width. The coast band is one component spanning most of the frame, so
e.g. max_width_frac=0.6 removes it while keeping boat/buoy-sized objects.

max_aspect (optional, any mode): drop blobs whose width/height ratio exceeds this
value. Mooring chains and ropes produce very thin horizontal strips (aspect ~16:1)
that survive the width filter; e.g. max_aspect=5 suppresses them while keeping
normal boat/buoy boxes (aspect typically < 3:1).

For MODD2 coastal sequences, mode="enclosed" (optionally with max_width_frac=0.6)
isolates floating obstacles for clean tracking.
"""

import cv2
import numpy as np

OBSTACLE_CLASS = 0
WATER_CLASS = 1


def mask_to_boxes(mask, min_area=80, mode="adjacent", max_width_frac=None,
                  max_aspect=None,
                  water_band=6, water_above_band=40, water_above_frac=0.5,
                  close_size=0):
    """
    mask       : HxW int array of train ids (0=obstacle,1=water,2=sky)
    close_size : morphological closing kernel size applied before component
                 labeling (0 = disabled). A value of 15-25 fills intra-boat
                 gaps (windows, reflections) so a fragmented hull becomes one
                 component rather than many partial detections.
    returns: (N,4) array of [x1,y1,x2,y2]
    """
    if mode not in ("all", "adjacent", "enclosed"):
        raise ValueError(f"mode must be all|adjacent|enclosed, got {mode!r}")

    H, W = mask.shape
    obstacle = (mask == OBSTACLE_CLASS).astype(np.uint8)
    if obstacle.sum() == 0:
        return np.zeros((0, 4), dtype=float)

    # fill intra-object gaps, then remove speckle
    if close_size > 0:
        obstacle = cv2.morphologyEx(obstacle, cv2.MORPH_CLOSE,
                                    np.ones((close_size, close_size), np.uint8))
    obstacle = cv2.morphologyEx(obstacle, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))

    water = (mask == WATER_CLASS).astype(np.uint8)
    water_d = cv2.dilate(water, np.ones((water_band, water_band), np.uint8))
    max_w = (max_width_frac * W) if max_width_frac else None

    n, labels, stats, _ = cv2.connectedComponentsWithStats(obstacle, connectivity=8)
    boxes = []
    for i in range(1, n):  # 0 is background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if max_w is not None and w > max_w:
            continue  # spans too much width -> shoreline/coast band
        if max_aspect is not None and h > 0 and (w / h) > max_aspect:
            continue  # too elongated -> mooring chain / rope artifact

        if mode in ("adjacent", "enclosed"):
            comp = labels[y:y + h, x:x + w] == i
            if not water_d[y:y + h, x:x + w][comp].any():
                continue  # no water nearby -> sky-only region

        if mode == "enclosed":
            top = max(0, y - water_above_band)
            band = water[top:y, x:x + w]
            # floating obstacle: the strip directly above it is mostly sea
            if band.size == 0 or band.mean() < water_above_frac:
                continue

        boxes.append([x, y, x + w, y + h])
    return np.array(boxes, dtype=float).reshape(-1, 4)
