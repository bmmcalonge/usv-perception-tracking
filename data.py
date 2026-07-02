"""
MaSTr1325 dataset loader + label remapping.

MaSTr1325 raw per-pixel label encoding (ViCoS convention):
    0 = obstacles / environment  (everything that is not water or sky:
                                   boats, buoys, shoreline, piers, etc.)
    1 = water (sea)
    2 = sky
    4 = unknown                  -> ignored during training/eval

We train a 3-class segmentation model:
    0 = obstacle   1 = water   2 = sky

NOTE: MaSTr1325 folds *all* non-water/non-sky pixels into one "obstacle"
class, so the model does not distinguish a floating buoy from a distant
shoreline. We recover individual obstacle instances later, geometrically,
in detect.py (connected components). This is a deliberate simplification —
see the "known failure modes" section of the README.
"""

import os
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset

IGNORE_INDEX = 255
RAW_TO_TRAIN = {0: 0, 1: 1, 2: 2, 4: IGNORE_INDEX}
CLASS_NAMES = ["obstacle", "water", "sky"]
NUM_CLASSES = 3

# BGR colors for visualization (obstacle=red, water=blue, sky=light)
CLASS_COLORS = {
    0: (0, 0, 220),     # obstacle  -> red
    1: (200, 120, 0),   # water     -> blue
    2: (220, 200, 160), # sky       -> pale
}


def remap_mask(raw_mask: np.ndarray) -> np.ndarray:
    """Map raw MaSTr1325 label ids to contiguous train ids (4 -> IGNORE)."""
    out = np.full(raw_mask.shape, IGNORE_INDEX, dtype=np.uint8)
    for raw, train in RAW_TO_TRAIN.items():
        out[raw_mask == raw] = train
    return out


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Turn a train-id mask into a BGR image for quick visual inspection."""
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        vis[mask == cid] = color
    return vis


class MaSTr1325(Dataset):
    """
    Expects two folders:
        img_dir  : MaSTr1325 RGB images  (.jpg or .png)
        mask_dir : corresponding label masks (.png)

    Mask filenames in the official release usually match the image stem with
    a trailing 'm' (e.g. 0001.jpg -> 0001m.png). We try a few common patterns;
    if your layout differs, pass mask_suffix explicitly.
    """

    def __init__(self, img_dir, mask_dir, transform=None, mask_suffix="m"):
        self.img_paths = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg"))
            + glob.glob(os.path.join(img_dir, "*.png"))
        )
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_suffix = mask_suffix
        # validate one pairing up front so failures are loud and early
        self._mask_path(self.img_paths[0])

    def _mask_path(self, img_path):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        candidates = [stem + self.mask_suffix, stem, stem + "_mask"]
        for cand in candidates:
            for ext in (".png", ".jpg"):
                p = os.path.join(self.mask_dir, cand + ext)
                if os.path.exists(p):
                    return p
        raise FileNotFoundError(
            f"No mask for {img_path}. Tried {candidates} in {self.mask_dir}. "
            f"Set mask_suffix to match your files."
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        raw = cv2.imread(self._mask_path(img_path), cv2.IMREAD_GRAYSCALE)
        mask = remap_mask(raw)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        return img, mask.long()
