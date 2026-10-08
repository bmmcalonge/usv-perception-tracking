"""
Adapter that runs eWaSR through this repo's pipeline, for like-for-like comparisons.

eWaSR (Teršek, Žust, Kristan, "eWaSR -- an embedded-compute-ready maritime obstacle
detection network", Sensors 2023) is a published embedded-oriented maritime segmentation
network trained on the same MaSTr1325 data and 512x384 input as this repo's model.
Code and weights (Apache-2.0): https://github.com/tersekmatija/eWaSR

segment() mirrors predict.segment(): same resize and ImageNet normalization, then the
network, then eWaSR's own output handling (bilinear upsampling of the logits to the input
size before argmax, as in its wasr/inference.py), then the same nearest-neighbour resize
back to the frame. Class ids match this repo: 0 obstacle, 1 water, 2 sky.

eWaSR pins timm==0.6.5 and imports pytorch-lightning, so this runs in a separate
environment (see README, "Comparison with eWaSR").
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HEIGHT, WIDTH = 384, 512


def load_ewasr(repo, weights, variant="ewasr_resnet18", device="cuda"):
    """Build eWaSR from its repository and load the released weights."""
    sys.path.insert(0, os.path.abspath(repo))
    import torchvision
    import wasr.models as models
    # eWaSR builds ResNet-18 with torchvision's old pretrained=True argument (an ImageNet
    # download); those weights are overwritten by the checkpoint, so build it without them.
    models.resnet18 = lambda pretrained=False, **kw: torchvision.models.resnet18(weights=None)
    model = models.get_model(variant, num_classes=3, pretrained=False)
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception:
        # official release checkpoint from the authors' GitHub; may need full unpickling
        state = torch.load(weights, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.imu = variant.endswith("_imu")
    return model.to(device).eval()


@torch.no_grad()
def segment(model, img_rgb, device="cuda", imu_mask=None):
    """Full-resolution train-id mask for an RGB image (same contract as predict.segment).

    imu_mask: optional HxW horizon mask (MODS 'imus/' png); required only by the _imu variant.
    """
    h0, w0 = img_rgb.shape[:2]
    tf = A.Compose([A.Resize(HEIGHT, WIDTH),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2()])
    x = tf(image=img_rgb)["image"].unsqueeze(0).to(device)
    if imu_mask is None:
        imu = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.bool, device=device)
    else:
        imu = cv2.resize(imu_mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
        imu = torch.from_numpy(imu).unsqueeze(0).to(device)
    logits = model({"image": x, "imu_mask": imu})["out"]
    logits = F.interpolate(logits, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)
