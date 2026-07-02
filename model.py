"""
Segmentation model factory.

We use segmentation_models_pytorch (smp): a pretrained ImageNet encoder plus a
decoder head. This is the fastest robust path to a working maritime segmenter —
no custom architecture, and the pretrained encoder means MaSTr1325's ~1.3k
images are plenty to fine-tune on.

Default: U-Net + ResNet-34. DeepLabV3+ tends to give a couple of mIoU points
more on thin structures (masts, distant boats) at higher compute — switch with
--arch deeplabv3plus in train.py.
"""

import segmentation_models_pytorch as smp

_ARCHS = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "fpn": smp.FPN,
}


def build_model(arch="unet", encoder="resnet34", num_classes=3,
                encoder_weights="imagenet"):
    arch = arch.lower()
    if arch not in _ARCHS:
        raise ValueError(f"arch must be one of {list(_ARCHS)}, got {arch}")
    return _ARCHS[arch](
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )


# ImageNet normalization (the pretrained encoders expect this)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
