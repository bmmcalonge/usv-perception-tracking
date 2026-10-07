# Maritime Obstacle Perception: Segmentation + Kalman Tracking

A USV (uncrewed surface vessel) perception pipeline that detects and tracks obstacles in camera video.

```
camera frame ──▶ semantic segmentation ──▶ obstacle boxes ──▶ Kalman/SORT tracker ──▶ tracked obstacles
                 (U-Net / ResNet-34,        (connected comps   (constant-velocity KF,     (+ optional IMU
                  trained on MaSTr1325)      + shoreline filter) IoU + Hungarian)          rotation compensation)
```

**Project page with videos:** https://bmmcalonge.github.io/usv-perception-tracking-web/

1. **Segmentation.** A U-Net with an ImageNet-pretrained ResNet-34 encoder, fine-tuned on **MaSTr1325**, labels every pixel as obstacle, water or sky.
2. **Detection.** Connected components turn obstacle pixels into boxes. Geometric filters separate floating obstacles from shoreline (MaSTr1325 labels both as "obstacle").
3. **Tracking.** A SORT tracker written in plain NumPy: one constant-velocity Kalman filter per track, IoU cost, Hungarian assignment. Optional IMU rotation compensation warps each track's predicted state by `H = K·ΔR·K⁻¹`.

I come from controls and state estimation, so the design splits learned perception (the messy pixel problem) from classical estimation (temporal tracking), and keeps the estimation half small and auditable.

---

## Results

| What | Result | How it was measured |
|---|---|---|
| **MODS benchmark** (94 sequences, 8,175 frames) | water-edge RMSE **18 px**, obstacle F1 **90.3%** (84.8% inside the 15 m danger zone) | official [mods_evaluation](https://github.com/bborja/mods_evaluation) toolkit, `analysis/mods_predict.py`. Precision / recall 92.5% / 88.1% (danger zone 77.0% / 94.4%). For reference, the example WaSR result in the toolkit README is 21 px and 91.4% (88.0%); not re-run here. |
| MaSTr1325 validation mIoU | 0.991 (obstacle 0.979, water 0.996, sky 0.997) | random 85/15 image split (seed 0). **In-distribution and optimistic:** 70 of 198 val images have a near-identical training frame (`analysis/split_leakage.py`). |
| End-to-end throughput | **69 FPS** | RTX 5070, fp32, batch 1, 512×384 inference on 1278×958 frames. Includes JPEG decode, drawing and MP4 encoding. Measured with the kope81 command below. |
| Tracker self-tests | pass | `python tracker.py`: IDs survive a 3-frame dropout and tracking error drops from 3.68 to 2.73 px; camera-motion compensation keeps one ID under a steady pan and a rocking camera. |

MODD2 and MODS have no track-ID annotations, so MOTA/IDF1/HOTA can't be computed here. Tracking is evaluated with self-tests and ground-truth-free proxies (below).

---

## IMU motion compensation: what the data shows

MODD2 logs IMU roll/pitch/yaw per frame. The camera←IMU rotation `R_cam_imu` comes from ground-plane PCA on the stereo calibration sequences (`calibrate_imu.py`). To check whether the IMU actually describes the camera's motion, `analysis/imu_vs_vision.py` measures each frame-to-frame camera rotation from tracked far-field features and re-solves the camera←IMU rotation per sequence:

| Sequence | IMU vs camera rotation, corr (pitch / pan / roll) | Per-sequence fit vs `R_cam_imu` | Image-motion residual, none → IMU |
|---|---|---|---|
| kope75 | 0.91 / 0.97 / 0.90 | 4.2° | 6.22 → 2.28 px |
| kope81 | 0.83 / 0.89 / 0.91 | 3.4° | 2.67 → 1.28 px |
| kope82 | 0.85 / 0.81 / 0.86 | 5.2° | 10.89 → 3.76 px |
| kope67 (2 sequences) | ≈0 / 0.83–0.99 / ≈0 | no rotation fits | 3.68 → 4.71 px, 1.63 → 1.85 px |
| kope71 | ≈0 / 0.79 / ≈0 | no rotation fits | 8.02 → 14.64 px |

- **The calibration holds:** three independent sessions re-derive it within 3–5°.
- **In the kope67 and kope71 sessions the IMU's pitch and roll don't follow the camera** (only yaw does), so compensation adds error there. In the good sessions the IMU readings are also offset from their frame labels by about one frame (an offset of −1 fits best).

Effect on tracking (`analysis/imu_tracking_effect.py`), measured as IoU between each predicted box and its best-overlapping detection:

| Sequence | Camera motion (median) | No IMU | IMU compensation |
|---|---|---|---|
| kope82 (choppy, good IMU) | 0.65°/frame | median IoU 0.00 (mean 0.21) | **median IoU 0.70 (mean 0.55)** |
| kope81 (calm, good IMU) | 0.16°/frame | 0.92 | 0.76 |
| kope67 (pitch/roll unreliable) | 0.21°/frame | 0.74 | 0.68 |

Compensation pays off when camera motion is large and erratic. On calm water the Kalman velocity already predicts the slow drift, and the IMU's ~1 px of noise makes predictions slightly worse.

Two bugs in the first version made this invisible. `run_video.py` reset the calibration to `None`, so `--imu` did nothing. The tracker also warped only a copy of each prediction used for matching, which counts camera motion twice because the Kalman velocity already contains it. The tracker now warps the predicted state itself, as BoT-SORT does, and `tracker.py` has a self-test for both cases.

---

## Setup (Linux + virtualenv)

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# Install the CUDA torch build first. cu130 supports RTX 50-series (Blackwell, sm_120);
# older wheels (cu121/cu128) fail there with "no kernel image available".
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"

pip install -r requirements.txt

# sanity check, no GPU or data needed
python tracker.py
```

**Pretrained weights** (skip training):
```bash
mkdir -p runs/unet_r34
wget -O runs/unet_r34/best.pt \
  https://github.com/bmmcalonge/usv-perception-tracking/releases/download/v1.1/unet_r34_mastr1325.pt
# sha256: acc51cfef620c33341480ec52d35f5b286f3a70d3b81cd71a46161d2e45c72b0
```

---

## Data

All from the University of Ljubljana ViCoS lab: https://box.vicos.si/borja/viamaro/index.html

- **MaSTr1325** (training, 1,325 labeled images, CC BY 4.0): the 512×384 images and masks.
- **MODD2** (video sequences with IMU, plus calibration sequences for `calibrate_imu.py`).
- **MODS** (benchmark, 4.3 GB): download link in the [mods_evaluation](https://github.com/bborja/mods_evaluation) README.

```
data/
  MaSTr1325_images_512x384/   0001.jpg ...
  MaSTr1325_masks_512x384/    0001m.png ...          # 'm' suffix (--mask-suffix)
  video_data/                                         # MODD2, 28 sequences
    kope81-00-00000560-00001080/
      frames/  00000560L.jpg 00000560R.jpg ...        # stereo interleaved, 1278x958
      imu/  calibration.yaml
  calibration_sequences/  scal0-.../ scal1-.../       # MODD2 calibration sequences
  mods/  mods.json  sequences/  calibration/          # MODS
```

MaSTr1325 masks: `0 = obstacle/environment, 1 = water, 2 = sky, 4 = unknown (ignored)`.
MODD2 frames are stereo interleaved, so always pass `--pattern '*L.jpg'`.

---

## Run

```bash
# Train (40 epochs, ~10 min on an RTX 5070)
python train.py --img-dir data/MaSTr1325_images_512x384 --mask-dir data/MaSTr1325_masks_512x384 \
                --arch unet --encoder resnet34 --epochs 40 --batch-size 8 --out runs/unet_r34

# Segmentation overlays on a folder of images
python predict.py --ckpt runs/unet_r34/best.pt --img-dir sample_imgs --out-dir preds

# Segment -> boxes on one image (for tuning the detector)
python viz_boxes.py --ckpt runs/unet_r34/best.pt --image frame.jpg --min-area 300 --overlay

# Rank MODD2 sequences by floating-obstacle count
python scan_sequences.py --ckpt runs/unet_r34/best.pt --video-root data/video_data

# Full pipeline -> annotated video + FPS (the throughput number above)
python run_video.py --ckpt runs/unet_r34/best.pt \
    --frames data/video_data/kope81-00-00000560-00001080/frames --pattern '*L.jpg' \
    --mode enclosed --max-width-frac 0.6 --min-area 300 --max-age 20 --iou 0.2 \
    --out out_kope81.mp4 --fps 15

# IMU compensation: calibrate once, then pass --imu
python calibrate_imu.py --calib-dir data/calibration_sequences
python run_video.py --ckpt runs/unet_r34/best.pt \
    --frames data/video_data/kope82-00-00011177-00011797/frames --pattern '*L.jpg' \
    --mode adjacent --max-width-frac 0.3 --max-aspect 5 --min-area 800 --min-hits 2 --max-age 20 --iou 0.2 \
    --imu data/video_data/kope82-00-00011177-00011797/imu \
    --cam-imu-calib data/calibration_sequences/R_cam_imu.npy --out out_kope82_imu.mp4
```

Detector modes (`--mode`): `all` (every obstacle blob), `adjacent` (blobs bordering water, the default), `enclosed` (floating obstacles only: water directly above the blob, which drops shoreline). `--max-width-frac` drops the wide coast band, and `--max-aspect` drops mooring chains and ropes (thin strips around 16:1). Tracker knobs: `--max-age` (frames a track coasts without a detection), `--min-hits` (detections before a track is shown), `--iou` (association threshold).

`run_video.py` writes MPEG-4 Part 2 video, which browsers don't play. To convert to H.264, use the ffmpeg bundled with `imageio-ffmpeg`:
```bash
FF=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
$FF -i out_kope81.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart out_kope81_h264.mp4
```

---

## Evaluate

```bash
# MODS: write masks in the toolkit's format, then score with the official toolkit
python analysis/mods_predict.py --ckpt runs/unet_r34/best.pt --mods-root data/mods
git clone https://github.com/bborja/mods_evaluation data/mods/mods_evaluation
cd data/mods/mods_evaluation   # toolkit deps: yacs colorama scikit-image matplotlib prettytable
python modb_evaluation.py unet_r34 --config-file ../predictions/mods_config.yaml --workers 8

# Near-duplicates across the MaSTr1325 train/val split
python analysis/split_leakage.py --img-dir data/MaSTr1325_images_512x384

# IMU vs camera rotation per sequence, and IMU effect on tracking
python analysis/imu_vs_vision.py --ckpt runs/unet_r34/best.pt
python analysis/imu_tracking_effect.py --ckpt runs/unet_r34/best.pt
```

---

## Files

| file | role |
|------|------|
| `data.py` | MaSTr1325 loader, label remap, mask colorizer |
| `model.py` | segmentation model factory (smp: U-Net / DeepLabV3+ / FPN) |
| `train.py` | training loop, streaming confusion-matrix mIoU, checkpointing |
| `predict.py` | `segment()`, the shared inference function; colorized overlays |
| `detect.py` | obstacle mask → boxes (`all` / `adjacent` / `enclosed`, width and aspect filters) |
| `viz_boxes.py` | segment → boxes on one image |
| `scan_sequences.py` | rank MODD2 sequences by floating-obstacle count |
| `tracker.py` | SORT: per-track Kalman filter, IoU + Hungarian association, camera-motion compensation, self-tests |
| `imu.py` | MODD2 IMU loader and inter-frame rotation homography `H = K·ΔR_cam·K⁻¹` |
| `calibrate_imu.py` | camera←IMU rotation from stereo ground-plane PCA → `R_cam_imu.npy` |
| `run_video.py` | segment → detect → track on a sequence → annotated video, FPS, optional IMU and MOT metrics |
| `analysis/` | MODS predictions, split-leakage check, IMU-vs-vision check, IMU tracking effect |

---

## Design decisions and known limitations

**Single obstacle class → geometric instances.** MaSTr1325 labels everything that isn't water or sky as one "obstacle" class, so the model can't tell a buoy from a shoreline. `detect.py` recovers instances with connected components, and `enclosed` mode keeps only blobs with water directly above them. This is a heuristic: it merges touching vessels into one box, and in `enclosed` mode coastline-only sequences produce no tracks. A dataset with instance labels (e.g. LaRS) and a detection head would be the next step.

**Small and distant obstacles.** Inference runs at the 512×384 training resolution, so small objects near the horizon can disappear (MODS scores this directly). Running the same model at higher resolution over-segments water texture it never saw in training; the real fix is retraining at higher resolution. Separately, `--min-area` trades small-buoy recall against noise tracks from glints and wakes.

**Constant-velocity motion model.** Tracks coast through missed detections for up to `--max-age` frames. The filter lags sharp maneuvers, and the camera's own rotation looks like target motion unless it's compensated (see the IMU section).

**IoU-only association.** Fast small targets that don't overlap frame to frame won't associate, and similar-sized crossing targets can swap IDs. Appearance embeddings (DeepSORT/BoT-SORT style) would address this.

**Evaluation scope.** The MaSTr1325 number is in-distribution and inflated by near-duplicate frames; MODS is the out-of-distribution test. There's no tracking ground truth, so tracking claims rest on self-tests and proxies. Night, fog and open-ocean conditions are untested.

---

## Credits

Datasets by the ViCoS lab, University of Ljubljana:
- B. Bovcon, J. Muhovič, J. Perš, M. Kristan, "The MaSTr1325 dataset for training deep USV obstacle detection models," IROS 2019. (CC BY 4.0)
- B. Bovcon, R. Mandeljc, J. Perš, M. Kristan, "Stereo obstacle detection for unmanned surface vehicles by IMU-assisted semantic segmentation," Robotics and Autonomous Systems, 2018. (MODD2)
- B. Bovcon, J. Muhovič, D. Vranac, D. Mozetič, J. Perš, M. Kristan, "MODS — A USV-oriented object detection and obstacle segmentation benchmark," IEEE T-ITS, 2021.

Tracking: A. Bewley et al., "Simple Online and Realtime Tracking," ICIP 2016; N. Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian Tracking," 2022.

Code: MIT License. The released weights were trained on MaSTr1325 (CC BY 4.0); please cite the dataset if you use them.
