# Maritime Obstacle Perception — Segmentation + Kalman Tracking

A small, honest USV (unmanned surface vessel) perception pipeline:

```
camera frame ──▶ semantic segmentation ──▶ obstacle boxes ──▶ Kalman/SORT tracker ──▶ tracked obstacles
                 (MaSTr1325-trained)        (connected comps)   (constant-velocity KF)
```

Two stages, deliberately:
1. **Segmentation** — a pretrained-encoder model fine-tuned on **MaSTr1325** classifies every pixel as `water / sky / obstacle`.
2. **Detection + tracking** — obstacle pixels become bounding boxes (connected components), which a **constant-velocity Kalman filter with IoU/Hungarian association** (SORT) tracks across frames with stable IDs, evaluated on **MODD2** video sequences.

The tracker is plain NumPy linear-algebra (no `filterpy`) so the estimation math is fully auditable — it's the same predict/correct state estimation used in closed-loop control, applied to obstacle motion.

---

## Why this design

I came at this from a controls / state-estimation background, so the architecture splits learned perception from classical estimation:
- A learned model is best for the messy pixel problem (what is water vs. an obstacle under glare, wakes, horizon haze).
- A Kalman filter is best for the temporal problem (where will this obstacle be next frame, maintain identity through a missed detection). This part is deterministic, cheap, and easy to reason about — which matters for a safety-relevant maritime system.

MaSTr1325 + MODD2 were chosen because they're the canonical small public USV benchmarks (segmentation training set + video evaluation set from the same group), so the whole thing runs on a single GPU in an evening.

---

## Setup (Linux + virtualenv recommended)

```bash
# 1) Python env
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# 2) Install the CUDA torch build FIRST. cu130 is the current stable default and
#    supports RTX 50-series (Blackwell, sm_120). Older wheels (cu121/cu128) will
#    fail at runtime on Blackwell with "no kernel image available".
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
#   Older driver fallback: use cu126.  Check https://pytorch.org/get-started/locally/

# 2b) Verify the GPU is actually recognized (sm_120 must appear for RTX 50-series):
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); \
print('device:', torch.cuda.get_device_name(0)); print('archs:', torch.cuda.get_arch_list())"

# 3) Everything else
pip install -r requirements.txt
# optional, for MOTA/IDF1:  pip install motmetrics

# 4) sanity check (no GPU/data needed) — verifies the Kalman tracker
python tracker.py
```

Windows works too, but Linux avoids CUDA/OpenCV-video friction; if on Windows use Conda and install torch from the same index URL.

---

## Data

- **MaSTr1325** (training, ~1,325 labeled images): https://www.vicos.si/resources/mastr1325/
- **MODD2** (evaluation video sequences): linked from the same ViCoS resources page.

Confirmed layout:
```
data/
  MaSTr1325_images_512x384/   0001.jpg 0002.jpg ...
  MaSTr1325_masks_512x384/    0001m.png 0002m.png ...   # 'm' suffix; set --mask-suffix
  video_data/                                            # MODD2, ~28 sequences
    kope81-00-00000560-00001080/
      frames/   00000560L.jpg 00000560R.jpg ...          # stereo interleaved, 1278x958
      imu/  calibration.yaml  stereo-method.yaml
    ...
```

MaSTr1325 raw mask ids: `0=obstacle/environment, 1=water, 2=sky, 4=unknown(ignored)`.

MODD2 frames are **stereo interleaved** (L/R in one folder) — always pass
`--pattern '*L.jpg'` to take a single camera. Most MODD2 sequences are marina
coastline with no discrete floating obstacle; use `scan_sequences.py` to find the
ones that do (e.g. `kope81-00-00000560-00001080` has a clean tracked buoy).

---

## Run

```bash
# Train segmentation (≈30–50 epochs; ~10 min on a modern GPU)
python train.py --img-dir data/MaSTr1325_images_512x384 \
                --mask-dir data/MaSTr1325_masks_512x384 \
                --arch unet --encoder resnet34 \
                --epochs 40 --batch-size 8 --out runs/unet_r34

# Inspect segmentation quality (colorized overlays)
python predict.py --ckpt runs/unet_r34/best.pt \
                  --img-dir sample_imgs --out-dir preds

# Find MODD2 sequences that actually contain floating obstacles
python scan_sequences.py --ckpt runs/unet_r34/best.pt \
                  --video-root data/video_data --min-area 300 --max-width-frac 0.6

# Full pipeline on a sequence -> annotated mp4 (+ FPS + track counts)
# 'enclosed' mode tracks only floating obstacles (drops shoreline); '*L.jpg' = one camera
python run_video.py --ckpt runs/unet_r34/best.pt \
                    --frames data/video_data/kope81-00-00000560-00001080/frames \
                    --pattern '*L.jpg' --mode enclosed --max-width-frac 0.6 \
                    --min-area 300 --max-age 20 --iou 0.2 \
                    --out out_kope81.mp4 --fps 15
```

Detector modes (`--mode`): `all` (every obstacle blob), `adjacent` (water-bordering
blobs, default), `enclosed` (floating obstacles only — water above them, drops
shoreline). `--max-width-frac 0.6` additionally drops the wide coast band.
`--max-aspect 5` suppresses mooring chains / dock ropes (thin horizontal strips, ~16:1
aspect) that survive the width filter.

Tracker knobs: `--max-age` (frames a track coasts without a detection),
`--min-hits` (detections before a track is shown), `--iou` (association threshold).

```bash
# IMU-compensated run (corrects predicted box positions by between-frame camera rotation
# before IoU matching — reduces ID swaps during waves or maneuvers)
# Requires: calibrate_imu.py to have been run first to produce R_cam_imu.npy
python run_video.py --ckpt runs/unet_r34/best.pt \
    --frames data/video_data/kope67-00-00062671-00063461/frames \
    --pattern '*L.jpg' --mode adjacent --max-width-frac 0.3 --max-aspect 5 \
    --min-area 800 --max-age 20 --min-hits 2 --iou 0.2 \
    --imu data/video_data/kope67-00-00062671-00063461/imu \
    --cam-imu-calib R_cam_imu.npy \
    --out out_kope67_imu.mp4 --fps 15
```

Report from training: `runs/.../metrics.csv` (per-epoch mIoU + per-class IoU).
Headline numbers for a résumé bullet: **val mIoU / obstacle IoU** (segmentation) and **end-to-end FPS** + **MOTA/IDF1** (tracking).

---

## Visualize / inspect results

Quick ways to *see* what the model is doing, useful right after training (and
before you have video to run the full pipeline on).

**Segmentation overlays** — colorized masks (water=blue, sky=pale, obstacle=red)
blended over the input, to confirm segmentation quality:

```bash
# overlay a small sample of the training images (don't dump all 1325)
mkdir -p sample_imgs
cp $(ls data/MaSTr1325_images_512x384/*.jpg | head -30) sample_imgs/
python predict.py --ckpt runs/unet_r34/best.pt --img-dir sample_imgs --out-dir preds
# open the preds/ folder, e.g.:  xdg-open preds
```
Note: MaSTr1325 samples are in-distribution, so overlays will look near-perfect.
That confirms training worked; the MODD2 video is the harder, more honest test.

**Detection boxes on one image** — runs segment → obstacle boxes (exactly what
the tracker is fed) and draws them. Best tool for tuning `--min-area` before the
video stage, especially since MODD2 frames (1278×958) are much larger than the
512×384 training size:

```bash
python viz_boxes.py --ckpt runs/unet_r34/best.pt --image sample_imgs/0001.jpg
# high-res frame? raise min-area and optionally blend the mask underneath:
python viz_boxes.py --ckpt runs/unet_r34/best.pt --image frame.jpg \
    --min-area 300 --overlay --out boxes_demo.jpg
```

---

## Report site (GitHub Pages)

`report/build_report.py` turns your real outputs into a static, Nerfies-style
report page (training plots + visualizations) you can publish on GitHub Pages.

```bash
pip install matplotlib    # if not already installed
# run from the project root, after training (and optionally after run_video.py):
python report/build_report.py --run runs/unet_r34 \
    --overlays preds --boxes boxes_demo.jpg --video out_kope81.mp4
# everything except --run is optional; missing pieces are skipped.
```

It reads `runs/unet_r34/metrics.csv` to plot the loss curve, per-class IoU curves,
and a final-IoU bar; copies a few `predict.py` overlays, your `viz_boxes.py` image,
and (if present) the tracked video into `report/static/images/`; and writes
`report/index.html`.

Preview locally by opening `report/index.html`. To publish: push the repo and set
GitHub Pages to serve the `report/` folder (or copy its contents to a `gh-pages`
branch). Rerun the script anytime new results come in to refresh the page.

| file | role |
|------|------|
| `data.py` | MaSTr1325 loader, label remap, mask colorizer |
| `model.py` | segmentation model factory (smp; U-Net / DeepLabV3+ / FPN) |
| `train.py` | training loop, streaming-confusion-matrix mIoU, checkpointing |
| `predict.py` | run segmenter on images → colorized overlays |
| `detect.py` | obstacle mask → boxes; `mode`=all/adjacent/enclosed; `--max-width-frac` drops coast band; `--max-aspect` drops mooring-chain strips |
| `viz_boxes.py` | segment→boxes on one image (`--mode`, `--max-width-frac`, `--max-aspect`, `--overlay`) |
| `scan_sequences.py` | rank MODD2 sequences by floating-obstacle count, to pick a demo seq |
| `tracker.py` | **SORT: per-track Kalman filter + IoU/Hungarian association** (self-tests) |
| `imu.py` | MODD2 IMU loader + delta rotation homography `H = K·ΔR_cam·K⁻¹` for camera motion compensation |
| `calibrate_imu.py` | compute camera←IMU rotation `R_cam_imu.npy` from stereo point cloud ground-plane PCA |
| `run_video.py` | segment→detect→track on a sequence → annotated video (+ optional MOT metrics, + optional IMU compensation) |
| `site/index.html` | static portfolio page: metrics, side-by-side IMU comparison videos, honest limitations |
| `report/build_report.py` | build a Nerfies-style report site (plots + visuals) from metrics.csv + images |

---

## Design decisions & known failure modes

Written down deliberately — these are the things worth being able to discuss, and the honest limits of a weekend-scale build.

**Single obstacle class → geometric instances.** MaSTr1325 labels *all* non-water/non-sky pixels as one "obstacle" class, so the model can't separate a buoy from a shoreline. `detect.py` recovers instances with connected components. Because the coast is one huge component, the `enclosed` mode keeps only obstacles with **water directly above them** (floating hazards below the sea edge) and `--max-width-frac` drops the wide coast band — mirroring MODD2's own large-vs-small-obstacle distinction. This is a heuristic, not a learned detector: it **merges visually overlapping boats** and **tracks only water-enclosed obstacles**, so most coastline-only MODD2 sequences yield no tracks (by design). A learned detection head (e.g. trained on WUTDet) is the natural next iteration.

**Detection range / small objects.** `predict.py::segment()` runs inference at the model's train resolution (512×384), downscaling the 1278×958 frames. Distant obstacles shrink below what segmentation (and `--min-area`) can catch, so e.g. a buoy is only picked up once it is **close**. `--infer-height` / `--infer-width` flags exist to override resolution, but feeding the 512×384-trained model larger inputs over-segments water texture (glints, small waves) it never saw in training — generating false positives rather than extending real detection range. The fix is to retrain on higher-resolution crops.

**Constant-velocity motion model.** Each track assumes roughly linear motion in image space. It handles missed detections by coasting (`max_age` frames), which is why IDs survive short occlusions — verified in `tracker.py`'s self-test through a 3-frame dropout. It will **lag on sharply maneuvering or accelerating targets**, and **own-ship roll/heave** induces apparent motion the filter reads as target motion.

**IMU motion compensation is implemented** (`imu.py`, `run_video.py --imu`): between-frame rotation homography `H = K·ΔR_cam·K⁻¹` shifts Kalman-predicted boxes by the camera's own rotation before IoU matching. The camera←IMU rotation is calibrated via ground-plane PCA on stereo point clouds (`calibrate_imu.py`). However, **this benefit cannot be demonstrated on MODD2**: the dataset has a hard split where the choppiest sequences (~0.010 rad/frame) are pure coastline, and the sequences with trackable floating obstacles have near-zero motion (~0.002–0.004 rad/frame). On kope67, IMU compensation slightly hurt tracking (30→33 total IDs, avg lifetime 147→117 frames). The architecture is correct; the dataset just doesn't exercise it.

**IoU-only association.** Matching uses bounding-box IoU alone. **Fast small targets that don't overlap frame-to-frame won't associate**, and **crossing targets of similar size can swap IDs** at the crossing point. Appearance embeddings (DeepSORT-style) would harden this; IoU was chosen for transparency and zero training cost.

**Domain shift.** The model trains on MaSTr1325's coastal Slovenian conditions. Expect **degradation on heavy glare, fog, night, or open-ocean** scenes outside that distribution — the standard segmentation generalization gap, and exactly why a production system needs far more diverse data than a public academic set.

**Eval scope.** mIoU is reported on a random MaSTr1325 val split (image-level); tracking metrics depend on supplying MODD2 ground truth. Numbers here demonstrate a working pipeline, not state-of-the-art.

---

## How this maps to a maritime perception role

- **Segmentation / object detection on real maritime data** — the MaSTr1325 stage.
- **Filters / sensor fusion / state estimation** — the Kalman tracker plus IMU motion compensation via rotation homography (`imu.py`, `calibrate_imu.py`), including camera←IMU calibration from stereo point cloud ground-plane PCA.
- **Low-latency / tracking pipelines** — `run_video.py` reports end-to-end FPS; the tracker is CPU-cheap.
- **Willingness to learn and pivot** — the whole thing is a controls engineer reaching into perception, with the bridges (estimation ↔ tracking) made explicit rather than hidden.
