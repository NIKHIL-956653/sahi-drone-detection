# SAHI Drone Detection — Small Object Detection via Sliced Inference

Detecting small objects in aerial/drone imagery by slicing high-resolution
images into tiles, running detection on each, and merging results — improving
small-object detection by **+69%** over standard YOLO inference on the same image.

## The Problem
Drone at 100m altitude:
A person = 15×15 pixels in a 4000×3000 image
YOLO resizes the whole image to 640×640 first
→ that 15×15 person shrinks to ~2×2 pixels
→ basically vanishes → undetectable

## The Solution — SAHI
Instead of shrinking the whole image:

Slice into overlapping 512×512 tiles
Run YOLO on each tile at FULL resolution
→ small objects stay full-size, detectable!
Translate tile coordinates back to original image
Merge overlapping detections with NMS


## Proven Result

| Method | Objects Detected |
|--------|------------------|
| Normal YOLOv8 | 13 |
| SAHI Sliced | 22 |
| **Improvement** | **+9 (+69%)** |

*Same image, same model, same confidence threshold — the only difference is
slicing.*

## Why Small Objects Are Hard

1. **Downsampling** — resize-to-640 destroys small objects below the
   detector's receptive field
2. **Training bias** — COCO/ImageNet are mostly large, centered objects;
   models never learned good small-object features
3. **Feature map stride** — objects smaller than the backbone stride
   (8/16/32px) have nowhere to "land" on the detection grid

## How SAHI Merges Predictions

1. **Coordinate translation** — each tile detects in local coords (0-512);
   SAHI adds the tile offset to map back to original-image coords
2. **Optional full-image pass** — catches large objects that slicing might
   split across tiles
3. **NMS merge** — overlapping duplicates (same object seen in adjacent
   tiles due to 20% overlap) collapse to the single highest-confidence box

## Tech Stack

- **SAHI** — slicing aided hyper inference framework
- **YOLOv8** (Ultralytics) — base detection model
- **FastAPI** — REST backend
- **OpenCV** — image processing + annotation
- **Vanilla JS frontend** — upload + side-by-side comparison

## Setup

```bash
git clone https://github.com/NIKHIL-956653/sahi-drone-detection.git
cd sahi-drone-detection
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

### CLI comparison
```bash
python download_test.py      # grab a sample aerial image
python detect.py test.jpg    # numeric comparison
python compare.py test.jpg   # saves annotated images to output/
```

### Web app
```bash
python api.py                # FastAPI on :8000
# open frontend/index.html
```

## Key Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `slice_height/width` | 512 | Tile size — smaller = better tiny objects, slower |
| `overlap_ratio` | 0.2 | 20% tile overlap so boundary objects aren't cut |
| `confidence_threshold` | 0.3 | Minimum detection confidence |

## Tradeoff

SAHI is **slower** than normal inference (runs the detector N times, once per
tile, instead of once). It's an accuracy-for-speed tradeoff — worth it when
small-object recall matters (surveillance, search & rescue, aerial survey,
infrastructure inspection).

## Author

**Nikhil Chandra Sairam Tokala**
AI/ML Engineer | GenAI Engineer | DevOps — Dubai, UAE
[LinkedIn](https://linkedin.com/in/nikhil-chandra-133ncsr200233) |
[GitHub](https://github.com/NIKHIL-956653)