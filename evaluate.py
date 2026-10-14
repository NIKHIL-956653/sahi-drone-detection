"""
SAHI Drone Detection — Quantitative Evaluation on VisDrone
===========================================================
Downloads the VisDrone2019-DET validation set, runs BOTH normal YOLO
inference and SAHI sliced inference over it, and computes COCO-style
metrics (mAP@0.5, mAP@0.5:0.95, with a small-object breakdown).

This turns "SAHI found more boxes" into a defensible benchmark result:
    "SAHI improved small-object AP by X points on VisDrone-val."

Usage:
    python evaluate.py                    # full val set (548 images, slow on CPU)
    python evaluate.py --limit 50         # quick run on first 50 images
    python evaluate.py --device cuda:0    # GPU
    python evaluate.py --skip-normal      # only evaluate SAHI

Requires: pip install pycocotools
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

# ──────────────────────────────────────────────────────────────
# VisDrone dataset (val split — ~78 MB zip, 548 images)
# ──────────────────────────────────────────────────────────────
VISDRONE_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "VisDrone2019-DET-val.zip"
)
DATA_DIR = "data"
VAL_DIR = os.path.join(DATA_DIR, "VisDrone2019-DET-val")

# VisDrone annotation categories (index = category id in annotation files)
# 0: ignored, 1: pedestrian, 2: people, 3: bicycle, 4: car, 5: van,
# 6: truck, 7: tricycle, 8: awning-tricycle, 9: bus, 10: motor, 11: others
#
# COCO-pretrained YOLOv8 doesn't know VisDrone classes, so we evaluate on
# the classes both vocabularies share. Mapping (COCO id -> eval class):
#   person(0)     -> "person"      (VisDrone pedestrian=1 + people=2)
#   bicycle(1)    -> "bicycle"     (VisDrone bicycle=3)
#   car(2)        -> "car"         (VisDrone car=4 + van=5)
#   motorcycle(3) -> "motor"       (VisDrone motor=10)
#   bus(5)        -> "bus"         (VisDrone bus=9)
#   truck(7)      -> "truck"       (VisDrone truck=6)
EVAL_CLASSES = ["person", "bicycle", "car", "motor", "bus", "truck"]
COCO_TO_EVAL = {0: "person", 1: "bicycle", 2: "car", 3: "motor", 5: "bus", 7: "truck"}
VISDRONE_TO_EVAL = {1: "person", 2: "person", 3: "bicycle", 4: "car",
                    5: "car", 6: "truck", 9: "bus", 10: "motor"}
EVAL_CAT_IDS = {name: i + 1 for i, name in enumerate(EVAL_CLASSES)}


def download_visdrone():
    """Download and extract VisDrone val set if not present."""
    if os.path.isdir(os.path.join(VAL_DIR, "images")):
        print(f"[OK] VisDrone val set already at {VAL_DIR}")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "visdrone_val.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading VisDrone val set (~78 MB)...")
        urllib.request.urlretrieve(VISDRONE_URL, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    print(f"[OK] Extracted to {VAL_DIR}")


def load_ground_truth(limit=None):
    """Parse VisDrone txt annotations into COCO-format ground truth."""
    img_dir = os.path.join(VAL_DIR, "images")
    ann_dir = os.path.join(VAL_DIR, "annotations")
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
    if limit:
        files = files[:limit]

    images, annotations = [], []
    ann_id = 1
    for img_id, fname in enumerate(files, start=1):
        path = os.path.join(img_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})

        ann_path = os.path.join(ann_dir, fname.replace(".jpg", ".txt"))
        if not os.path.exists(ann_path):
            continue
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().rstrip(",").split(",")
                if len(parts) < 6:
                    continue
                x, y, bw, bh = map(int, parts[:4])
                score, cat = int(parts[4]), int(parts[5])
                # score==0 marks ignored regions in VisDrone
                if score == 0 or cat not in VISDRONE_TO_EVAL:
                    continue
                eval_cls = VISDRONE_TO_EVAL[cat]
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": EVAL_CAT_IDS[eval_cls],
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1

    gt = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cid, "name": name} for name, cid in EVAL_CAT_IDS.items()],
    }
    print(f"[OK] Ground truth: {len(images)} images, {len(annotations)} boxes")
    return gt


def predictions_to_coco(pred_list, img_id):
    """Convert SAHI ObjectPrediction list to COCO detection dicts."""
    out = []
    for det in pred_list:
        coco_id = det.category.id
        if coco_id not in COCO_TO_EVAL:
            continue
        b = det.bbox
        out.append({
            "image_id": img_id,
            "category_id": EVAL_CAT_IDS[COCO_TO_EVAL[coco_id]],
            "bbox": [b.minx, b.miny, b.maxx - b.minx, b.maxy - b.miny],
            "score": float(det.score.value),
        })
    return out


def run_inference(model, gt, mode, slice_size=512, overlap=0.2):
    """Run inference over all GT images. mode: 'normal' or 'sahi'."""
    img_dir = os.path.join(VAL_DIR, "images")
    results, total_time = [], 0.0
    n = len(gt["images"])
    for i, im in enumerate(gt["images"], 1):
        path = os.path.join(img_dir, im["file_name"])
        t0 = time.time()
        if mode == "normal":
            preds = get_prediction(path, model).object_prediction_list
        else:
            preds = get_sliced_prediction(
                path, model,
                slice_height=slice_size, slice_width=slice_size,
                overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                verbose=0,
            ).object_prediction_list
        total_time += time.time() - t0
        results.extend(predictions_to_coco(preds, im["id"]))
        if i % 10 == 0 or i == n:
            print(f"  [{mode}] {i}/{n} images  "
                  f"(avg {total_time / i * 1000:.0f} ms/img)", flush=True)
    return results, total_time / max(n, 1)


def coco_eval(gt, detections, label):
    """Run COCOeval and return key metrics."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_path, dt_path = "temp_gt.json", "temp_dt.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f)
    with open(dt_path, "w") as f:
        json.dump(detections, f)

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(dt_path) if detections else None
    os.remove(gt_path)
    os.remove(dt_path)
    if coco_dt is None:
        return {"AP": 0, "AP50": 0, "AP_small": 0, "AP_medium": 0, "AP_large": 0, "AR100": 0}

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    print(f"\n===== COCO EVAL: {label} =====")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    return {"AP": s[0], "AP50": s[1], "AP_small": s[3],
            "AP_medium": s[4], "AP_large": s[5], "AR100": s[8]}


def main():
    ap = argparse.ArgumentParser(description="Evaluate normal vs SAHI on VisDrone-val")
    ap.add_argument("--limit", type=int, default=None, help="Max images (default: all 548)")
    ap.add_argument("--device", default="cpu", help="'cpu' or 'cuda:0'")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--slice-size", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--skip-normal", action="store_true")
    args = ap.parse_args()

    try:
        import pycocotools  # noqa: F401
    except ImportError:
        sys.exit("pycocotools missing — run: pip install pycocotools")

    download_visdrone()
    gt = load_ground_truth(limit=args.limit)

    print("Loading YOLOv8 model...")
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path="yolov8n.pt",
        confidence_threshold=args.conf, device=args.device,
    )

    rows = []
    if not args.skip_normal:
        print("\n--- Normal inference ---")
        dets, avg_t = run_inference(model, gt, "normal")
        m = coco_eval(gt, dets, "NORMAL YOLO")
        rows.append(("Normal YOLOv8n", m, avg_t))

    print("\n--- SAHI sliced inference ---")
    dets, avg_t = run_inference(model, gt, "sahi",
                                slice_size=args.slice_size, overlap=args.overlap)
    m = coco_eval(gt, dets, f"SAHI {args.slice_size}px/{args.overlap}")
    rows.append((f"SAHI {args.slice_size}px slices", m, avg_t))

    # Markdown summary — paste straight into the README
    print("\n" + "=" * 70)
    print("RESULTS (paste into README)")
    print("=" * 70)
    print("\n| Method | mAP@0.5:0.95 | mAP@0.5 | AP-small | AP-medium | AR@100 | ms/img |")
    print("|--------|--------------|---------|----------|-----------|--------|--------|")
    for name, m, avg_t in rows:
        print(f"| {name} | {m['AP']:.3f} | {m['AP50']:.3f} | {m['AP_small']:.3f} "
              f"| {m['AP_medium']:.3f} | {m['AR100']:.3f} | {avg_t*1000:.0f} |")
    if len(rows) == 2:
        d = rows[1][1]["AP_small"] - rows[0][1]["AP_small"]
        print(f"\nSmall-object AP gain from SAHI: {d:+.3f} "
              f"({'+' if d >= 0 else ''}{d*100:.1f} points)")


if __name__ == "__main__":
    main()
