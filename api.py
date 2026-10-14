"""
SAHI Drone Detection — FastAPI Backend (v2)
Upload image → get normal vs SAHI comparison.

v2 additions:
- Tunable params per request: slice_size, overlap, confidence
- Full detection JSON (class name, confidence, bbox) alongside images
- Class labels + confidence drawn on boxes
- Per-class counts
- Unique temp file per request (concurrent-safe)
- Proper error handling for invalid/corrupt uploads
"""
import base64
import os
import time
import uuid
from collections import Counter

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

app = FastAPI(title="SAHI Drone Detection", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_MB = 25
DEVICE = os.environ.get("SAHI_DEVICE", "cpu")  # set SAHI_DEVICE=cuda:0 for GPU

# Load model once at startup
print(f"Loading YOLO model on {DEVICE}...")
model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolov8n.pt",
    confidence_threshold=0.3,
    device=DEVICE,
)
print("Model ready!")


def draw_boxes(img, detections, color, label):
    """Draw labelled boxes + a header banner with the count."""
    img = img.copy()
    for det in detections:
        box = det.bbox
        x1, y1 = int(box.minx), int(box.miny)
        x2, y2 = int(box.maxx), int(box.maxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        tag = f"{det.category.name} {det.score.value:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y1 - 4 if y1 - th - 8 > 40 else y2 + th + 6
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(img, tag, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img, f"{label}: {len(detections)} objects",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img


def detections_json(detections):
    """Serialize detections + per-class counts."""
    dets = [{
        "class": det.category.name,
        "confidence": round(float(det.score.value), 3),
        "bbox": [int(det.bbox.minx), int(det.bbox.miny),
                 int(det.bbox.maxx), int(det.bbox.maxy)],
    } for det in detections]
    by_class = dict(Counter(d["class"] for d in dets).most_common())
    return dets, by_class


def img_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@app.get("/health")
def health():
    return {"status": "online", "model": "YOLOv8n + SAHI", "device": DEVICE}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    slice_size: int = Form(512),
    overlap: float = Form(0.2),
    confidence: float = Form(0.3),
):
    # ── Validate params ──────────────────────────────────────
    if not 128 <= slice_size <= 2048:
        raise HTTPException(422, "slice_size must be between 128 and 2048")
    if not 0.0 <= overlap <= 0.5:
        raise HTTPException(422, "overlap must be between 0.0 and 0.5")
    if not 0.05 <= confidence <= 0.95:
        raise HTTPException(422, "confidence must be between 0.05 and 0.95")

    # ── Validate upload ──────────────────────────────────────
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB)")
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image — upload a valid JPG/PNG")

    # Unique temp file per request → concurrent requests don't collide
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", f"{uuid.uuid4().hex}.jpg")
    cv2.imwrite(temp_path, img)

    model.confidence_threshold = confidence
    try:
        t1 = time.time()
        normal = get_prediction(temp_path, model).object_prediction_list
        normal_time = (time.time() - t1) * 1000

        t2 = time.time()
        sahi = get_sliced_prediction(
            temp_path, model,
            slice_height=slice_size, slice_width=slice_size,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            verbose=0,
        ).object_prediction_list
        sahi_time = (time.time() - t2) * 1000
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    normal_img = draw_boxes(img, normal, (0, 0, 255), "NORMAL")
    sahi_img = draw_boxes(img, sahi, (0, 255, 0), "SAHI")

    normal_dets, normal_classes = detections_json(normal)
    sahi_dets, sahi_classes = detections_json(sahi)

    extra = len(sahi) - len(normal)
    pct = (extra / len(normal) * 100) if len(normal) > 0 else 0

    return JSONResponse({
        "params": {"slice_size": slice_size, "overlap": overlap,
                   "confidence": confidence},
        "normal": {
            "count": len(normal),
            "time_ms": round(normal_time, 1),
            "by_class": normal_classes,
            "detections": normal_dets,
            "image": img_to_base64(normal_img),
        },
        "sahi": {
            "count": len(sahi),
            "time_ms": round(sahi_time, 1),
            "by_class": sahi_classes,
            "detections": sahi_dets,
            "image": img_to_base64(sahi_img),
        },
        "improvement": {
            "extra_objects": extra,
            "percent": round(pct, 1),
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
