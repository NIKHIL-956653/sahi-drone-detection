"""
SAHI Drone Detection — FastAPI Backend
Upload image → get normal vs SAHI comparison
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
import cv2
import numpy as np
import base64
import os
import time

app = FastAPI(title="SAHI Drone Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
print("Loading YOLO model...")
model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolov8n.pt",
    confidence_threshold=0.3,
    device="cpu",
)
print("Model ready!")


def draw_boxes(img, detections, color, label):
    img = img.copy()
    for det in detections:
        box = det.bbox
        x1, y1, x2, y2 = int(box.minx), int(box.miny), int(box.maxx), int(box.maxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img, f"{label}: {len(detections)} objects",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img


def img_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@app.get("/health")
def health():
    return {"status": "online", "model": "YOLOv8n + SAHI"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    os.makedirs("temp", exist_ok=True)
    temp_path = "temp/upload.jpg"
    cv2.imwrite(temp_path, img)

    t1 = time.time()
    normal = get_prediction(temp_path, model).object_prediction_list
    normal_time = (time.time() - t1) * 1000

    t2 = time.time()
    sahi = get_sliced_prediction(
        temp_path, model,
        slice_height=512, slice_width=512,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
    ).object_prediction_list
    sahi_time = (time.time() - t2) * 1000

    normal_img = draw_boxes(img, normal, (0, 0, 255), "NORMAL")
    sahi_img = draw_boxes(img, sahi, (0, 255, 0), "SAHI")

    extra = len(sahi) - len(normal)
    pct = (extra / len(normal) * 100) if len(normal) > 0 else 0

    return JSONResponse({
        "normal": {
            "count": len(normal),
            "time_ms": round(normal_time, 1),
            "image": img_to_base64(normal_img),
        },
        "sahi": {
            "count": len(sahi),
            "time_ms": round(sahi_time, 1),
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