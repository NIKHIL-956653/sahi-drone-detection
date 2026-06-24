"""
SAHI Drone Detection — Core Engine
Compares normal YOLO inference vs SAHI sliced inference
"""
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
import cv2
import os
import time

# ─────────────────────────────────────────────
# Load YOLO model via SAHI's AutoDetectionModel
# ─────────────────────────────────────────────
def load_model(confidence=0.3, device="cpu"):
    """Load YOLOv8 through SAHI. Downloads weights on first run."""
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolov8n.pt",   # nano = fastest; downloads automatically
        confidence_threshold=confidence,
        device=device,             # "cpu" or "cuda:0"
    )
    return model


# ─────────────────────────────────────────────
# NORMAL inference (whole image at once)
# ─────────────────────────────────────────────
def detect_normal(model, image_path):
    start = time.time()
    result = get_prediction(image_path, model)
    elapsed = (time.time() - start) * 1000
    detections = result.object_prediction_list
    return detections, elapsed


# ─────────────────────────────────────────────
# SAHI SLICED inference (tiles + merge)
# ─────────────────────────────────────────────
def detect_sahi(model, image_path, slice_size=512, overlap=0.2):
    start = time.time()
    result = get_sliced_prediction(
        image_path,
        model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )
    elapsed = (time.time() - start) * 1000
    detections = result.object_prediction_list
    return detections, elapsed


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Run download_test.py first, or pass an image path!")
        sys.exit(1)

    print("Loading YOLO model (downloads weights on first run)...")
    model = load_model(confidence=0.3, device="cpu")

    print("\n" + "="*50)
    print("NORMAL INFERENCE (whole image)")
    print("="*50)
    normal_dets, normal_time = detect_normal(model, image_path)
    print(f"Detections: {len(normal_dets)}")
    print(f"Time: {normal_time:.0f} ms")

    print("\n" + "="*50)
    print("SAHI SLICED INFERENCE (tiles + merge)")
    print("="*50)
    sahi_dets, sahi_time = detect_sahi(model, image_path)
    print(f"Detections: {len(sahi_dets)}")
    print(f"Time: {sahi_time:.0f} ms")

    print("\n" + "="*50)
    print("RESULT")
    print("="*50)
    extra = len(sahi_dets) - len(normal_dets)
    if len(normal_dets) > 0:
        pct = (extra / len(normal_dets)) * 100
        print(f"SAHI found {extra} MORE objects (+{pct:.0f}%)")
    else:
        print(f"SAHI found {len(sahi_dets)} objects, normal found 0!")
    print(f"SAHI is slower ({sahi_time:.0f}ms vs {normal_time:.0f}ms) — the tradeoff for accuracy")