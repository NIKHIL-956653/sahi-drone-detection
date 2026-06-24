"""
Visual comparison — saves annotated images side by side
so you can SEE the difference between normal and SAHI.
"""
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
import cv2
import os
import sys


def load_model(confidence=0.3, device="cpu"):
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolov8n.pt",
        confidence_threshold=confidence,
        device=device,
    )


def draw_boxes(image_path, detections, color, label):
    """Draw bounding boxes on a copy of the image."""
    img = cv2.imread(image_path)
    for det in detections:
        box = det.bbox
        x1, y1, x2, y2 = int(box.minx), int(box.miny), int(box.maxx), int(box.maxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Header banner with count
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img, f"{label}: {len(detections)} objects",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    os.makedirs("output", exist_ok=True)
    print("Loading model...")
    model = load_model()

    # Normal
    print("Running normal inference...")
    normal = get_prediction(image_path, model).object_prediction_list
    normal_img = draw_boxes(image_path, normal, (0, 0, 255), "NORMAL YOLO")  # red
    cv2.imwrite("output/1_normal.jpg", normal_img)

    # SAHI
    print("Running SAHI sliced inference...")
    sahi = get_sliced_prediction(
        image_path, model,
        slice_height=512, slice_width=512,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
    ).object_prediction_list
    sahi_img = draw_boxes(image_path, sahi, (0, 255, 0), "SAHI SLICED")  # green
    cv2.imwrite("output/2_sahi.jpg", sahi_img)

    # Side by side
    h = min(normal_img.shape[0], sahi_img.shape[0])
    combined = cv2.hconcat([
        cv2.resize(normal_img, (int(normal_img.shape[1]*h/normal_img.shape[0]), h)),
        cv2.resize(sahi_img, (int(sahi_img.shape[1]*h/sahi_img.shape[0]), h)),
    ])
    cv2.imwrite("output/3_comparison.jpg", combined)

    print("\n✅ Saved to output/ folder:")
    print("   1_normal.jpg     — red boxes")
    print("   2_sahi.jpg       — green boxes")
    print("   3_comparison.jpg — SIDE BY SIDE (use this for interviews!)")
    print(f"\nNormal: {len(normal)} | SAHI: {len(sahi)} | +{len(sahi)-len(normal)} more")