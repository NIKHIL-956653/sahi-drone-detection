"""
SAHI Drone Detection — Ablation Study
======================================
Answers the interview question "why 512px slices with 0.2 overlap?"
with data instead of a shrug.

Sweeps slice size × overlap ratio (plus a normal-inference baseline)
on one image or a folder of images, and prints a markdown table of
detection counts and latency for each configuration.

Usage:
    python ablation.py                     # uses test.jpg
    python ablation.py path/to/image.jpg
    python ablation.py path/to/folder/     # averages over all images
    python ablation.py --device cuda:0
"""
import argparse
import os
import time

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

SLICE_SIZES = [256, 512, 768]
OVERLAPS = [0.1, 0.2, 0.3]


def collect_images(path):
    if os.path.isdir(path):
        exts = (".jpg", ".jpeg", ".png")
        imgs = [os.path.join(path, f) for f in sorted(os.listdir(path))
                if f.lower().endswith(exts)]
        if not imgs:
            raise SystemExit(f"No images found in {path}")
        return imgs
    if not os.path.exists(path):
        raise SystemExit(f"Image not found: {path} — run download_test.py first")
    return [path]


def main():
    ap = argparse.ArgumentParser(description="Slice-size / overlap ablation for SAHI")
    ap.add_argument("path", nargs="?", default="test.jpg",
                    help="Image file or folder of images (default: test.jpg)")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    images = collect_images(args.path)
    print(f"Ablating over {len(images)} image(s), conf={args.conf}, device={args.device}")

    print("Loading model...")
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path="yolov8n.pt",
        confidence_threshold=args.conf, device=args.device,
    )

    rows = []

    # Baseline: normal full-image inference
    total_dets, total_ms = 0, 0.0
    for img in images:
        t0 = time.time()
        preds = get_prediction(img, model).object_prediction_list
        total_ms += (time.time() - t0) * 1000
        total_dets += len(preds)
    baseline_dets = total_dets / len(images)
    baseline_ms = total_ms / len(images)
    rows.append(("— (normal)", "—", baseline_dets, baseline_ms, 0.0))
    print(f"  baseline: {baseline_dets:.1f} dets, {baseline_ms:.0f} ms")

    # Sweep
    for size in SLICE_SIZES:
        for ov in OVERLAPS:
            total_dets, total_ms = 0, 0.0
            for img in images:
                t0 = time.time()
                preds = get_sliced_prediction(
                    img, model,
                    slice_height=size, slice_width=size,
                    overlap_height_ratio=ov, overlap_width_ratio=ov,
                    verbose=0,
                ).object_prediction_list
                total_ms += (time.time() - t0) * 1000
                total_dets += len(preds)
            avg_dets = total_dets / len(images)
            avg_ms = total_ms / len(images)
            gain = ((avg_dets - baseline_dets) / baseline_dets * 100) if baseline_dets else 0
            rows.append((f"{size}px", f"{int(ov*100)}%", avg_dets, avg_ms, gain))
            print(f"  slice={size} overlap={ov}: {avg_dets:.1f} dets, {avg_ms:.0f} ms")

    # Markdown table — paste into README
    print("\n" + "=" * 60)
    print("ABLATION RESULTS (paste into README)")
    print("=" * 60)
    print("\n| Slice size | Overlap | Avg detections | Avg latency (ms) | vs normal |")
    print("|------------|---------|----------------|------------------|-----------|")
    for slice_s, ov, dets, ms, gain in rows:
        gain_s = "baseline" if slice_s.startswith("—") else f"{gain:+.0f}%"
        print(f"| {slice_s} | {ov} | {dets:.1f} | {ms:.0f} | {gain_s} |")

    best = max(rows[1:], key=lambda r: r[2])
    print(f"\nMost detections: slice={best[0]}, overlap={best[1]} "
          f"({best[2]:.1f} avg dets, {best[3]:.0f} ms) — note the latency cost;")
    print("pick the config where the detection curve flattens, not just the max.")


if __name__ == "__main__":
    main()
