"""
SAHI Drone Detection — Video + Object Tracking
===============================================
Runs SAHI sliced inference frame-by-frame on drone footage and (optionally)
tracks objects across frames with ByteTrack, so each object keeps a stable
ID as it moves. Writes an annotated output video.

Usage:
    python video_detect.py drone_footage.mp4
    python video_detect.py drone_footage.mp4 --no-track      # detection only
    python video_detect.py footage.mp4 --skip 2              # process every 3rd frame (faster)
    python video_detect.py footage.mp4 --device cuda:0
    python video_detect.py footage.mp4 --normal              # baseline, no slicing

Requires: pip install supervision   (for ByteTrack + annotators)
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction


def preds_to_sv_detections(pred_list):
    """Convert SAHI ObjectPrediction list to a supervision.Detections object."""
    import supervision as sv
    if not pred_list:
        return sv.Detections.empty()
    xyxy = np.array([[p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy]
                     for p in pred_list], dtype=np.float32)
    conf = np.array([p.score.value for p in pred_list], dtype=np.float32)
    class_id = np.array([p.category.id for p in pred_list], dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)


def main():
    ap = argparse.ArgumentParser(description="SAHI sliced inference on video, with tracking")
    ap.add_argument("video", help="Path to input video (mp4/avi/mov)")
    ap.add_argument("-o", "--output", default=None,
                    help="Output path (default: output/<name>_sahi.mp4)")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--slice-size", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--device", default="cpu", help="'cpu' or 'cuda:0'")
    ap.add_argument("--skip", type=int, default=0,
                    help="Skip N frames between processed frames (0 = every frame)")
    ap.add_argument("--normal", action="store_true",
                    help="Use normal full-frame inference instead of SAHI (baseline)")
    ap.add_argument("--no-track", action="store_true", help="Disable ByteTrack tracking")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}")

    try:
        import supervision as sv
    except ImportError:
        sys.exit("supervision missing — run: pip install supervision")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs("output", exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    mode = "normal" if args.normal else "sahi"
    out_path = args.output or os.path.join("output", f"{stem}_{mode}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("Loading YOLOv8 model...")
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path="yolov8n.pt",
        confidence_threshold=args.conf, device=args.device,
    )
    class_names = model.category_mapping  # {id_str: name}

    tracker = None if args.no_track else sv.ByteTrack(frame_rate=int(fps))
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.45, text_thickness=1)
    trace_annotator = None if args.no_track else sv.TraceAnnotator(thickness=2, trace_length=int(fps))

    frame_i, processed, t_start = 0, 0, time.time()
    unique_ids = set()
    last_detections = sv.Detections.empty()

    print(f"Processing {n_frames} frames "
          f"({'every frame' if args.skip == 0 else f'every {args.skip + 1}th frame'})...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        run_this_frame = (args.skip == 0) or ((frame_i - 1) % (args.skip + 1) == 0)
        if run_this_frame:
            # SAHI needs a file path or PIL/numpy RGB image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if args.normal:
                preds = get_prediction(rgb, model).object_prediction_list
            else:
                preds = get_sliced_prediction(
                    rgb, model,
                    slice_height=args.slice_size, slice_width=args.slice_size,
                    overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
                    verbose=0,
                ).object_prediction_list
            detections = preds_to_sv_detections(preds)
            if tracker is not None:
                detections = tracker.update_with_detections(detections)
                if detections.tracker_id is not None:
                    unique_ids.update(int(t) for t in detections.tracker_id)
            last_detections = detections
            processed += 1
        else:
            detections = last_detections  # reuse boxes on skipped frames

        # Annotate
        labels = []
        for i in range(len(detections)):
            cid = int(detections.class_id[i]) if detections.class_id is not None else -1
            name = class_names.get(str(cid), str(cid))
            conf = detections.confidence[i] if detections.confidence is not None else 0
            if tracker is not None and detections.tracker_id is not None:
                labels.append(f"#{int(detections.tracker_id[i])} {name} {conf:.2f}")
            else:
                labels.append(f"{name} {conf:.2f}")

        annotated = frame.copy()
        if trace_annotator is not None:
            annotated = trace_annotator.annotate(annotated, detections)
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)

        # HUD banner
        cv2.rectangle(annotated, (0, 0), (w, 36), (0, 0, 0), -1)
        hud = (f"{mode.upper()} | frame {frame_i}/{n_frames} | "
               f"objects: {len(detections)}")
        if tracker is not None:
            hud += f" | unique tracked: {len(unique_ids)}"
        cv2.putText(annotated, hud, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 156), 2)

        writer.write(annotated)
        if frame_i % 25 == 0:
            elapsed = time.time() - t_start
            print(f"  frame {frame_i}/{n_frames} "
                  f"({frame_i / max(elapsed, 1e-6):.1f} fps overall)", flush=True)

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    print(f"\n[DONE] {out_path}")
    print(f"  {frame_i} frames, {processed} inferences, {elapsed:.1f}s "
          f"({frame_i / max(elapsed, 1e-6):.1f} fps effective)")
    if tracker is not None:
        print(f"  Unique objects tracked: {len(unique_ids)}")


if __name__ == "__main__":
    main()
