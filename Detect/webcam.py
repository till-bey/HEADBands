"""Run a ground-only detector on a live USB webcam or a local video file.

Examples:
    python webcam.py --input 0
    python webcam.py --input field.mp4 --output result.mp4
"""

import argparse
import time
from pathlib import Path

import cv2

from detector import DetectorSettings, detect, draw_detections, prepare_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic ground-frisbee webcam/video detector")
    parser.add_argument("--input", default="0", help="Camera index or local video path")
    parser.add_argument("--ground-start", type=int, default=25, help="Ground ROI begins at this image-height percentage")
    parser.add_argument("--output", default="", help="Optional MP4 output file")
    parser.add_argument("--no-display", action="store_true", help="Process video without opening a window")
    args = parser.parse_args()

    src = int(args.input) if args.input.isdecimal() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.input}")
    cfg = DetectorSettings(ground_start_pct=args.ground_start)
    writer = None
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = prepare_frame(frame)
            start = time.perf_counter()
            matches, _ = detect(frame, cfg)
            result = draw_detections(frame, matches, cfg)
            fps = 1.0 / max(1e-6, time.perf_counter() - start)
            cv2.putText(result, f"Ellipse contours | {len(matches)} candidate(s) | {fps:.1f} proc FPS",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            if args.output:
                if writer is None:
                    output = Path(args.output)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    frame_fps = cap.get(cv2.CAP_PROP_FPS)
                    if frame_fps < 1 or frame_fps > 240:
                        frame_fps = 25.0
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                             frame_fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create video output: {output}")
                writer.write(result)
            frame_count += 1
            if not args.no_display:
                cv2.imshow("Ground frisbee detection (q to quit)", result)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()
