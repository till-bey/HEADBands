#!/usr/bin/env python3
"""Run the Ultralytics model on test.mp4 or live webcam frames."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "exp-mix.pt"
DEFAULT_VIDEO = SCRIPT_DIR / "test.mp4"
DEFAULT_OUTPUT = SCRIPT_DIR / "test_annotated.mp4"


def predict_timed(model, frame, confidence: float, device: str | None):
	started = time.perf_counter()
	result = model.predict(frame, conf=confidence, device=device, verbose=False)[0]
	return result, (time.perf_counter() - started) * 1000.0


def run(model_path: Path, video_path: Path, output_path: Path, confidence: float, device: str | None) -> None:
	if not model_path.is_file():
		raise FileNotFoundError(f"TorchScript model not found: {model_path}")
	if not video_path.is_file():
		raise FileNotFoundError(f"Input video not found: {video_path}")

	model = YOLO(str(model_path), task="detect")
	capture = cv2.VideoCapture(str(video_path))
	if not capture.isOpened():
		raise RuntimeError(f"Could not open video: {video_path}")

	fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
	width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	writer = cv2.VideoWriter(
		str(output_path),
		cv2.VideoWriter_fourcc(*"mp4v"),
		fps,
		(width, height),
	)
	if not writer.isOpened():
		capture.release()
		raise RuntimeError(f"Could not create output video: {output_path}")

	print(f"Model: {model_path}")
	print(f"Input: {video_path} ({width}x{height}, {fps:.2f} FPS, {frame_count} frames)")
	print(f"Output: {output_path}")
	processed = 0
	total_inference_ms = 0.0
	try:
		while True:
			ok, frame = capture.read()
			if not ok:
				break
			result, inference_ms = predict_timed(model, frame, confidence, device)
			writer.write(result.plot())
			processed += 1
			total_inference_ms += inference_ms
			if processed % 25 == 0:
				average_ms = total_inference_ms / processed
				print(f"Processed {processed}/{frame_count or '?'} | inference: {inference_ms:.2f} ms | average: {average_ms:.2f} ms", end="\r")
	finally:
		capture.release()
		writer.release()
	print(f"Processed {processed} frames. Saved: {output_path}")


def run_camera(model_path: Path, camera_index: int, confidence: float, device: str | None) -> None:
	if not model_path.is_file():
		raise FileNotFoundError(f"Model not found: {model_path}")

	model = YOLO(str(model_path), task="detect")
	capture = cv2.VideoCapture(camera_index)
	if not capture.isOpened():
		raise RuntimeError(f"Could not open webcam {camera_index}")

	print(f"Model: {model_path}")
	print(f"Webcam: {camera_index}. Press q or Esc to quit.")
	processed = 0
	total_inference_ms = 0.0
	try:
		while True:
			ok, frame = capture.read()
			if not ok:
				print("Could not read a webcam frame.")
				break
			result, inference_ms = predict_timed(model, frame, confidence, device)
			annotated = result.plot()
			cv2.imshow("TorchScript webcam inference", annotated)
			processed += 1
			total_inference_ms += inference_ms
			if processed % 10 == 0:
				print(f"Frames: {processed} | inference: {inference_ms:.2f} ms | average: {total_inference_ms / processed:.2f} ms", flush=True)
			key = cv2.waitKey(1) & 0xFF
			if key in (ord("q"), 27):
				break
	finally:
		capture.release()
		cv2.destroyAllWindows()


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
	parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--camera", type=int, help="Webcam index; use this instead of --video")
	parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence")
	parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu or 0")
	args = parser.parse_args()
	if args.camera is not None:
		run_camera(args.model, args.camera, args.conf, args.device)
	else:
		run(args.model, args.video, args.output, args.conf, args.device)


if __name__ == "__main__":
	main()
