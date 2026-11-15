#!/usr/bin/env python3
"""Fit and use a planar camera-pixel to robot-floor XY calibration."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class CameraFloorCalibration:
	"""Homography mapping image pixels ``(u, v)`` to robot floor ``(X, Y)``."""

	homography: np.ndarray
	rms_error_m: float
	point_count: int
	image_size: tuple[int, int] | None = None

	def pixel_to_robot(self, pixel: Iterable[float]) -> np.ndarray:
		point = np.array([float(tuple(pixel)[0]), float(tuple(pixel)[1]), 1.0])
		mapped = self.homography @ point
		if abs(mapped[2]) < 1e-12:
			raise ValueError("pixel maps to infinity; check the calibration points")
		return mapped[:2] / mapped[2]

	def robot_to_pixel(self, position: Iterable[float]) -> np.ndarray:
		point = np.array([float(tuple(position)[0]), float(tuple(position)[1]), 1.0])
		mapped = np.linalg.inv(self.homography) @ point
		return mapped[:2] / mapped[2]

	def save(self, path: str | Path) -> Path:
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"homography": self.homography.tolist(),
			"rms_error_m": self.rms_error_m,
			"point_count": self.point_count,
			"image_size": self.image_size,
			"source_to_target": "pixel_u_v_to_robot_X_Y",
		}
		path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
		return path

	@classmethod
	def load(cls, path: str | Path) -> "CameraFloorCalibration":
		payload = json.loads(Path(path).read_text(encoding="utf-8"))
		return cls(np.asarray(payload["homography"], dtype=float), float(payload["rms_error_m"]), int(payload["point_count"]), tuple(payload["image_size"]) if payload.get("image_size") else None)


def _normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	center = points.mean(axis=0)
	distance = np.sqrt(((points - center) ** 2).sum(axis=1)).mean()
	if distance < 1e-12:
		raise ValueError("calibration points must not all be identical")
	scale = np.sqrt(2.0) / distance
	transform = np.array([[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]])
	homogeneous = np.column_stack([points, np.ones(len(points))])
	return (transform @ homogeneous.T).T[:, :2], transform


def fit_homography(pixels: np.ndarray, robot_xy: np.ndarray) -> CameraFloorCalibration:
	pixels = np.asarray(pixels, dtype=float)
	robot_xy = np.asarray(robot_xy, dtype=float)
	if pixels.shape != robot_xy.shape or pixels.ndim != 2 or pixels.shape[1] != 2:
		raise ValueError("pixels and robot_xy must both have shape (N, 2)")
	if len(pixels) < 4:
		raise ValueError("at least four point correspondences are required")
	pixel_normalized, pixel_transform = _normalize(pixels)
	robot_normalized, robot_transform = _normalize(robot_xy)
	matrix = []
	for (u, v), (x, y) in zip(pixel_normalized, robot_normalized):
		matrix.extend(([-u, -v, -1, 0, 0, 0, u * x, v * x, x], [0, 0, 0, -u, -v, -1, u * y, v * y, y]))
	_, _, vectors = np.linalg.svd(np.asarray(matrix), full_matrices=True)
	normalized_homography = vectors[-1].reshape(3, 3)
	homography = np.linalg.inv(robot_transform) @ normalized_homography @ pixel_transform
	homography /= homography[2, 2]
	predicted = (homography @ np.column_stack([pixels, np.ones(len(pixels))]).T).T
	predicted = predicted[:, :2] / predicted[:, 2:3]
	rms = float(np.sqrt(np.mean(np.sum((predicted - robot_xy) ** 2, axis=1))))
	return CameraFloorCalibration(homography, rms, len(pixels))


def load_correspondences(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
	with Path(path).open(newline="", encoding="utf-8-sig") as stream:
		rows = list(csv.DictReader(row for row in stream if not row.lstrip().startswith("#")))
	if not rows:
		raise ValueError(f"no calibration rows found in {path}")
	keys = set(rows[0])
	pixel_keys = ("pixel_x", "pixel_y") if {"pixel_x", "pixel_y"} <= keys else ("u", "v")
	robot_keys = ("robot_x", "robot_y") if {"robot_x", "robot_y"} <= keys else ("X", "Y")
	if not set(pixel_keys) <= keys or not set(robot_keys) <= keys:
		raise ValueError("CSV needs pixel_x,pixel_y,robot_x,robot_y columns (u,v,X,Y are also accepted)")
	pixels = np.array([[float(row[pixel_keys[0]]), float(row[pixel_keys[1]])] for row in rows])
	robot_xy = np.array([[float(row[robot_keys[0]]), float(row[robot_keys[1]])] for row in rows])
	return pixels, robot_xy


def fit_from_csv(path: str | Path, image_size: tuple[int, int] | None = None, robot_scale: float = 1.0) -> CameraFloorCalibration:
	pixels, robot_xy = load_correspondences(path)
	calibration = fit_homography(pixels, robot_xy * robot_scale)
	calibration.image_size = image_size
	return calibration


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	subparsers = parser.add_subparsers(dest="command", required=True)
	fit = subparsers.add_parser("fit", help="fit and save a pixel-to-robot homography")
	fit.add_argument("--points", type=Path, required=True)
	fit.add_argument("--output", type=Path, required=True)
	fit.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
	fit.add_argument("--robot-scale", type=float, default=1.0, help="multiply CSV robot coordinates by this scale, e.g. 0.001 for millimetres to metres")
	convert = subparsers.add_parser("pixel", help="convert one pixel to robot XY")
	convert.add_argument("--calibration", type=Path, required=True)
	convert.add_argument("--pixel", type=float, nargs=2, required=True, metavar=("U", "V"))
	overlay = subparsers.add_parser("overlay", help="save an image showing calibration points")
	overlay.add_argument("--image", type=Path, required=True)
	overlay.add_argument("--points", type=Path, required=True)
	overlay.add_argument("--output", type=Path, required=True)
	overlay.add_argument("--robot-scale", type=float, default=1.0, help="multiply CSV robot coordinates by this scale")
	args = parser.parse_args()
	if args.command == "fit":
		calibration = fit_from_csv(args.points, tuple(args.image_size) if args.image_size else None, args.robot_scale)
		calibration.save(args.output)
		print(f"Saved {args.output}; RMS error {calibration.rms_error_m:.3f} output units using {calibration.point_count} points")
	elif args.command == "pixel":
		calibration = CameraFloorCalibration.load(args.calibration)
		print(calibration.pixel_to_robot(args.pixel))
	else:
		import matplotlib.pyplot as plt
		from PIL import Image, ImageOps
		pixels, robot_xy = load_correspondences(args.points)
		mapped_robot_xy = robot_xy * args.robot_scale
		calibration = fit_homography(pixels, mapped_robot_xy)
		displayed_image = ImageOps.exif_transpose(Image.open(args.image))
		figure, axis = plt.subplots(figsize=(12, 7))
		axis.imshow(displayed_image)
		axis.scatter(pixels[:, 0], pixels[:, 1], c="#f3ad53", s=55, label="calibration pixels")
		for index, (pixel, robot) in enumerate(zip(pixels, mapped_robot_xy)):
			axis.annotate(f"{index}: ({robot[0]:.2f}, {robot[1]:.2f})", pixel, xytext=(5, 5), textcoords="offset points", color="white")
		axis.set_title(f"Camera calibration correspondences; RMS {calibration.rms_error_m:.3f} output units")
		axis.legend()
		axis.set_axis_off()
		figure.tight_layout()
		Path(args.output).parent.mkdir(parents=True, exist_ok=True)
		figure.savefig(args.output, dpi=160)
		print(f"Saved overlay: {args.output}")


if __name__ == "__main__":
	main()
