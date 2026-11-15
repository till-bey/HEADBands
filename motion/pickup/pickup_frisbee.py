#!/usr/bin/env python3
"""Train and generate SEDS pickup paths without starting ROS."""

from __future__ import annotations

import argparse
from pathlib import Path

from camera_calibration import CameraFloorCalibration
from pickup_seds import generate_trajectory, train_seds, write_cartesian_csv


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE / "models"


def train_command(args: argparse.Namespace) -> None:
	paths = sorted(Path(args.demo_dir).glob(args.pattern)) if args.demo_dir else [Path(path) for path in args.demo]
	if not paths:
		raise SystemExit("No demonstration CSV files found")
	model = train_seds(paths, n_gaussians=args.n_gaussians)
	output = Path(args.output) if args.output else DEFAULT_MODEL_DIR / f"pickup_seds_g{args.n_gaussians}.npz"
	model.save(output)
	print(f"Trained {model.components}-Gaussian model from {len(paths)} demonstrations")
	print(f"Saved: {output}")


def generate_command(args: argparse.Namespace) -> None:
	if args.target_pixel is not None:
		if args.calibration is None:
			raise SystemExit("--target-pixel requires --calibration")
		target_xy = CameraFloorCalibration.load(args.calibration).pixel_to_robot(args.target_pixel)
	else:
		target_xy = args.target_xy
	trajectory = generate_trajectory(
		args.model,
		current_position=args.current,
		target_xy=target_xy,
		floor_z=args.floor_z,
		duration=args.duration,
		dt=args.dt,
		orientation_rpy=args.orientation,
	)
	output = write_cartesian_csv(args.output, trajectory)
	print(f"Generated {len(trajectory)} waypoints")
	print(f"Start: {trajectory[0, 1:4]}")
	print(f"Target: {trajectory[-1, 1:4]}")
	print(f"Saved: {output}")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	subparsers = parser.add_subparsers(dest="command", required=True)

	train = subparsers.add_parser("train", help="train and save a SEDS/GMM model")	
	train.add_argument("demo", nargs="*", help="trajectory CSV files")
	train.add_argument("--demo-dir", type=Path, help="directory containing demonstration CSVs")
	train.add_argument("--pattern", default="*.csv", help="glob used with --demo-dir")
	train.add_argument("--n-gaussians", type=int, default=8)
	train.add_argument("--output", type=Path)
	train.set_defaults(function=train_command)

	generate = subparsers.add_parser("generate", help="generate a pickup path from a saved model")
	generate.add_argument("--model", type=Path, required=True)
	generate.add_argument("--current", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
	target = generate.add_mutually_exclusive_group(required=True)
	target.add_argument("--target-xy", type=float, nargs=2, metavar=("X", "Y"))
	target.add_argument("--target-pixel", type=float, nargs=2, metavar=("U", "V"))
	generate.add_argument("--calibration", type=Path, help="pixel-to-robot JSON from camera_calibration.py")
	generate.add_argument("--floor-z", type=float, default=0.0)
	generate.add_argument("--duration", type=float, default=4.0)
	generate.add_argument("--dt", type=float, default=0.02)
	generate.add_argument("--orientation", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("ROLL", "PITCH", "YAW"))
	generate.add_argument("--output", type=Path, required=True)
	generate.set_defaults(function=generate_command)

	args = parser.parse_args()
	args.function(args)


if __name__ == "__main__":
	main()
