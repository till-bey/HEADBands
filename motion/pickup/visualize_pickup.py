#!/usr/bin/env python3
"""Visualize pickup demonstrations and an optional generated SEDS path."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pickup_seds import SEDSModel, load_trajectory_csv


COLORS = ("#73d0c0", "#ef8f75", "#c59cf2", "#f3ad53", "#8fc7f2")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--demo-dir", type=Path, required=True)
	parser.add_argument("--pattern", default="*.csv")
	parser.add_argument("--model", type=Path)
	parser.add_argument("--current", type=float, nargs=3, metavar=("X", "Y", "Z"))
	parser.add_argument("--target-xy", type=float, nargs=2, metavar=("X", "Y"))
	parser.add_argument("--floor-z", type=float, default=0.0)
	parser.add_argument("--duration", type=float, default=4.0)
	parser.add_argument("--dt", type=float, default=0.02)
	parser.add_argument("--save", type=Path, help="save the figure instead of only showing it")
	args = parser.parse_args()

	files = sorted(args.demo_dir.glob(args.pattern))
	if not files:
		parser.error(f"No demonstration CSV files found in {args.demo_dir}")

	figure = plt.figure(figsize=(10, 7))
	axis = figure.add_subplot(111, projection="3d")
	for index, path in enumerate(files):
		_, positions = load_trajectory_csv(path)
		axis.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=COLORS[index % len(COLORS)], alpha=0.45, label=path.stem)

	if args.model:
		if args.current is None or args.target_xy is None:
			parser.error("--model requires --current X Y Z and --target-xy X Y")
		model = SEDSModel.load(args.model)
		trajectory = model.generate_trajectory(args.current, args.target_xy, args.floor_z, args.duration, args.dt)
		axis.plot(trajectory[:, 1], trajectory[:, 2], trajectory[:, 3], color="#ffffff", linewidth=2.8, label="SEDS pickup")
		axis.scatter(*trajectory[0, 1:4], color="#f3ad53", s=55, label="current gripper")
		axis.scatter(args.target_xy[0], args.target_xy[1], args.floor_z, color="#ffffff", marker="x", s=80, label="frisbee target")

	axis.set_xlabel("X forward (m)")
	axis.set_ylabel("Y left (m)")
	axis.set_zlabel("Z up (m)")
	axis.set_title("Frisbee pickup demonstrations and stable generated path")
	axis.legend(loc="best", fontsize=8)
	axis.set_box_aspect((1, 1, 1))
	figure.tight_layout()
	if args.save:
		args.save.parent.mkdir(parents=True, exist_ok=True)
		figure.savefig(args.save, dpi=160)
		print(f"Saved figure: {args.save}")
	else:
		plt.show()


if __name__ == "__main__":
	main()
