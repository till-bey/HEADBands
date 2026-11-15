#!/usr/bin/env python3
"""Stable point-to-point pickup dynamics learned from Cartesian demonstrations.

The demonstrations are CSV files with columns ``t`` or ``seconds_elapsed`` and
``x,y,z`` in the same frame used by cartesian_follow.py. The model learns a
GMM over position relative to each demonstration's endpoint and a stable
velocity field with a shared attractor at zero.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


AXES = ("x", "y", "z")


def load_trajectory_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a trajectory CSV and return strictly increasing time and XYZ arrays."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(row for row in stream if not row.lstrip().startswith("#")))
    if not rows or not all(axis in rows[0] for axis in AXES):
        raise ValueError(f"{path}: expected CSV columns x,y,z and t or seconds_elapsed")
    time_key = "t" if "t" in rows[0] else "seconds_elapsed" if "seconds_elapsed" in rows[0] else None
    if time_key is None:
        time = np.arange(len(rows), dtype=float)
    else:
        time = np.array([float(row[time_key]) for row in rows], dtype=float)
    positions = np.array([[float(row[axis]) for axis in AXES] for row in rows], dtype=float)
    valid = np.isfinite(time) & np.isfinite(positions).all(axis=1)
    time, positions = time[valid], positions[valid]
    if len(time) < 3 or np.any(np.diff(time) <= 0):
        raise ValueError(f"{path}: trajectory needs at least 3 rows with increasing time")
    return time, positions


def _logsumexp(values: np.ndarray, axis: int = 1) -> np.ndarray:
    maximum = values.max(axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def _fit_diagonal_gmm(samples: np.ndarray, components: int, iterations: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(samples) < components:
        raise ValueError(f"need at least {components} samples for the requested GMM")
    means = samples[np.linspace(0, len(samples) - 1, components, dtype=int)].copy()
    variance = np.maximum(np.var(samples, axis=0), 1e-6)
    variances = np.repeat(variance[None, :], components, axis=0)
    weights = np.full(components, 1.0 / components)
    for _ in range(iterations):
        log_probability = -0.5 * (
            ((samples[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :]).sum(axis=2)
            + np.log(variances).sum(axis=1)[None, :]
            + 3 * np.log(2 * np.pi)
        ) + np.log(np.maximum(weights, 1e-12))[None, :]
        responsibilities = np.exp(log_probability - _logsumexp(log_probability, axis=1)[:, None])
        counts = responsibilities.sum(axis=0) + 1e-12
        weights = counts / len(samples)
        means = (responsibilities.T @ samples) / counts[:, None]
        difference = samples[:, None, :] - means[None, :, :]
        variances = np.maximum(
            np.einsum("nk,nkd->kd", responsibilities, difference * difference) / counts[:, None],
            1e-6,
        )
    return weights, means, variances


@dataclass
class SEDSModel:
    """A stable, shared-attractor dynamical system with a learned GMM shape."""

    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    gains: np.ndarray
    global_gain: np.ndarray
    metadata: dict

    @property
    def components(self) -> int:
        return int(len(self.weights))

    def _responsibilities(self, relative_position: np.ndarray) -> np.ndarray:
        difference = relative_position[None, :] - self.means
        log_probability = -0.5 * (
            (difference * difference / self.variances).sum(axis=1)
            + np.log(self.variances).sum(axis=1)
            + 3 * np.log(2 * np.pi)
        ) + np.log(np.maximum(self.weights, 1e-12))
        log_probability -= log_probability.max()
        return np.exp(log_probability) / np.exp(log_probability).sum()

    def vector_field(self, relative_position: Iterable[float]) -> np.ndarray:
        """Return a stable velocity for position relative to the target."""
        relative_position = np.asarray(relative_position, dtype=float)
        responsibilities = self._responsibilities(relative_position)
        local_velocity = -(self.gains * relative_position[None, :])
        blended = responsibilities @ local_velocity
        fallback = -self.global_gain * relative_position
        return 0.85 * blended + 0.15 * fallback

    def generate_trajectory(
        self,
        current_position: Iterable[float],
        target_xy: Iterable[float],
        floor_z: float = 0.0,
        duration: float = 4.0,
        dt: float = 0.02,
        orientation_rpy: Iterable[float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Generate ``t,x,y,z,roll,pitch,yaw`` for a floor pickup target.

        ``current_position`` is XYZ in body_link. ``target_xy`` is the frisbee
        position in the same frame; Z is set to ``floor_z``. The output is
        ready for ``cartesian_follow.py --path`` after reachability checking.
        """
        if duration <= 0 or dt <= 0:
            raise ValueError("duration and dt must be positive")
        current = np.asarray(current_position, dtype=float)
        target = np.array([float(target_xy[0]), float(target_xy[1]), float(floor_z)])
        if current.shape != (3,) or not np.isfinite(current).all():
            raise ValueError("current_position must contain three finite values")
        steps = int(np.ceil(duration / dt)) + 1
        times = np.arange(steps, dtype=float) * dt
        times[-1] = min(times[-1], duration)
        positions = np.empty((steps, 3), dtype=float)
        positions[0] = current
        for index in range(1, steps):
            relative = positions[index - 1] - target
            velocity = self.vector_field(relative)
            positions[index] = positions[index - 1] + dt * velocity
            positions[index, 2] = max(float(floor_z), positions[index, 2])
        positions[-1] = target
        orientation = np.asarray(tuple(orientation_rpy), dtype=float)
        if orientation.shape != (3,):
            raise ValueError("orientation_rpy must contain roll, pitch, yaw")
        return np.column_stack([times, positions, np.repeat(orientation[None, :], steps, axis=0)])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights,
            means=self.means,
            variances=self.variances,
            gains=self.gains,
            global_gain=self.global_gain,
            metadata=np.array(json.dumps(self.metadata)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SEDSModel":
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            return cls(
                weights=archive["weights"],
                means=archive["means"],
                variances=archive["variances"],
                gains=archive["gains"],
                global_gain=archive["global_gain"],
                metadata=metadata,
            )


def train_seds(demonstrations: Iterable[str | Path], n_gaussians: int = 8) -> SEDSModel:
    """Train a stable model from point-to-point XYZ demonstrations."""
    if n_gaussians < 1:
        raise ValueError("n_gaussians must be at least 1")
    demonstrations = list(demonstrations)
    samples = []
    for path in demonstrations:
        time, positions = load_trajectory_csv(path)
        relative = positions - positions[-1]
        velocity = np.gradient(relative, time, axis=0)
        samples.append(np.column_stack([relative, velocity]))
    if not samples:
        raise ValueError("no demonstration CSV files were provided")
    training = np.concatenate(samples, axis=0)
    positions, velocities = training[:, :3], training[:, 3:]
    weights, means, variances = _fit_diagonal_gmm(positions, n_gaussians)
    difference = positions[:, None, :] - means[None, :, :]
    log_probability = -0.5 * (difference * difference / variances[None, :, :]).sum(axis=2) + np.log(weights)[None, :]
    responsibilities = np.exp(log_probability - _logsumexp(log_probability, axis=1)[:, None])
    gains = np.empty((n_gaussians, 3), dtype=float)
    for component in range(n_gaussians):
        numerator = (responsibilities[:, component, None] * positions * velocities).sum(axis=0)
        denominator = (responsibilities[:, component, None] * positions * positions).sum(axis=0) + 1e-9
        gains[component] = np.clip(-numerator / denominator, 0.05, 8.0)
    global_gain = np.clip(np.median(gains, axis=0), 0.05, 8.0)
    return SEDSModel(weights, means, variances, gains, global_gain, {
        "type": "stable_shared_attractor_seds",
        "demonstrations": [str(Path(path)) for path in demonstrations],
        "n_gaussians": n_gaussians,
        "position_dimensions": 3,
        "target_convention": "relative position to demonstration endpoint",
    })


def write_cartesian_csv(path: str | Path, trajectory: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, trajectory, delimiter=",", fmt="%.8f", header="t,x,y,z,roll,pitch,yaw", comments="")
    return path


def generate_trajectory(
    model: SEDSModel | str | Path,
    current_position: Iterable[float],
    target_xy: Iterable[float],
    floor_z: float = 0.0,
    duration: float = 4.0,
    dt: float = 0.02,
    orientation_rpy: Iterable[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Generate a pickup trajectory from a model and a floor XY target."""
    loaded = SEDSModel.load(model) if isinstance(model, (str, Path)) else model
    return loaded.generate_trajectory(current_position, target_xy, floor_z, duration, dt, orientation_rpy)
