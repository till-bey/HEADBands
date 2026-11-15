"""Local media browsing and random-access OpenCV video frame utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpeg", ".mpg"})


def list_media(directory: Path, extensions: frozenset[str]) -> list[Path]:
    """Find supported files in a local folder (including subfolders)."""
    if not directory.is_dir():
        return []
    return sorted(
        (p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in extensions),
        key=lambda p: str(p.relative_to(directory)).casefold(),
    )


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int


def video_info(path: Path) -> VideoInfo | None:
    """Read video metadata without decoding the whole video."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if count <= 0:
            return None  # Codec/container does not support reliable frame navigation.
        return VideoInfo(count, fps, width, height)
    finally:
        cap.release()


def read_video_frame(path: Path, index: int) -> np.ndarray | None:
    """Seek to a zero-based frame index, decode one frame, then release the reader."""
    if index < 0:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()
