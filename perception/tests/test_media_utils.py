"""Tests for local Dataset directory discovery and random-access video frames."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from website.media_utils import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, list_media, read_video_frame, video_info

ROOT = Path(__file__).resolve().parents[1]


def test_images_and_videos_are_listed():
    images = list_media(ROOT / "Dataset" / "Images", IMAGE_EXTENSIONS)
    videos = list_media(ROOT / "Dataset" / "Videos", VIDEO_EXTENSIONS)
    assert len(images) >= 2
    assert len(videos) >= 1
    assert all(p.is_file() for p in images + videos)


def test_extensions_and_nested_folders(tmp_path):
    child = tmp_path / "subfolder"
    child.mkdir()
    (child / "sample.JPG").touch()
    (child / "note.txt").touch()
    assert [p.name for p in list_media(tmp_path, IMAGE_EXTENSIONS)] == ["sample.JPG"]
    assert list_media(tmp_path / "missing", VIDEO_EXTENSIONS) == []


def test_video_exact_frame_seek():
    path = ROOT / "samples" / "05_moving_disc_demo.mp4"
    info = video_info(path)
    assert info is not None
    assert info.frame_count == 45
    first = read_video_frame(path, 0)
    middle = read_video_frame(path, 20)
    last = read_video_frame(path, 44)
    assert first is not None and middle is not None and last is not None
    assert first.shape == middle.shape == last.shape
    # Synthetic white disc moves between frames. This confirms random access isn't stuck at frame 0.
    assert float(np.mean(cv2.absdiff(first, middle))) > 0.1
    assert float(np.mean(cv2.absdiff(middle, last))) > 0.1
    assert read_video_frame(path, -1) is None
