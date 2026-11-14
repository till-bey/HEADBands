"""Generate synthetic, illustrative sample images; no photos or training dataset."""
from pathlib import Path

import cv2
import numpy as np

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def field(seed: int = 10, width: int = 800, height: int = 480) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.empty((height, width, 3), np.uint8)
    horizon = 130
    for y in range(height):
        if y < horizon:
            t = y / horizon
            img[y] = (int(230 - 27 * t), int(204 - 12 * t), int(155 + 8 * t))
        else:
            t = (y - horizon) / (height - horizon)
            img[y] = (int(58 - 18 * t), int(117 - 20 * t), int(55 - 10 * t))
    noise = rng.normal(0, 7.0, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for _ in range(1400):
        x = int(rng.integers(0, width))
        y = int(rng.integers(horizon, height))
        length = int(rng.integers(1, 6))
        col = (int(rng.integers(35, 88)), int(rng.integers(100, 165)), int(rng.integers(32, 85)))
        cv2.line(img, (x, y), (x + int(rng.integers(-2, 2)), y - length), col, 1, cv2.LINE_AA)
    return img


def add_disc(img: np.ndarray, center: tuple[int, int], axes: tuple[int, int], angle: float) -> np.ndarray:
    result = img.copy()
    cx, cy = center
    cv2.ellipse(result, (cx + 7, cy + 10), axes, angle, 0, 360, (28, 62, 30), -1, cv2.LINE_AA)
    cv2.ellipse(result, center, axes, angle, 0, 360, (134, 150, 144), -1, cv2.LINE_AA)
    inset = (max(2, axes[0] - 5), max(2, axes[1] - 4))
    cv2.ellipse(result, center, inset, angle, 0, 360, (241, 243, 238), -1, cv2.LINE_AA)
    cv2.ellipse(result, center, inset, angle, 0, 360, (205, 211, 211), 2, cv2.LINE_AA)
    return result


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    samples = {
        "01_round_white_disc.png": add_disc(field(3), (390, 330), (57, 51), -10),
        "02_angled_white_disc.png": add_disc(field(4), (415, 330), (87, 31), 19),
        "03_empty_field.png": field(5),
        "04_two_white_discs.png": add_disc(add_disc(field(6), (270, 310), (51, 33), -22), (530, 360), (70, 30), 12),
    }
    for filename, image in samples.items():
        outpath = SAMPLES / filename
        assert cv2.imwrite(str(outpath), image)
        print(f"Wrote {outpath}")

    movie_path = SAMPLES / "05_moving_disc_demo.mp4"
    writer = cv2.VideoWriter(str(movie_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (800, 480))
    if not writer.isOpened():
        raise RuntimeError("Unable to create sample MP4 (OpenCV FFmpeg is needed)")
    background = field(15)
    for n in range(45):
        x = int(205 + n * 8)
        y = int(320 + 12 * np.sin(n / 8))
        writer.write(add_disc(background, (x, y), (60, 35), 12))
    writer.release()
    print(f"Wrote {movie_path}")


if __name__ == "__main__":
    main()
