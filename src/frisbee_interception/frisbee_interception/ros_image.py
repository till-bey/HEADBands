"""sensor_msgs Image / CompressedImage <-> OpenCV BGR without cv_bridge (one dependency less on the robot)."""

from __future__ import annotations

import cv2
import numpy as np

_CHANNELS = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1, "8UC1": 1, "8UC3": 3}


def image_to_bgr(msg) -> np.ndarray:
    enc = msg.encoding
    h, w, step = msg.height, msg.width, msg.step
    buf = np.frombuffer(bytes(msg.data) if not isinstance(msg.data, (bytes, bytearray)) else msg.data, np.uint8)
    if enc in ("yuv422", "uyvy", "yuv422_yuy2", "yuyv"):
        img = buf[: h * step].reshape(h, step)[:, : w * 2].reshape(h, w, 2)
        code = cv2.COLOR_YUV2BGR_UYVY if enc in ("yuv422", "uyvy") else cv2.COLOR_YUV2BGR_YUYV
        return cv2.cvtColor(img, code)
    if enc not in _CHANNELS:
        raise ValueError(f"unsupported image encoding '{enc}'")
    c = _CHANNELS[enc]
    img = buf[: h * step].reshape(h, step)[:, : w * c].reshape(h, w, c) if c > 1 else buf[: h * step].reshape(h, step)[:, :w]
    if enc == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if c == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def compressed_to_bgr(msg) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode CompressedImage")
    return img


def bgr_to_image(img: np.ndarray, header, image_cls):
    msg = image_cls()
    msg.header = header
    msg.height, msg.width = img.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = img.shape[1] * 3
    msg.data = np.ascontiguousarray(img).tobytes()
    return msg


def bgr_to_compressed(img: np.ndarray, header, compressed_cls, quality: int = 80):
    msg = compressed_cls()
    msg.header = header
    msg.format = "jpeg"
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    msg.data = buf.tobytes() if ok else b""
    return msg
