"""Pure decode of the RealSense wire shapes into Frame inputs (see camera.md).

The 5W head camera publishes aligned depth-to-RGB, so rgb and depth share one grid.
Every function here takes the message fields it needs and returns plain arrays, or
None when the encoding is one we cannot decode — the backend then keeps the blank
frame rather than feeding the session garbage.
"""
from __future__ import annotations

import numpy as np

__all__ = ["decode_depth", "decode_intrinsics", "decode_rgb"]

_RGB_ENCODINGS = {"rgb8": 3, "bgr8": 3}  # channels; bgr8 needs the swap
_DEPTH_ENCODINGS = {"16UC1": (np.uint16, 2, 0.001), "32FC1": (np.float32, 4, 1.0)}


def _as_bytes(data) -> bytes:
    """sensor_msgs/Image.data arrives as bytes on Python 3 rospy, or a list of ints on
    the older shapes; normalize both to the buffer frombuffer wants."""
    return data if isinstance(data, (bytes, bytearray)) else bytes(bytearray(data))


def decode_rgb(msg) -> np.ndarray | None:
    """sensor_msgs/Image (rgb8/bgr8) -> uint8 (h, w, 3) RGB, or None when unknown."""
    channels = _RGB_ENCODINGS.get(msg.encoding)
    if channels is None:
        return None
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    flat = np.frombuffer(_as_bytes(msg.data), dtype=np.uint8)
    rows = flat[:height * step].reshape(height, step)[:, :width * channels]
    pixels = rows.reshape(height, width, channels)
    if msg.encoding == "bgr8":
        pixels = pixels[:, :, ::-1]
    return np.ascontiguousarray(pixels, dtype=np.uint8)


def decode_depth(msg) -> np.ndarray | None:
    """sensor_msgs/Image (16UC1 mm / 32FC1 m) -> float (h, w) metres, NaN where invalid.

    0 mm and non-positive float values mark invalid pixels (RealSense's own convention);
    NaN pixels arrive NaN.
    """
    spec = _DEPTH_ENCODINGS.get(msg.encoding)
    if spec is None:
        return None
    dtype, itemsize, metres = spec
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    flat = np.frombuffer(_as_bytes(msg.data), dtype=dtype)
    depth = flat[:height * step // itemsize].reshape(height, step // itemsize)[:, :width]
    values = depth.astype(float) * metres
    return np.where(np.isfinite(values) & (values > 0), values, np.nan)


def decode_intrinsics(msg) -> np.ndarray | None:
    """sensor_msgs/CameraInfo -> the 3x3 K matrix, or None when uncalibrated (all zeros)."""
    k = np.asarray(msg.k, dtype=float)
    if k.shape != (9,) or not np.isfinite(k).all() or not k.any():
        return None
    return k.reshape(3, 3)
