"""seam: camera decode — the RealSense wire shapes into Frame inputs.

The real robot publishes RealSense topics (assets/robot/kuavo_5w/camera.md):
aligned depth-to-RGB, so rgb and depth share one grid. Decode is pure — message
fields in, arrays out — so it is tested without a ROS daemon.
"""
import numpy as np
import pytest

from homebody.backends.kuavo.camera import decode_depth, decode_intrinsics, decode_rgb


class ImageMsg:
    """The fields decode reads off sensor_msgs/Image."""

    def __init__(self, height, width, encoding, data, step):
        self.height, self.width = height, width
        self.encoding, self.data, self.step = encoding, data, step


def rgb_image(encoding, height=2, width=3, step=None):
    """One small image: pixel (v,u) = (u, v, u+v) with the rows strided as RealSense pads."""
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    for v in range(height):
        for u in range(width):
            pixels[v, u] = (u, v, u + v)
    stride = step if step is not None else width * 3
    rows = np.zeros((height, stride), dtype=np.uint8)
    rows[:, :width * 3] = pixels.reshape(height, width * 3)
    return ImageMsg(height, width, encoding, rows.tobytes(), stride)


def depth_image(encoding, height=2, width=3, step=None, values=None, element=None):
    """One small depth image in millimetres (16UC1) or metres (32FC1), 0 marks invalid.
    An unknown encoding passes its (dtype, itemsize) in `element` so the message builds."""
    dtype, size = element if element is not None else {"16UC1": (np.uint16, 2),
                                                      "32FC1": (np.float32, 4)}[encoding]
    depth = np.zeros((height, width), dtype=dtype)
    if values is not None:
        depth[:, :] = values
    stride = step if step is not None else width * size
    rows = np.zeros((height, stride), dtype=dtype)
    rows[:, :width] = depth
    return ImageMsg(height, width, encoding, rows.tobytes(), stride)


class InfoMsg:
    """The one field decode reads off sensor_msgs/CameraInfo."""

    def __init__(self, k):
        self.k = k


def test_rgb_decodes_to_height_width_3():
    frame = decode_rgb(rgb_image("rgb8"))
    assert frame.shape == (2, 3, 3)
    assert frame[1, 2].tolist() == [2, 1, 3]  # (u, v, u+v)


def test_rgb_bgr8_swaps_the_channels():
    frame = decode_rgb(rgb_image("bgr8"))
    assert frame[1, 2].tolist() == [3, 1, 2]


def test_rgb_skips_row_padding():
    padded = rgb_image("rgb8", step=3 * 3 + 4)
    assert decode_rgb(padded).shape == (2, 3, 3)
    assert decode_rgb(padded)[0, 1].tolist() == rgb_image("rgb8")[0, 1].tolist() \
        if False else decode_rgb(padded)[1, 2].tolist() == [2, 1, 3]


def test_rgb_unknown_encoding_decodes_to_nothing():
    assert decode_rgb(rgb_image("yuv422")) is None


def test_depth_16uc1_is_millimetres_and_zero_is_invalid():
    frame = decode_depth(depth_image("16UC1", values=[1250, 0, 2500]))
    assert frame.shape == (2, 3)
    assert frame[0, 0] == pytest.approx(1.25)
    assert not np.isfinite(frame[0, 1])  # 0 mm -> NaN
    assert frame[0, 2] == pytest.approx(2.5)


def test_depth_32fc1_is_metres_and_nonpositive_is_invalid():
    frame = decode_depth(depth_image("32FC1", values=[1.25, 0.0, np.nan]))
    assert frame[0, 0] == pytest.approx(1.25)
    assert not np.isfinite(frame[0, 1])
    assert not np.isfinite(frame[0, 2])


def test_depth_skips_row_padding():
    frame = decode_depth(depth_image("16UC1", step=3 * 2 + 8))
    assert frame.shape == (2, 3)


def test_depth_unknown_encoding_decodes_to_nothing():
    assert decode_depth(depth_image("8UC1", element=(np.uint8, 1))) is None


def test_intrinsics_reads_the_k_matrix():
    info = InfoMsg([500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0])
    k = decode_intrinsics(info)
    assert k.shape == (3, 3)
    assert k[0, 0] == 500.0 and k[0, 2] == 320.0
    assert k[1, 1] == 500.0 and k[1, 2] == 240.0


def test_intrinsics_of_an_uncalibrated_camera_is_nothing():
    assert decode_intrinsics(InfoMsg([0.0] * 9)) is None
