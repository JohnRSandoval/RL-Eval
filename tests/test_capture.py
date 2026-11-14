from io import BytesIO

import pytest
from PIL import Image

from rleval.capture import (
    CaptureError,
    SyntheticCapture,
    WindowCapture,
    WindowRegion,
    _intersects,
    _on_one_monitor,
)


class FakeNative:
    def __init__(self, *regions):
        self.regions = list(regions)

    def locate(self, *_):
        if len(self.regions) > 1:
            return self.regions.pop(0)
        return self.regions[0]


class FakeMSS:
    def __init__(self, black=False):
        self.monitors = [
            {},
            {"left": -1920, "top": 0, "width": 1920, "height": 1080},
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
        ]
        self.calls = []
        self.closed = False
        self.black = black

    def grab(self, box):
        self.calls.append(box)
        image = Image.new("RGB", (box["width"], box["height"]), "black" if self.black else "red")
        return type("Shot", (), {"size": image.size, "rgb": image.tobytes()})()

    def close(self):
        self.closed = True


def test_only_client_rectangle_is_captured_and_resized():
    target = WindowRegion(2**48, 123, -1500, 100, 800, 450)
    capture = WindowCapture({"max_width": 400})
    capture._native = FakeNative(target)
    capture._mss = FakeMSS()
    image = Image.open(BytesIO(capture.grab()))
    assert image.size == (400, 225)
    assert capture._mss.calls == [{"left": -1500, "top": 100, "width": 800, "height": 450}]


def test_window_move_discards_frame():
    capture = WindowCapture({})
    capture._native = FakeNative(
        WindowRegion(1, 123, 0, 0, 800, 450), WindowRegion(1, 123, 20, 20, 800, 450)
    )
    capture._mss = FakeMSS()
    with pytest.raises(CaptureError, match="moved"):
        capture.grab()


def test_black_frame_has_actionable_error():
    capture = WindowCapture({})
    capture._native = FakeNative(WindowRegion(1, 123, 0, 0, 800, 450))
    capture._mss = FakeMSS(black=True)
    with pytest.raises(CaptureError, match="Borderless"):
        capture.grab()


def test_offscreen_window_is_rejected_before_capture():
    capture = WindowCapture({})
    capture._native = FakeNative(WindowRegion(1, 123, -100, 0, 800, 450))
    capture._mss = FakeMSS()
    with pytest.raises(CaptureError, match="one monitor"):
        capture.grab()
    assert not capture._mss.calls


def test_monitor_geometry_handles_negative_coordinates():
    monitors = [{"left": -1920, "top": -100, "width": 1920, "height": 1080}]
    assert _on_one_monitor(WindowRegion(1, 2, -1800, 0, 800, 450), monitors)
    assert not _on_one_monitor(WindowRegion(1, 2, -1800, -101, 800, 450), monitors)
    assert _intersects((0, 0, 100, 100), (99, 99, 120, 120))
    assert not _intersects((0, 0, 100, 100), (100, 0, 120, 100))


def test_synthetic_frames_are_jpegs_and_change_without_desktop_access():
    capture = SyntheticCapture({"max_width": 480})
    first, second = capture.grab(), capture.grab()
    assert Image.open(BytesIO(first)).size == (480, 270)
    assert first != second
    capture.close()
    with pytest.raises(CaptureError, match="closed"):
        capture.grab()


def test_close_releases_capture_and_prevents_reuse():
    capture = WindowCapture({})
    fake = FakeMSS()
    capture._mss = fake
    capture.close()
    assert fake.closed
    with pytest.raises(CaptureError, match="closed"):
        capture.grab()
