"""Capture only the visible Rocket League client area, never a desktop fallback.

Create/use a WindowCapture in one capture thread. Native/MSS handles are initialized
lazily on the first grab, not on the thread that constructs the object.
"""

import ctypes
import math
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from io import BytesIO

import psutil
from PIL import Image, ImageDraw, ImageFont


class CaptureError(RuntimeError):
    """The game cannot currently be captured without including other windows."""


@dataclass(frozen=True)
class WindowRegion:
    hwnd: int
    pid: int
    left: int
    top: int
    width: int
    height: int

    def box(self) -> dict:
        return {k: getattr(self, k) for k in ("left", "top", "width", "height")}


def _intersects(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _on_one_monitor(region: WindowRegion, monitors: list[dict]) -> bool:
    """Do not capture offscreen pixels or gaps between displays."""
    return any(
        region.left >= m["left"]
        and region.top >= m["top"]
        and region.left + region.width <= m["left"] + m["width"]
        and region.top + region.height <= m["top"] + m["height"]
        for m in monitors
    )


def _jpeg(image: Image.Image, max_width: int, quality: int) -> bytes:
    image = image.convert("RGB")
    if image.width > max_width:
        height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


class _Windows:
    """Small Win32 surface with explicitly pointer-sized HWND declarations."""

    def __init__(self):
        if sys.platform != "win32":
            raise CaptureError(
                "Live screenshot capture requires Windows. Use synthetic capture for a smoke test."
            )
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
        self.callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        signatures = {
            "EnumWindows": ([self.callback_type, wintypes.LPARAM], wintypes.BOOL),
            "GetWindowTextLengthW": ([wintypes.HWND], ctypes.c_int),
            "GetWindowTextW": ([wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
            "GetWindowThreadProcessId": (
                [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)],
                wintypes.DWORD,
            ),
            "IsWindowVisible": ([wintypes.HWND], wintypes.BOOL),
            "IsIconic": ([wintypes.HWND], wintypes.BOOL),
            "GetClientRect": ([wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
            "GetWindowRect": ([wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
            "ClientToScreen": ([wintypes.HWND, ctypes.POINTER(wintypes.POINT)], wintypes.BOOL),
            "GetForegroundWindow": ([], wintypes.HWND),
            "GetWindow": ([wintypes.HWND, wintypes.UINT], wintypes.HWND),
            "SetThreadDpiAwarenessContext": ([ctypes.c_void_p], ctypes.c_void_p),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.user32, name)
            fn.argtypes = args
            fn.restype = result
        self.dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long

    def _pid(self, hwnd: int) -> int:
        pid = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value

    def _cloaked(self, hwnd: int) -> bool:
        value = wintypes.DWORD()
        ok = self.dwmapi.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(value), ctypes.sizeof(value))
        return ok == 0 and bool(value.value)

    def _title(self, hwnd: int) -> str:
        text = ctypes.create_unicode_buffer(self.user32.GetWindowTextLengthW(hwnd) + 1)
        self.user32.GetWindowTextW(hwnd, text, len(text))
        return text.value

    def locate(self, title: str, require_foreground: bool) -> WindowRegion:
        # PER_MONITOR_AWARE_V2 prevents scaling client coordinates on mixed-DPI displays.
        if not self.user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4)):
            raise CaptureError(
                "Could not set DPI-aware capture coordinates on this Windows version."
            )
        matches = []

        @self.callback_type
        def callback(hwnd, _):
            if (
                self.user32.IsWindowVisible(hwnd)
                and title.casefold() in self._title(hwnd).casefold()
            ):
                try:
                    pid = self._pid(hwnd)
                    if psutil.Process(pid).name().casefold() == "rocketleague.exe":
                        matches.append((hwnd, pid))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return True

        if not self.user32.EnumWindows(callback, 0):
            raise CaptureError("Could not enumerate game windows.")
        if not matches:
            raise CaptureError(
                "No visible RocketLeague.exe window found. Start the offline match first."
            )
        foreground = self.user32.GetForegroundWindow()
        selected = [w for w in matches if w[0] == foreground]
        if not selected and len(matches) != 1:
            raise CaptureError(
                "Multiple Rocket League windows found. Focus the game to select one."
            )
        hwnd, pid = selected[0] if selected else matches[0]
        if self.user32.IsIconic(hwnd) or self._cloaked(hwnd):
            raise CaptureError(
                "Rocket League is minimized or on another desktop. Restore the game window."
            )
        if require_foreground and hwnd != foreground:
            raise CaptureError(
                "Rocket League lost focus. Bring the game to the foreground to resume capture."
            )
        client = wintypes.RECT()
        origin = wintypes.POINT(0, 0)
        if not self.user32.GetClientRect(
            hwnd, ctypes.byref(client)
        ) or not self.user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            raise CaptureError("Could not read Rocket League's client-area coordinates.")
        width, height = client.right - client.left, client.bottom - client.top
        if width < 64 or height < 64:
            raise CaptureError(
                "Rocket League's client area is too small. Restore or resize the game."
            )
        region = WindowRegion(int(hwnd), pid, origin.x, origin.y, width, height)
        self._check_occlusion(region)
        return region

    def _check_occlusion(self, region: WindowRegion):
        target = (region.left, region.top, region.left + region.width, region.top + region.height)
        hwnd = self.user32.GetWindow(region.hwnd, 3)  # GW_HWNDPREV: next window above us.
        visited = set()
        while hwnd:
            if hwnd in visited or len(visited) > 512:
                raise CaptureError(
                    "Window order changed during capture. Retry with the game unobscured."
                )
            visited.add(hwnd)
            if (
                self.user32.IsWindowVisible(hwnd)
                and not self.user32.IsIconic(hwnd)
                and not self._cloaked(hwnd)
                and self._pid(hwnd) != region.pid
            ):
                rect = wintypes.RECT()
                if not self.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    raise CaptureError("Unable to verify that the game window is unobscured.")
                if _intersects(target, (rect.left, rect.top, rect.right, rect.bottom)):
                    raise CaptureError(
                        "Another window overlaps Rocket League. Close overlays or move them away from the game."
                    )
            hwnd = self.user32.GetWindow(hwnd, 3)


class WindowCapture:
    def __init__(self, config: dict):
        self.config = dict(config)
        self._native = None
        self._mss = None
        self._thread_id = None
        self._closed = False

    def grab(self) -> bytes:
        if self._closed:
            raise CaptureError("The screenshot capture has been closed.")
        thread_id = threading.get_ident()
        if self._thread_id is not None and self._thread_id != thread_id:
            raise CaptureError("Use each WindowCapture from one thread only.")
        self._thread_id = thread_id
        if self._native is None:
            self._native = _Windows()
        title = self.config.get("window_title", "Rocket League")
        require_foreground = self.config.get("require_foreground", True)
        region = self._native.locate(title, require_foreground)
        try:
            if self._mss is None:
                import mss

                self._mss = mss.mss()
            if not _on_one_monitor(region, self._mss.monitors[1:]):
                raise CaptureError("Keep the entire Rocket League window inside one monitor.")
            shot = self._mss.grab(region.box())
            # Discard frames when the window moved, lost focus, or became occluded mid-capture.
            if self._native.locate(title, require_foreground) != region:
                raise CaptureError("Rocket League moved during capture; the frame was discarded.")
            image = Image.frombytes("RGB", shot.size, shot.rgb)
            if max(high for _, high in image.getextrema()) < 3:
                raise CaptureError(
                    "Captured a black game frame. Use Windowed or Borderless display mode."
                )
            return _jpeg(
                image, self.config.get("max_width", 960), self.config.get("jpeg_quality", 80)
            )
        except CaptureError:
            raise
        except Exception as exc:
            raise CaptureError(
                f"Game screenshot failed ({type(exc).__name__}). Use Windowed or Borderless mode."
            ) from exc

    def close(self):
        if self._mss is not None:
            self._mss.close()
            self._mss = None
        self._closed = True


class SyntheticCapture:
    """A visibly artificial changing scene for tests; this never observes the desktop."""

    def __init__(self, config: dict):
        self.config = dict(config)
        self._start = time.monotonic()
        self._frame = 0
        self._closed = False

    def grab(self) -> bytes:
        if self._closed:
            raise CaptureError("The synthetic screenshot capture has been closed.")
        self._frame += 1
        image = Image.new("RGB", (960, 540), "#10212a")
        draw = ImageDraw.Draw(image)
        draw.rectangle((60, 90, 900, 470), fill="#254937", outline="#c1d1bd", width=3)
        draw.line((480, 90, 480, 470), fill="#c1d1bd", width=2)
        draw.ellipse((405, 205, 555, 355), outline="#c1d1bd", width=2)
        draw.rectangle((60, 215, 110, 345), fill="#3174bd")
        draw.rectangle((850, 215, 900, 345), fill="#cd7332")
        phase = self._frame * 0.12
        x, y = int(480 + 180 * math.sin(phase)), int(280 + 100 * math.cos(phase))
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill="#eee8d7")
        draw.rounded_rectangle((300, 330, 345, 355), radius=5, fill="#4596ee")
        font = ImageFont.load_default(size=24)
        small = ImageFont.load_default(size=17)
        draw.text((60, 25), "SYNTHETIC SMOKE TEST", fill="#ffdb80", font=font)
        draw.text(
            (60, 490),
            f"Frame {self._frame} | {time.monotonic() - self._start:.2f}s | Not game footage",
            fill="white",
            font=small,
        )
        return _jpeg(image, self.config.get("max_width", 960), self.config.get("jpeg_quality", 80))

    def close(self):
        self._closed = True
