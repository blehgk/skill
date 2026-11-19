"""Screen capture backends.

Two ways to read pixels off the screen, both returning a raw BGRA buffer of the
requested region so the rest of the pipeline is backend-agnostic:

``mss``
    GDI ``BitBlt``. Works everywhere, no extra dependencies. Measured cost on a
    298x298 region: **~9.3 ms p50** (a raw ctypes BitBlt measures the same, so
    this is the GDI/compositor floor, not mss overhead).

``dxcam``
    Windows Desktop Duplication API. Measured **~0.05 ms** per call, which
    brings the whole per-frame pipeline from ~14.2 ms to ~6.7 ms and is what
    lets the bot keep up with a 120 fps game (one frame = 8.33 ms). Requires the
    optional ``dxcam`` + ``comtypes`` packages, and needs the game in borderless
    windowed mode (Desktop Duplication cannot see a true exclusive-fullscreen
    swapchain).

Both backends were verified to produce byte-identical BGRA for the same frame.
"""

import time

import numpy as np

BACKENDS = ("auto", "dxcam", "mss")


class MssGrabber:
    """GDI capture through :mod:`mss`. Always available."""

    name = "mss"

    def __init__(self, region):
        from mss import mss

        self._mss = mss()
        self.region = dict(region)

    def grab(self, timeout_ms=0):
        """Capture the region. Never returns ``None``."""
        shot = self._mss.grab(self.region)
        return np.frombuffer(shot.bgra, dtype=np.uint8).reshape(
            self.region["height"], self.region["width"], 4)

    def close(self):
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
            self._mss = None


class DxcamGrabber:
    """Desktop Duplication capture through :mod:`dxcam`.

    ``grab`` returns ``None`` when the desktop has not produced a new frame
    since the last call, which the caller must handle: there is nothing new to
    classify.
    """

    name = "dxcam"

    def __init__(self, region, output_idx=0):
        import dxcam

        self._dxcam = dxcam
        self._cam = dxcam.create(output_idx=output_idx, output_color="BGRA")
        if self._cam is None:
            raise RuntimeError("dxcam.create() returned None")
        self.region = region
        self._box = (
            region["left"], region["top"],
            region["left"] + region["width"], region["top"] + region["height"],
        )

    def grab(self, timeout_ms=0):
        """Return the newest frame as BGRA, or ``None`` if there is no new one.

        The returned array comes from dxcam's own ring buffer and is reused, so
        callers must not hold on to it across grabs.
        """
        if timeout_ms and timeout_ms > 0:
            deadline = time.perf_counter() + timeout_ms / 1000.0
            while True:
                frame = self._cam.grab(self._box)
                if frame is not None:
                    return frame
                if time.perf_counter() >= deadline:
                    return None
                time.sleep(0.0005)
        return self._cam.grab(self._box)

    def close(self):
        if self._cam is not None:
            cam, self._cam = self._cam, None
            for method in ("stop", "release"):
                try:
                    getattr(cam, method)()
                except Exception:
                    pass


def dxcam_available():
    """Whether the optional Desktop Duplication backend can be used."""
    try:
        import dxcam  # noqa: F401
    except Exception:
        return False
    return True


def make_grabber(region, backend="auto"):
    """Build the best available grabber for ``region``.

    ``auto`` prefers dxcam and silently falls back to mss. Asking for ``dxcam``
    explicitly raises if it cannot be used, so a misconfiguration is visible.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown capture backend {backend!r}; expected one of {BACKENDS}")

    if backend in ("auto", "dxcam"):
        try:
            return DxcamGrabber(region)
        except Exception:
            if backend == "dxcam":
                raise
    return MssGrabber(region)
