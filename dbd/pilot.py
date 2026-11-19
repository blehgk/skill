"""Screen-capture bot loop: capture, classify, and press the skill-check key.

This module owns the whole decision path. It is deliberately free of any UI
framework so that it can be driven by a Gradio generator, a test harness, or a
plain script.

Timing model
------------
A skill check hit is a hard real-time event: everything between *observing* the
frame and *sending* the key press eats into the reaction budget, so the loop
keeps that path short.

1. capture the crop around the screen centre (see :mod:`dbd.capture`),
2. normalise it and run the ONNX classifier,
3. on a firing class, press and release the key in a single ``SendInput`` batch.

The capture backend dominates the frame cost (GDI ~9.3 ms vs Desktop
Duplication ~0.05 ms), so a 120 fps game calls for the dxcam backend to keep
pace with its 8.33 ms frame interval.

Two guards avoid wasted work without delaying a hit:

* **Static-frame gate** - when the crop is pixel-identical to the previous one
  *and* nothing is currently on screen, classification cannot change the
  decision, so the frame is skipped. The gate disables itself automatically
  whenever the previous frame classified as a skill check.
* **One-shot latch** - a hit class is acted on once per appearance and re-armed
  only after a frame that is not a hit class, so a needle resting inside the
  great zone cannot produce repeated key presses. A cooldown covers the case
  where two distinct skill checks appear back to back.
"""

import os
import threading
import time

import numpy as np

from dbd.AI_model import DEFAULT_THREADS, AI_model, pil_from_bgra
from dbd.input_controller import InputController
from dbd.labels import ANTE_FRONTIER_ID, NONE_ID, NUM_CLASSES, label_for
from dbd.utils.directkeys import SPACE, VK_XBUTTON2
from dbd.utils.monitor import get_monitor_attributes

#: Keys of the telemetry sample returned by :meth:`SkillCheckPilot.step`.
SAMPLE_FIELDS = ("fps", "class_id", "desc", "hit", "frame", "paused", "toggle_pause")


class PilotError(RuntimeError):
    """Raised when the pilot cannot start (missing model, bad monitor, ...)."""


class SkillCheckPilot:
    """Drives the capture -> classify -> press loop for one monitor.

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` model.
    monitor_id:
        1-based ``mss`` monitor index.
    crop_size:
        Edge of the square region sampled at the centre of the monitor. The
        model was trained on 224x224 crops from a 1080p screen, and that scale
        is preserved on other resolutions.
    hit_ante_ms:
        Extra delay applied to the ante-frontier class, whose needle has not
        reached the great zone yet.
    cpu_threads:
        Intra-op thread count for the ONNX session. ``None`` uses the measured
        optimum from :data:`dbd.AI_model.DEFAULT_THREADS` (oversubscribing is
        much slower, so it is never left to ONNX Runtime's own default).
    capture_backend:
        ``"auto"`` prefers Desktop Duplication (dxcam) and falls back to GDI
        (mss); ``"dxcam"`` or ``"mss"`` force one. This dominates frame cost:
        mss ~9.3 ms vs dxcam ~0.05 ms for the same 298x298 region.
    hold_ms:
        How long the key stays down. ``0`` batches press and release together.
    static_gate_threshold:
        Mean absolute pixel difference below which two consecutive frames count
        as identical. ``None`` disables the gate.
    rehit_cooldown_s:
        Minimum delay between two skill checks the bot reacts to.
    confirm_frames:
        Consecutive frames that must land in the zone before the key is pressed.
        ``1`` (default) fires on the first frame seen in the zone. ``2+`` rejects
        a one-frame blip but costs a whole frame of latency, which is expensive
        here: a generator Great zone is 3% of the dial over a 1.1 s check, i.e.
        **33 ms**, so one extra frame at 120 fps eats 25% of the entire window
        and drops the usable window from 83% to 58% of the zone. Only raise this
        if the model is observed flickering, and expect fewer Greats.
    pause_key:
        Virtual key code used as the pause toggle.
    clock:
        Injectable time source (``perf_counter``-like) to make the loop testable.
    """

    def __init__(
        self,
        model_path="models/model.onnx",
        monitor_id=1,
        crop_size=224,
        hit_ante_ms=3,
        cpu_threads=None,
        capture_backend="auto",
        hold_ms=0.0,
        static_gate_threshold=1.0,
        rehit_cooldown_s=0.35,
        confirm_frames=1,
        pause_key=VK_XBUTTON2,
        clock=time.perf_counter,
    ):
        if not os.path.isfile(model_path):
            raise PilotError(f"AI model file not found: {model_path}")

        self.model_path = model_path
        self.monitor_id = monitor_id
        self.crop_size = int(crop_size)
        self.hit_ante_ms = max(0.0, float(hit_ante_ms))
        self.static_gate_threshold = static_gate_threshold
        self.rehit_cooldown_s = max(0.0, float(rehit_cooldown_s))
        #: Consecutive classified frames required in the zone before pressing.
        #: 1 reproduces the old single-frame trigger; 2+ rejects one-frame blips
        #: at the cost of roughly one frame of extra latency.
        self.confirm_frames = max(1, int(confirm_frames))
        self.pause_key = pause_key
        self._clock = clock

        self.region = get_monitor_attributes(monitor_id, crop_size=self.crop_size)
        self.input = InputController(hold_ms=hold_ms)
        self.model = AI_model(
            model_path=model_path,
            nb_cpu_threads=cpu_threads,
            monitor_id=monitor_id,
            crop_size=self.crop_size,
            backend=capture_backend,
        )

        # Telemetry
        self.frames_seen = 0
        self.frames_classified = 0
        self.frames_skipped = 0
        self.hits_sent = 0
        self.class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

        # Loop state
        self._hit_latched = False
        self._consecutive_hits = 0
        self._last_class = NONE_ID
        self._last_hit_time = None
        self._pause_was_down = False
        self._stop = threading.Event()
        self._fps_window_start = self._clock()
        self._fps_window_frames = 0

    # -- lifecycle ---------------------------------------------------------
    @property
    def provider(self) -> str:
        """Execution provider actually in use."""
        return self.model.check_provider()

    @property
    def capture_backend(self) -> str:
        """Capture backend actually in use (``"mss"`` or ``"dxcam"``)."""
        return self.model.backend

    def stop(self) -> None:
        """Ask the loop to finish and release model resources."""
        self._stop.set()
        self.close()

    def close(self) -> None:
        """Release model resources. Safe to call more than once."""
        self.model.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -- perception --------------------------------------------------------
    def _is_static(self, frame) -> bool:
        """Whether the crop is unchanged and nothing is on screen.

        Uses a sub-sampled signature of the raw BGRA buffer, so the gate never
        pays for a full-resolution comparison.
        """
        if self.static_gate_threshold is None:
            return False
        # Never skip while a skill check is being tracked: the gate would drop
        # the frames the hysteresis depends on.
        if self._last_class != NONE_ID or self._consecutive_hits > 0:
            return False
        return self.model.frame_changed(frame) < self.static_gate_threshold

    # -- decision ----------------------------------------------------------
    def _fire(self, class_id: int) -> None:
        """Press the skill-check key for ``class_id``."""
        if class_id == ANTE_FRONTIER_ID and self.hit_ante_ms:
            time.sleep(self.hit_ante_ms / 1000.0)
        self.input.tap(SPACE)
        self.hits_sent += 1
        self._last_hit_time = self._clock()

    def _cooldown_active(self) -> bool:
        if self._last_hit_time is None:
            return False
        return (self._clock() - self._last_hit_time) < self.rehit_cooldown_s

    # -- main loop ---------------------------------------------------------
    def step(self, paused: bool) -> dict:
        """Run one loop iteration and return a telemetry sample.

        Keys are :data:`SAMPLE_FIELDS`. ``frame`` carries the last classified
        PIL image (``None`` when the iteration was skipped) and ``toggle_pause``
        is set on the press edge of the pause button.
        """
        sample = {
            "fps": None,
            "class_id": self._last_class,
            "desc": label_for(self._last_class).desc,
            "hit": False,
            "frame": None,
            "paused": paused,
            "toggle_pause": False,
        }

        # Edge detection: toggle on the press transition only.
        pause_down = self.input.pressed(self.pause_key)
        if pause_down and not self._pause_was_down:
            sample["toggle_pause"] = True
        self._pause_was_down = pause_down

        if paused:
            # Keep the FPS window fresh so statistics restart after a pause.
            self._fps_window_start = self._clock()
            self._fps_window_frames = 0
            return sample

        frame = self.model.grab()
        if frame is None:
            # Desktop Duplication found no new frame since the last call, so
            # there is nothing new to classify. Not an error, and not a skip.
            return sample

        self.frames_seen += 1

        if self._is_static(frame):
            self.frames_skipped += 1
            return sample

        tensor = self.model.preprocess(frame)
        self.frames_classified += 1
        # predict_class only returns the winning class: the loop never reads the
        # softmax probabilities, and computing them every frame is pure overhead.
        class_id = self.model.predict_class(tensor)
        self.class_counts[class_id] += 1
        self._last_class = class_id
        label = label_for(class_id)
        sample.update(class_id=class_id, desc=label.desc,
                      frame=pil_from_bgra(frame))

        if label.hit:
            # Temporal hysteresis: a single frame in the zone is not enough when
            # `confirm_frames > 1`. A one-frame blip (the model flickering into a
            # hit class and straight back out) is rejected. Any hit-enabled class
            # counts as "in the zone", so a flicker between two great classes
            # does not reset the run.
            self._consecutive_hits += 1
            if (self._consecutive_hits >= self.confirm_frames
                    and not self._hit_latched
                    and not self._cooldown_active()):
                self._hit_latched = True
                sample["hit"] = True
                self._fire(class_id)
        else:
            self._hit_latched = False
            self._consecutive_hits = 0

        self._fps_window_frames += 1
        elapsed = self._clock() - self._fps_window_start
        if elapsed >= 1.0:
            sample["fps"] = round(self._fps_window_frames / elapsed, 1)
            self._fps_window_start = self._clock()
            self._fps_window_frames = 0

        return sample

    def stopped(self) -> bool:
        """Whether :meth:`stop` has been requested."""
        return self._stop.is_set()

    def stats(self) -> dict:
        """Aggregate counters, useful for a status panel or a test."""
        total = max(1, self.frames_seen)
        return {
            "frames_seen": self.frames_seen,
            "frames_classified": self.frames_classified,
            "frames_skipped": self.frames_skipped,
            "skipped_ratio": self.frames_skipped / total,
            "hits_sent": self.hits_sent,
            "class_counts": self.class_counts.copy(),
        }


def describe_provider(provider: str) -> str:
    """Human-readable note about the execution provider in use."""
    if provider != "CPUExecutionProvider":
        # Not expected: this build is CPU-only by design.
        return f"Running AI model on {provider} (unexpected)"
    return f"Running AI model on CPU ({DEFAULT_THREADS} threads)"
