"""One-shot self-check for the pilot's decision logic and live pipeline.

Run it after touching the loop, the model wrapper, or the input path:

    python -m dbd.pilot_selftest
"""

import numpy as np
from PIL import Image


class _FakeModel:
    """Feeds a fixed sequence of class ids, one per captured frame."""

    backend = "fake"

    def __init__(self, sequence):
        self.sequence = sequence
        self.i = -1
        # Small BGRA frame; the pilot only forwards it to pil_from_bgra.
        self._frame = np.zeros((8, 8, 4), dtype=np.uint8)

    def grab(self, timeout_ms=0):
        self.i += 1
        return self._frame

    def preprocess(self, _):
        return np.zeros((1, 3, 4, 4), dtype=np.float32)

    def frame_changed(self, _):
        return float("inf")  # gate disabled for the logic tests

    def predict_class(self, _):
        return self.sequence[min(self.i, len(self.sequence) - 1)]

    def predict(self, _):
        cid = self.sequence[min(self.i, len(self.sequence) - 1)]
        return cid, f"class{cid}", {}, False

    def cleanup(self):
        pass


class _FakeInput:
    def __init__(self, held=False):
        self.taps = []
        self.held = held

    def tap(self, vk):
        self.taps.append(vk)

    def pressed(self, vk):
        return self.held


def _pilot(sequence, held=False, cooldown=0.0, confirm_frames=1):
    """Build a pilot without touching the screen or the model file."""
    import time

    from dbd.pilot import SkillCheckPilot

    p = object.__new__(SkillCheckPilot)
    p.model_path = "fake"
    p.static_gate_threshold = None
    p.hit_ante_ms = 0.0
    p.rehit_cooldown_s = cooldown
    p.confirm_frames = confirm_frames
    p.pause_key = 0
    p.model = _FakeModel(sequence)
    p.input = _FakeInput(held)
    p._clock = time.perf_counter
    p._hit_latched = False
    p._consecutive_hits = 0
    p._last_class = 0
    p._last_hit_time = None
    p._pause_was_down = False
    p._fps_window_start = 0.0
    p._fps_window_frames = 0
    p.frames_seen = p.frames_classified = p.frames_skipped = p.hits_sent = 0
    p.class_counts = np.zeros(11, dtype=np.int64)
    return p


def check_decisions():
    from dbd.labels import NONE_ID

    # One skill check = one press, however many frames it lasts.
    p = _pilot([1, 1, NONE_ID, 1, 1])
    for _ in range(5):
        p.step(paused=False)
    assert p.hits_sent == 2, f"latch: expected 2, got {p.hits_sent}"

    # Distinct checks are all acted on.
    p = _pilot([1, NONE_ID, 1, NONE_ID, 1])
    for _ in range(5):
        p.step(paused=False)
    assert p.hits_sent == 3, f"re-arm: expected 3, got {p.hits_sent}"

    # Pause toggles once per press, not once per held frame.
    p = _pilot([NONE_ID] * 4, held=True)
    assert sum(p.step(paused=False)["toggle_pause"] for _ in range(4)) == 1

    # While paused nothing is captured and nothing is pressed.
    p = _pilot([1] * 3)
    for _ in range(3):
        p.step(paused=True)
    assert p.hits_sent == 0 and p.frames_seen == 0, "paused loop did work"

    # Out-of-zone classes never press.
    p = _pilot([3, 5, 7, 9, 10])
    for _ in range(5):
        p.step(paused=False)
    assert p.hits_sent == 0, f"false positives: {p.hits_sent}"

    # The cooldown suppresses a second check that follows immediately.
    p = _pilot([1, NONE_ID, 1], cooldown=60.0)
    for _ in range(3):
        p.step(paused=False)
    assert p.hits_sent == 1, f"cooldown: expected 1, got {p.hits_sent}"

    print("[ok] decision logic: latch, re-arm, pause edge, no false positives, cooldown")


def check_hysteresis():
    """confirm_frames must reject blips without swallowing real checks."""
    from dbd.labels import NONE_ID

    # A one-frame blip into a hit class must NOT press.
    p = _pilot([NONE_ID, 1, NONE_ID, NONE_ID], confirm_frames=2)
    for _ in range(4):
        p.step(paused=False)
    assert p.hits_sent == 0, f"blip fired: {p.hits_sent}"

    # Two consecutive frames in the zone must press, once.
    p = _pilot([NONE_ID, 1, 1, 1, NONE_ID], confirm_frames=2)
    for _ in range(5):
        p.step(paused=False)
    assert p.hits_sent == 1, f"sustained check: expected 1, got {p.hits_sent}"

    # The run must survive flipping between two different great classes: the
    # needle crossing a zone edge shows up as a class change, not as an exit.
    p = _pilot([NONE_ID, 8, 1, NONE_ID], confirm_frames=2)
    for _ in range(4):
        p.step(paused=False)
    assert p.hits_sent == 1, f"class flip reset the run: {p.hits_sent}"

    # Leaving the zone resets the run: hits separated by a non-hit class must
    # not accumulate into a confirmation.
    p = _pilot([1, 9, 9, 1], confirm_frames=2)
    for _ in range(4):
        p.step(paused=False)
    assert p.hits_sent == 0, f"counter accumulated across a reset: {p.hits_sent}"

    # confirm_frames=1 still behaves like the old single-frame trigger.
    p = _pilot([NONE_ID, 1, NONE_ID], confirm_frames=1)
    for _ in range(3):
        p.step(paused=False)
    assert p.hits_sent == 1, f"confirm_frames=1 regressed: {p.hits_sent}"

    # A non-hit class inside the sequence resets the counter (out/frontier are
    # not confirmation).
    p = _pilot([1, 9, 1, NONE_ID], confirm_frames=2)
    for _ in range(4):
        p.step(paused=False)
    assert p.hits_sent == 0, f"frontier counted as confirmation: {p.hits_sent}"

    print("[ok] hysteresis: rejects blips, keeps sustained checks, tolerates class flips")


def check_live(n=120):
    from dbd.pilot import SkillCheckPilot, describe_provider
    import time

    print("[..] loading model and driving the live loop")
    ai = SkillCheckPilot("models/model.onnx", monitor_id=1, cpu_threads=8,
                         static_gate_threshold=1.0)
    assert ai.model.input_size == (224, 224), ai.model.input_size
    print(f"     provider={describe_provider(ai.provider)} input_size={ai.model.input_size}")

    for _ in range(5):  # warmup
        ai.step(paused=False)
    t0 = time.perf_counter()
    for _ in range(n):
        ai.step(paused=False)
    dt = (time.perf_counter() - t0) / n
    st = ai.stats()
    print(f"     {dt*1000:.2f} ms/frame ({1/dt:.0f} fps max) | "
          f"classified={st['frames_classified']} skipped={st['skipped_ratio']*100:.0f}%")

    ai.close()
    ai.close()  # cleanup must be idempotent
    print("[ok] live loop runs, model input size honoured, double close is safe")


if __name__ == "__main__":
    check_decisions()
    check_hysteresis()
    check_live()
    print("\nAll pilot checks passed.")
