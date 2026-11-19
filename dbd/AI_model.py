"""ONNX Runtime wrapper (CPU only) plus frame preprocessing.

This project deliberately targets CPU inference: the model is a small 11-class
classifier (224x224 input, ~2.7 ms per frame on 8 threads) and a GPU runtime
would add CUDA / TensorRT / torch deployment weight for no measurable gain in
the real loop.

Capture is delegated to :mod:`dbd.capture`, which offers a GDI backend (``mss``)
and a Desktop Duplication backend (``dxcam``). The capture backend dominates
per-frame cost, so it is the knob that matters:

======================  ==========  ==========
stage                   mss         dxcam
======================  ==========  ==========
capture (298x298)       ~9.3 ms     ~0.05 ms
preprocess              ~1.4 ms     ~1.4 ms
inference (8 threads)   ~2.7 ms     ~2.7 ms
**full frame**          **~14 ms**  **~6.7 ms**
======================  ==========  ==========

A 120 fps game hands over a frame every 8.33 ms, so only the dxcam path keeps
pace with it.
"""

import numpy as np
import onnxruntime as ort
from PIL import Image

from dbd.capture import make_grabber
from dbd.labels import LABELS, NONE_ID, label_for
from dbd.utils.monitor import get_monitor_attributes

#: Intra-op thread count. Inference time is a U-curve in this value: measured
#: 1->4.59 ms, 2->3.41, 4->2.93, 8->2.67 (best), 16->3.79, 32->13.26 ms.
#: Letting ONNX Runtime choose gave 7.36 ms, so it is worth pinning explicitly.
DEFAULT_THREADS = 8

#: ImageNet statistics the model was trained with.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: Stride used by the cheap frame-difference signature.
_SIGNATURE_STEP = 4


def suggested_threads():
    """Thread count to use by default: :data:`DEFAULT_THREADS`, capped to cores.

    Oversubscribing hurts badly (32 threads measured 5x slower than 8), so this
    never asks for more threads than the machine has cores.
    """
    import os

    return max(1, min(DEFAULT_THREADS, os.cpu_count() or 1))


def pil_from_bgra(frame):
    """BGRA ndarray -> PIL RGB image at the native frame size.

    The channel swap is materialised into a contiguous array before handing it
    to Pillow. A zero-copy strided view (``frame[:, :, 2::-1]``) looks cheaper
    but measures 2.28x slower (0.86 ms vs 0.38 ms for 298x298), because Pillow
    then copies element by element instead of a bulk memcpy.
    """
    rgb = np.ascontiguousarray(frame[:, :, 2::-1])
    return Image.fromarray(rgb)


class AI_model:
    """Runs the skill-check classifier on CPU and captures the screen region.

    ``pred_dict`` and :mod:`dbd.labels` describe the same 11 classes;
    :meth:`predict_label` returns the richer :class:`dbd.labels.Label` object.
    """

    #: Legacy id -> (description, is_hit) mapping, kept for compatibility.
    pred_dict = {
        label.id: {"desc": label.desc, "hit": label.hit} for label in LABELS
    }

    def __init__(self, model_path="model.onnx", nb_cpu_threads=None, monitor_id=1,
                 crop_size=224, backend="auto"):
        import os

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.model_path = model_path
        self.nb_cpu_threads = nb_cpu_threads or suggested_threads()
        self.crop_size = int(crop_size)
        self.region = get_monitor_attributes(monitor_id, crop_size=self.crop_size)

        #: The raw mss-style region dict, handy for the UI preview.
        self.monitor = dict(self.region)

        self.grabber = make_grabber(self.region, backend=backend)
        self.backend = self.grabber.name

        self.ort_session = None
        self.input_name = None
        self.input_size = None  # (height, width) declared by the model

        self._prev_signature = None
        self._load_onnx()

    # -- model -------------------------------------------------------------
    def _load_onnx(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.nb_cpu_threads
        options.inter_op_num_threads = self.nb_cpu_threads
        # Measured: ENABLE_ALL 2.91 ms vs ENABLE_BASIC 6.70 ms vs DISABLE_ALL
        # 6.77 ms. This is also the ORT default; it is set explicitly so the
        # gain cannot be lost to an environment change.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.ort_session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )

        model_input = self.ort_session.get_inputs()[0]
        self.input_name = model_input.name

        # Trust the model's declared spatial size rather than assuming it
        # matches the monitor crop; a 1440p screen yields a 298x298 crop.
        shape = model_input.shape
        if len(shape) == 4 and all(isinstance(dim, int) for dim in shape[2:]):
            self.input_size = (int(shape[2]), int(shape[3]))

    def check_provider(self):
        """Execution provider in use. Always CPU for this project."""
        return self.ort_session.get_providers()[0]

    # -- capture -----------------------------------------------------------
    def grab(self, timeout_ms=0):
        """Capture the region as a BGRA uint8 array, or ``None`` if no new frame.

        Only the Desktop Duplication backend can return ``None``.
        """
        return self.grabber.grab(timeout_ms=timeout_ms)

    def preprocess(self, frame):
        """Normalise a BGRA frame into the model's NCHW float32 input tensor."""
        pil = pil_from_bgra(frame)
        target = self.input_size or (self.crop_size, self.crop_size)
        if (pil.height, pil.width) != target:
            pil = pil.resize((target[1], target[0]), Image.Resampling.BICUBIC)

        img = np.asarray(pil, dtype=np.float32)
        img *= np.float32(1.0 / 255.0)
        img -= MEAN
        img /= STD
        return np.ascontiguousarray(img.transpose(2, 0, 1))[None]

    def frame_signature(self, frame):
        """Cheap sub-sampled signature of a frame, for the static-frame gate.

        Compares a strided slice instead of every pixel: 298x298 gives 5.3k
        values rather than 355k, which keeps the gate near-free.
        """
        signature = np.ascontiguousarray(
            frame[::_SIGNATURE_STEP, ::_SIGNATURE_STEP, 0], dtype=np.int16)
        return signature

    def frame_changed(self, frame):
        """Mean absolute difference against the previous frame's signature."""
        signature = self.frame_signature(frame)
        previous, self._prev_signature = self._prev_signature, signature
        if previous is None or previous.shape != signature.shape:
            return float("inf")
        return float(np.abs(signature - previous).mean())

    def reset_gate(self):
        """Forget the previous frame, so the next one counts as changed."""
        self._prev_signature = None

    # -- inference ---------------------------------------------------------
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def predict(self, img_np):
        """Return ``(class_id, description, probabilities, is_hit)``.

        Kept as the general-purpose entry point. The real-time loop uses
        :meth:`predict_class` instead, which skips the softmax and the
        per-class dictionary that a hit/no-hit decision never reads.
        """
        logits = np.squeeze(self.ort_session.run(None, {self.input_name: img_np})[0])
        class_id = int(np.argmax(logits))
        probs = self.softmax(logits)
        probs_dict = {label.desc: float(probs[label.id]) for label in LABELS}
        return class_id, self.pred_dict[class_id]["desc"], probs_dict, \
            self.pred_dict[class_id]["hit"]

    def predict_class(self, img_np):
        """Return just the winning class id. This is the hot path."""
        logits = self.ort_session.run(None, {self.input_name: img_np})[0]
        return int(np.argmax(logits))

    def predict_label(self, img_np):
        """Return the :class:`dbd.labels.Label` for ``img_np``."""
        return label_for(self.predict_class(img_np))

    # -- lifecycle ---------------------------------------------------------
    def cleanup(self):
        """Release the capture handle and the session. Safe to call repeatedly."""
        if self.ort_session is None and self.grabber is None:
            return

        if self.grabber is not None:
            try:
                self.grabber.close()
            except Exception:
                pass
            self.grabber = None

        # Dropping the session is what actually frees the ONNX Runtime arena.
        self.ort_session = None
        self.input_name = None
        self._prev_signature = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


__all__ = ["AI_model", "DEFAULT_THREADS", "NONE_ID", "suggested_threads", "pil_from_bgra"]
