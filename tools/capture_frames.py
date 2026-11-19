"""Capture training frames from a monitor, then thin out the near-duplicates.

Replaces ``dbd/save_frames.py``, which had no similarity filter, plus the
``delete_consecutive_images`` / ``delete_similar_images`` helpers it never
called. Both passes now run in one deterministic pipeline so the captured
folder is directly usable as training data.

Inferring the class of a folder of images is in ``tools/predict_folder.py``.

Usage
-----
    python tools/capture_frames.py                      # capture until Ctrl+C, then clean
    python tools/capture_frames.py --keep-consecutive   # keep every captured frame
    python tools/capture_frames.py --clean-only <dir>   # only run the passes
"""

import argparse
import os
import time

import numpy as np
from PIL import Image
from mss import mss
from mss.tools import to_png

from dbd.utils.monitor import get_monitor_attributes

#: Frames closer together than this are considered the same moment.
KEEP_EVERY_NTH = 2
#: Mean per-pixel difference (0-255) below which two frames count as similar.
SIMILAR_THRESHOLD = 2.5


def capture(folder, monitor_id=1, crop_size=224):
    """Save PNG frames to ``folder`` until interrupted. Returns the file list."""
    os.makedirs(folder, exist_ok=True)
    monitor = get_monitor_attributes(monitor_id, crop_size=crop_size)
    files = []

    print(f"Capturing to {folder} (Ctrl+C to stop)")
    with mss() as sct:
        try:
            while True:
                shot = sct.grab(monitor)
                path = os.path.join(folder, f"{len(files):05d}.png")
                to_png(shot.rgb, shot.size, output=path)
                files.append(path)
        except KeyboardInterrupt:
            print(f"\nStopped after {len(files)} frames.")

    # A frame captured mid-write is not valid training data.
    if files:
        try:
            with Image.open(files[-1]) as img:
                img.verify()
        except Exception:
            os.remove(files[-1])
            files.pop()
            print("Dropped the last (possibly truncated) frame.")

    return files


def _frame_diff(path_a, path_b):
    """Mean absolute pixel difference (0-255) between two images."""
    with Image.open(path_a) as a, Image.open(path_b) as b:
        if a.size != b.size:
            return 255.0
        arr_a = np.asarray(a.convert("RGB"), dtype=np.int16)
        arr_b = np.asarray(b.convert("RGB"), dtype=np.int16)
    return float(np.abs(arr_a - arr_b).mean())


def thin_out(files, keep_every=KEEP_EVERY_NTH, threshold=SIMILAR_THRESHOLD):
    """Delete near-duplicate frames in place. Returns the surviving files.

    Two passes, in this order:

    1. keep every ``keep_every``-th frame, so a static screen cannot flood the
       set with identical frames;
    2. walk what is left and drop a frame whose difference from the previous
       *survivor* is below ``threshold``.
    """
    files = sorted(files)
    if not files:
        return []

    if keep_every > 1:
        keep = set(files[::keep_every])
        dropped = [p for p in files if p not in keep]
        for path in dropped:
            os.remove(path)
        files = sorted(keep)
        print(f"Pass 1: dropped {len(dropped)} consecutive frames (kept 1 in {keep_every})")

    survivors = [files[0]]
    removed = 0
    for path in files[1:]:
        if _frame_diff(survivors[-1], path) < threshold:
            os.remove(path)
            removed += 1
        else:
            survivors.append(path)
    print(f"Pass 2: dropped {removed} similar frames (threshold {threshold})")
    return survivors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?",
                        help="output folder; defaults to a timestamp")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--keep-every", type=int, default=KEEP_EVERY_NTH)
    parser.add_argument("--threshold", type=float, default=SIMILAR_THRESHOLD)
    parser.add_argument("--keep-consecutive", action="store_true",
                        help="skip pass 1 and keep every captured frame")
    parser.add_argument("--clean-only", metavar="DIR",
                        help="do not capture; only thin out an existing folder")
    args = parser.parse_args()

    if args.clean_only:
        files = [os.path.join(args.clean_only, f)
                 for f in os.listdir(args.clean_only) if f.lower().endswith(".png")]
    else:
        folder = args.folder or time.strftime("%Y%m%d-%H%M%S")
        files = capture(folder, args.monitor, args.crop_size)

    keep_every = 1 if args.keep_consecutive else args.keep_every
    survivors = thin_out(files, keep_every, args.threshold)
    print(f"Done: {len(survivors)} frames kept.")


if __name__ == "__main__":
    main()
