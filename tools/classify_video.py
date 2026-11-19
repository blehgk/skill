"""Extract frames from a video and classify each with the skill-check model.

The model was trained on 224x224 crops from a 1080p screen, so a crop of the
same *relative* size on a 720p recording is 224 * 720/1080 = 149 px. Both scales
are tried so we can see which one the model actually responds to.

    python tools/classify_video.py <video> <outdir> [max_frames]
"""
import os
import subprocess
import sys
from time import time

import numpy as np
from PIL import Image

from dbd.AI_model import AI_model
from dbd.labels import label_for

FFMPEG = (r"C:\Users\Zeta\AppData\Local\Microsoft\WinGet\Packages"
          r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
          r"\ffmpeg-9.0-full_build\bin\ffmpeg.exe")


def extract(video, outdir, max_frames=0):
    os.makedirs(outdir, exist_ok=True)
    for old in os.listdir(outdir):
        os.remove(os.path.join(outdir, old))
    cmd = [FFMPEG, "-v", "error", "-i", video]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    # ffmpeg 9 dropped -vsync; -fps_mode is its replacement.
    cmd += ["-fps_mode", "passthrough", os.path.join(outdir, "%06d.png")]
    t0 = time()
    subprocess.run(cmd, check=True)
    files = sorted(os.path.join(outdir, f) for f in os.listdir(outdir))
    print(f"extracted {len(files)} frames in {time()-t0:.1f}s -> {outdir}")
    return files


def crop_center(pil, size):
    """Centre square crop of ``size`` px: the region the bot samples."""
    w, h = pil.size
    size = min(size, h, w)
    left, top = (w - size) // 2, (h - size) // 2
    return pil.crop((left, top, left + size, top + size))


def as_bgra(pil):
    """AI_model.preprocess() takes the raw capture buffer (BGRA), so wrap the
    RGB crop into that layout; the alpha channel carries no information."""
    rgb = np.asarray(pil, dtype=np.uint8)
    out = np.empty((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    out[:, :, 0] = rgb[:, :, 2]
    out[:, :, 1] = rgb[:, :, 1]
    out[:, :, 2] = rgb[:, :, 0]
    out[:, :, 3] = 255
    return out


def classify(frames, model, crop_size):
    """One pass: per-class counts and the number of 0->hit press edges."""
    counts = np.zeros(11, dtype=np.int64)
    edges = 0
    prev_hit = False
    sequence = []
    t0 = time()
    for path in frames:
        with Image.open(path) as im:
            region = crop_center(im.convert("RGB"), crop_size)
        cid = model.predict_class(model.preprocess(as_bgra(region)))
        counts[cid] += 1
        hit = label_for(cid).hit
        if hit and not prev_hit:
            edges += 1
        prev_hit = hit
        sequence.append(cid)
    return counts, edges, sequence, time() - t0


def report(tag, counts, edges, elapsed):
    total = int(counts.sum())
    print(f"\n=== {tag} ({elapsed:.1f}s, {total/elapsed:.0f} img/s) ===")
    for cid in range(11):
        if counts[cid]:
            print(f"   {counts[cid]:6d}  {counts[cid]/total*100:5.1f}%  "
                  f"{label_for(cid).desc}")
    hits = int(sum(counts[c] for c in range(11) if label_for(c).hit))
    print(f"   'None' {counts[0]/total*100:.1f}% | hit-class frames {hits} "
          f"({hits/total*100:.1f}%) | press edges {edges}")


def main():
    video, outdir = sys.argv[1], sys.argv[2]
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    frames = extract(video, outdir, max_frames)

    model = AI_model("models/model.onnx", monitor_id=1, backend="mss")
    height = Image.open(frames[0]).height
    proportional = int(round(224 * height / 1080))
    print(f"frame height {height} -> proportional crop {proportional}px "
          f"and 224px both tested")

    for crop_size in (proportional, 224):
        counts, edges, seq, elapsed = classify(frames, model, crop_size)
        report(f"crop {crop_size}px", counts, edges, elapsed)
        non_none = [i for i, c in enumerate(seq) if c != 0]
        if non_none:
            print(f"   first non-None at frame {non_none[0]}, "
                  f"last at {non_none[-1]}, total {len(non_none)}")

    model.cleanup()


if __name__ == "__main__":
    main()
