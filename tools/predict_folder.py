"""Classify a folder of screenshots with the ONNX model.

Replaces ``dbd/predict_folder.py``, which reloaded the model once per image
inside the loop (the reason its own benchmark reported seconds per folder) and
crashed whenever ``--copy`` / ``--move`` was used, because the destination
class folder was never created.

Three modes, chosen by the destination flag:

* default - report predictions, change nothing;
* ``--copy DIR`` - copy each image into ``DIR/<class>``;
* ``--move DIR`` - move each image into ``DIR/<class>``.

If the folder being scanned is itself named after a class, the script also
prints a confusion count, which is what you want when checking dataset quality.

Usage
-----
    python tools/predict_folder.py dataset/tests
    python tools/predict_folder.py dataset/tests --copy dataset/sorted
"""

import argparse
import os
import shutil
import time
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

from dbd.AI_model import AI_model
from dbd.labels import label_for

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def find_images(folder):
    """Image files directly inside ``folder``, in deterministic order."""
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_SUFFIXES)
    )


def infer_folder(images, model, destination=None, move=False):
    """Classify ``images`` with a single loaded ``model``.

    Returns ``(results, counts)`` where ``results`` is a list of
    ``(path, class_id)`` and ``counts`` maps class id to occurrences.
    """
    results = []
    counts = Counter()

    for path in tqdm(images, unit="img"):
        # The model owns resizing: it accepts any square/rect image and scales
        # it to its own input size.
        with Image.open(path) as img:
            tensor = model.pil_to_numpy(img.convert("RGB"))
        class_id, _, _, _ = model.predict(tensor)

        results.append((path, class_id))
        counts[class_id] += 1

        if destination:
            class_dir = os.path.join(destination, label_for(class_id).desc.replace("/", "-"))
            os.makedirs(class_dir, exist_ok=True)
            target = os.path.join(class_dir, os.path.basename(path))
            if move:
                shutil.move(path, target)
            else:
                shutil.copy2(path, target)

    return results, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="folder containing screenshots")
    parser.add_argument("--model", default="models/model.onnx")
    parser.add_argument("--threads", type=int, default=0,
                        help="ONNX intra-op threads; 0 lets runtime decide")
    parser.add_argument("--copy", metavar="DIR",
                        help="copy each image into DIR/<predicted class>")
    parser.add_argument("--move", metavar="DIR",
                        help="move each image into DIR/<predicted class>")
    args = parser.parse_args()

    if args.copy and args.move:
        parser.error("--copy and --move are mutually exclusive")
    if not os.path.isdir(args.folder):
        parser.error(f"not a folder: {args.folder}")

    images = find_images(args.folder)
    if not images:
        print(f"No images found in {args.folder}")
        return

    model = AI_model(args.model, nb_cpu_threads=args.threads or None)
    print(f"Model: {args.model} | provider: {model.check_provider()} | {len(images)} images")

    started = time.perf_counter()
    results, counts = infer_folder(
        images, model,
        destination=args.copy or args.move,
        move=bool(args.move),
    )
    elapsed = time.perf_counter() - started
    model.cleanup()

    print(f"\n{len(results)} images in {elapsed:.2f}s "
          f"({len(results)/elapsed:.0f} img/s, {elapsed/len(results)*1000:.1f} ms/img)")
    print("\nPredicted classes:")
    for class_id, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:6d}  ({count/len(results)*100:5.1f}%)  {label_for(class_id).desc}")

    # If the folder is named after a class, treat that as ground truth.
    folder_name = os.path.basename(os.path.normpath(args.folder))
    truth = next((c for c in range(11) if label_for(c).desc == folder_name), None)
    if truth is not None:
        correct = sum(1 for _, c in results if c == truth)
        print(f"\nFolder is '{folder_name}' (class {truth}): "
              f"{correct}/{len(results)} = {correct/len(results)*100:.1f}% match")
    if args.copy or args.move:
        verb = "Moved" if args.move else "Copied"
        print(f"{verb} into {args.copy or args.move}/<class>/")


if __name__ == "__main__":
    main()
