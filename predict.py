"""Predict a digit from an image file, or try the model on a random dataset sample.

Usage:
    python predict.py path/to/digit.png
    python predict.py path/to/digit.png --invert   # for black digit on white background
    python predict.py --sample 5                   # 5 random samples from the dataset
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.datasets import load_digits

MODEL_PATH = Path("models/model.joblib")


def image_to_features(path, invert=False):
    """Convert any image to the 8x8, 0-16 intensity format used by sklearn's digits.

    The dataset has bright strokes on a dark background, so pass invert=True
    for the usual dark-ink-on-white-paper images.
    """
    img = Image.open(path).convert("L")
    if invert:
        img = ImageOps.invert(img)
    img = img.resize((8, 8), Image.LANCZOS)
    arr = np.asarray(img, dtype=float) / 255.0 * 16.0
    return arr.reshape(1, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="path to an image of a digit")
    parser.add_argument("--invert", action="store_true", help="invert image colors")
    parser.add_argument("--sample", type=int, help="predict N random dataset samples")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("No model found. Run `python train.py` first.")
    model = joblib.load(MODEL_PATH)

    if args.sample:
        digits = load_digits()
        idx = np.random.default_rng().choice(len(digits.data), args.sample, replace=False)
        for i in idx:
            pred = model.predict(digits.data[i:i + 1])[0]
            print(f"sample #{i}: predicted {pred}, actual {digits.target[i]}")
    elif args.image:
        X = image_to_features(args.image, args.invert)
        pred = model.predict(X)[0]
        msg = f"Predicted digit: {pred}"
        if hasattr(model, "predict_proba"):
            msg += f" (confidence {model.predict_proba(X).max():.0%})"
        print(msg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
