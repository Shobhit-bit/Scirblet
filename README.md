# Equation Scribble Pad

Draw a math equation by hand, hit **Solve**, and get it read and evaluated —
no camera, no file upload. You draw directly into a native window, and the
same CNN-based symbol classifier from the original photo pipeline reads it
and hands the equation off to SymPy to solve.

![status](https://img.shields.io/badge/status-working%20prototype-yellow)

## What's in here

| File | What it is |
|---|---|
| `scribble_pad_app.py` | The app itself — a Tkinter drawing pad, wired directly into the recognition pipeline. Run this. |
| `equation_reader_v2.keras` | The trained symbol classifier (16 classes: digits 0–9, `+ - * %`, and `[ ]` for brackets). |

The app is self-contained — segmentation, symbol normalization, and equation
solving all live inside `scribble_pad_app.py` itself, adapted from the
original file-based pipeline to work on the in-memory drawing directly
instead of a saved photo.

## Setup

You'll need Python 3.9+ and below 3.12 a handful of packages:

```bash
pip install pillow opencv-contrib-python sympy tensorflow
```

(`opencv-contrib-python` rather than plain `opencv-python` — only strictly
needed if you turn stroke-thinning back on, see below, but it's harmless to
have installed either way.)

Then just run it:

```bash
python scribble_pad_app.py
```

A window opens with a blank canvas. `MODEL_PATH` at the top of the script
points at `equation_reader_v2.keras` by default — keep the two files in the
same folder, or update the path if you move things around.

## How to use it

1. Draw your equation on the pad with the mouse — digits, `+ - * %`, and
   square brackets `[ ]` for grouping (since parentheses are awkward to
   distinguish from other strokes, brackets stand in for them).
2. Click **Solve**.
3. A small popup shows the equation as the model read it, followed by the
   evaluated answer. If nothing readable was found, it tells you that
   instead of guessing.
4. **Clear** wipes the pad so you can try again.

## How it works under the hood

1. **Capture** — every stroke you draw on the visible canvas is mirrored,
   pixel-for-pixel, into an in-memory image. No screenshot logic, no
   temp files — by the time you hit Solve, the drawing already exists as a
   plain numpy array.
2. **Threshold** — Otsu binarization cleans the drawing into a clean
   black-on-white binary image.
3. **Segment** — bounding boxes are found per stroke, then merged
   horizontally so that multi-stroke characters (like `+`, `%`, or a
   two-stroke `=`) are treated as one symbol instead of getting split apart.
4. **Classify** — each segmented symbol is cropped, padded to a square,
   resized to 28×28, and fed through the CNN one at a time.
5. **Solve** — the recognized characters are joined into a string, brackets
   are swapped for parentheses, and the whole thing is handed to SymPy to
   evaluate. If it doesn't parse as a valid equation, you get a clear error
   instead of a wrong number.

## Known limitations (read this before assuming it's broken)

- **Handwriting style matters.** The classifier was trained on a Kaggle
  handwritten digits/operators dataset plus augmentation for stroke-width
  and pose variation — but it's still not going to recognize wildly
  unconventional handwriting reliably. Clear, reasonably sized characters
  work best.
- **Stroke thickness (`STROKE_WIDTH`, default 10px)** should roughly match
  a normal marker/pen stroke. Too thin or too thick and segmentation or
  classification can misfire.
- **`apply_thinning` is off by default** in this version — testing showed it
  actually hurt accuracy on real scribbles rather than helping (it was
  originally added to mimic MNIST-style thin strokes, but tends to distort
  already-clean drawn strokes). Leave it `False` unless you've verified
  otherwise on your own handwriting.
- **Symbol spacing matters.** If characters are drawn touching or
  overlapping, `x_gap_thresh` in `segment_symbols()` may merge them into one
  blob, or split a single multi-stroke character into two. If you notice
  this happening a lot, that's the first knob to tune.
- **No parentheses, only brackets.** Use `[` and `]` in place of `(` and `)`
  — the model was trained on the former, not the latter.

## Dataset credit

Symbol classifier trained on the
[Handwritten Digits and Operators](https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators)
dataset by Michel Heusser on Kaggle.
