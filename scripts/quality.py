#!/usr/bin/env python3
"""Quantitative quality metrics for generated Qwen-Image-2.1 PNGs.

No human-in-the-loop viewing is required: every number here is computed from the
pixels, so runs on different offload recipes / resolutions / models can be compared
directly.

Per image:
  * width, height, mode, bytes
  * sharpness   Laplacian variance (higher = more high-frequency detail)
  * entropy     Shannon entropy of the luminance histogram (bits, 0..8)
  * contrast    luminance std / 127.5
  * saturation  mean HSV saturation
  * alpha_transparent_fraction / alpha_mean  (RGBA output only)

Optional:
  * --ocr-text   run tesseract and report the detected text (text rendering)
  * --psnr a.png b.png   compare two images of identical size

Usage:
  python3 quality.py results/full-offload --ocr-text "QWEN IMAGE 2.1"
  python3 quality.py --psnr results/a.png results/b.png
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def entropy_of(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def image_metrics(path: Path) -> dict:
    img = Image.open(path)
    arr = np.asarray(img)
    rgb = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    metrics = {
        "file": path.name,
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "bytes": path.stat().st_size,
        "megapixels": round(img.width * img.height / 1e6, 3),
        "sharpness_laplacian_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "entropy_bits": round(entropy_of(gray), 4),
        "contrast": round(float(gray.std()) / 127.5, 4),
        "saturation": round(float(hsv[:, :, 1].mean()) / 255.0, 4),
        "luma_mean": round(float(gray.mean()), 2),
        "unique_colors_rgb": int(len(np.unique(rgb.reshape(-1, 3), axis=0))),
    }
    if img.mode == "RGBA":
        alpha = arr[:, :, 3]
        metrics["alpha_mean"] = round(float(alpha.mean()), 2)
        metrics["alpha_transparent_fraction"] = round(float((alpha < 250).mean()), 4)
        metrics["alpha_fully_opaque_fraction"] = round(float((alpha >= 255).mean()), 4)
    return metrics


def ocr_text(path: Path) -> str:
    try:
        out = subprocess.run(
            ["tesseract", str(path), "-", "--psm", "11"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<ocr failed: {exc}>"


def normalize(text: str) -> str:
    keep = [c for c in text.lower() if c.isalnum() or c.isspace()]
    return "".join(keep)


def psnr(a: Path, b: Path) -> dict:
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    if x.shape != y.shape:
        return {"a": a.name, "b": b.name, "error": f"shape mismatch {x.shape} vs {y.shape}"}
    mse = float(((x - y) ** 2).mean())
    value = math.inf if mse == 0 else 10 * math.log10((255.0**2) / mse)
    return {
        "a": a.name,
        "b": b.name,
        "mse": round(mse, 3),
        "psnr_db": "inf" if math.isinf(value) else round(value, 2),
        "max_abs_diff": int(np.abs(x - y).max()),
        "mean_abs_diff": round(float(np.abs(x - y).mean()), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="PNG files or directories of PNGs")
    ap.add_argument("--ocr-text", default=None, help="expected substring to look for via tesseract")
    ap.add_argument("--psnr", nargs=2, metavar=("A", "B"), default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.psnr:
        result = psnr(Path(args.psnr[0]), Path(args.psnr[1]))
        print(json.dumps(result, indent=2))
        return 0

    if not args.paths:
        ap.error("give at least one PNG path/dir, or use --psnr A B")

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.png")))
        else:
            files.append(path)

    rows = []
    expected = normalize(args.ocr_text) if args.ocr_text else None
    for path in files:
        row = image_metrics(path)
        if expected:
            text = ocr_text(path)
            found = normalize(text)
            row["ocr_text"] = " ".join(text.split())
            row["ocr_expected_hit"] = bool(expected and expected in found)
        rows.append(row)

    print(f"{'file':44s} {'WxH':>11s} {'mode':>5s} {'MB':>6s} {'sharp':>9s} {'ent':>6s} {'contr':>6s} {'sat':>5s} {'colors':>8s}")
    for r in rows:
        print(
            f"{r['file'][:44]:44s} {r['width']}x{r['height']:<5d} {r['mode']:>5s} "
            f"{r['bytes']/1e6:6.2f} {r['sharpness_laplacian_var']:9.1f} "
            f"{r['entropy_bits']:6.3f} {r['contrast']:6.3f} {r['saturation']:5.3f} "
            f"{r['unique_colors_rgb']:8d}"
        )
        if "alpha_transparent_fraction" in r:
            print(
                f"    alpha: mean={r['alpha_mean']} transparent_fraction="
                f"{r['alpha_transparent_fraction']} fully_opaque_fraction="
                f"{r['alpha_fully_opaque_fraction']}"
            )
        if "ocr_text" in r:
            print(f"    ocr[{r['ocr_expected_hit']}]: {r['ocr_text'][:160]!r}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print("wrote", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
