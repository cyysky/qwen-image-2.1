#!/usr/bin/env python3
"""Speed/quality benchmark for the Qwen-Image-2.1 OpenAI-compatible service.

Talks to SGLang-Diffusion's OpenAI Images API (the same surface the existing
z-image service uses):

  POST {base}/v1/images/generations   JSON, text-to-image
  POST {base}/v1/images/edits          multipart, image editing

For every (size, steps) cell it warms up once, then repeats the request
`--reps` times and records:

  * wall_ms        client-side latency of the whole HTTP call
  * infer_s        server-reported inference_time_s (denoise + decode)
  * peak_mb        server-reported peak_memory_mb for the request
  * png_bytes      decoded image size

Results go to <out>/<tag>/results.json + results.csv and the PNGs land in the
same directory.

Examples:
  python3 bench.py --tag full-offload
  python3 bench.py --tag enc-layerwise --sizes 1024x1024 2048x2048 --steps 40 --reps 3
  python3 bench.py --tag edit --mode edit --image ref.png --steps 40
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests


DEFAULT_PROMPT = (
    'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, '
    "reflections on wet pavement"
)

TRANSPARENT_PROMPT = (
    "This is an RGBA image with transparency. A cute cartoon dragon sticker. "
    "The image has alpha channel and the background is transparent."
)


class GpuSampler(threading.Thread):
    """Polls nvidia-smi for used/peak VRAM of one GPU while requests run."""

    def __init__(self, gpu_index: int, interval: float = 0.25) -> None:
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[tuple[float, int]] = []
        self._stop_event = threading.Event()

    def run(self) -> None:
        query = f"nvidia-smi --id={self.gpu_index} --query-gpu=memory.used --format=csv,noheader,nounits"
        while not self._stop_event.is_set():
            try:
                out = subprocess.run(query, shell=True, capture_output=True, text=True, timeout=5)
                if out.returncode == 0 and out.stdout.strip():
                    self.samples.append((time.time(), int(out.stdout.strip().splitlines()[0])))
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def stop(self) -> dict:
        self._stop_event.set()
        self.join(timeout=5)
        if not self.samples:
            return {"vram_peak_mb": None, "vram_idle_mb": None}
        values = [v for _, v in self.samples]
        return {"vram_peak_mb": max(values), "vram_idle_mb": min(values)}


def post_generation(base_url: str, payload: dict, timeout: float) -> dict:
    r = requests.post(
        f"{base_url}/v1/images/generations", json=payload, timeout=timeout
    )
    r.raise_for_status()
    return r.json()


def post_edit(
    base_url: str, payload: dict, image_paths: list[Path], timeout: float
) -> dict:
    files = [
        ("image", (p.name, open(p, "rb"), "image/png")) for p in image_paths
    ]
    form = {
        k: ("true" if v is True else "false" if v is False else str(v))
        for k, v in payload.items()
        if v is not None
    }
    try:
        r = requests.post(
            f"{base_url}/v1/images/edits",
            data=form,
            files=files,
            timeout=timeout,
        )
    finally:
        for _, (_, fh, _) in files:
            fh.close()
    r.raise_for_status()
    return r.json()


def save_images(response: dict, out_dir: Path, stem: str) -> list[Path]:
    saved = []
    for i, item in enumerate(response.get("data", [])):
        b64 = item.get("b64_json")
        if not b64:
            continue
        suffix = ".png"
        path = out_dir / f"{stem}_{i}{suffix}"
        path.write_bytes(base64.b64decode(b64))
        saved.append(path)
    return saved


def run_case(
    *,
    base_url: str,
    mode: str,
    prompt: str,
    size: str,
    steps: int,
    seed: int,
    guidance: float,
    background: str,
    output_format: str,
    image_paths: list[Path],
    out_dir: Path,
    stem: str,
    timeout: float,
    extra: dict | None = None,
) -> dict:
    width, height = (int(x) for x in size.split("x"))
    payload: dict = {
        "model": "Qwen-Image-2.1",
        "prompt": prompt,
        "size": size,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "seed": seed,
        "response_format": "b64_json",
        "output_format": output_format,
        "n": 1,
    }
    if guidance and guidance > 0:
        payload["guidance_scale"] = guidance
    if background != "auto":
        payload["background"] = background
    if extra:
        payload.update(extra)

    started = time.perf_counter()
    if mode == "edit":
        response = post_edit(base_url, payload, image_paths, timeout)
    else:
        response = post_generation(base_url, payload, timeout)
    wall_s = time.perf_counter() - started

    files = save_images(response, out_dir, stem)
    sizes = [p.stat().st_size for p in files]
    return {
        "stem": stem,
        "mode": mode,
        "size": size,
        "steps": steps,
        "seed": seed,
        "guidance_scale": guidance,
        "wall_ms": round(wall_s * 1000, 1),
        "server_infer_s": response.get("inference_time_s"),
        "server_peak_mb": response.get("peak_memory_mb"),
        "png_bytes": sizes[0] if sizes else None,
        "image_count": len(files),
        "files": [p.name for p in files],
        "usage": response.get("usage"),
    }


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["mode"], row["size"], row["steps"]), []).append(row)
    summary = []
    for (mode, size, steps), items in groups.items():
        walls = [i["wall_ms"] for i in items]
        infers = [i["server_infer_s"] for i in items if i["server_infer_s"]]
        peaks = [i["server_peak_mb"] for i in items if i["server_peak_mb"]]
        summary.append(
            {
                "mode": mode,
                "size": size,
                "steps": steps,
                "reps": len(items),
                "wall_ms_median": round(statistics.median(walls), 1),
                "wall_ms_min": round(min(walls), 1),
                "server_infer_s_median": round(statistics.median(infers), 3) if infers else None,
                "server_peak_mb_max": round(max(peaks), 1) if peaks else None,
            }
        )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:7853")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--mode", choices=["text", "edit"], default="text")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--sizes", nargs="+", default=["1024x1024"])
    ap.add_argument("--steps", nargs="+", type=int, default=[40])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guidance-scale", type=float, default=0.0)
    ap.add_argument("--background", default="auto", choices=["auto", "transparent", "opaque"])
    ap.add_argument("--output-format", default="png", choices=["png", "jpeg", "webp"])
    ap.add_argument("--image", action="append", default=[], help="reference image for --mode edit")
    ap.add_argument("--gpu-index", type=int, default=2, help="GPU to sample VRAM from")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "results"),
        help="artifact directory (default: <project>/results)",
    )
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument(
        "--extra-json",
        default=None,
        help=(
            "extra request-body fields merged into every call, e.g. "
            '{"enable_cache_dit": true} for SGLang Cache-DiT'
        ),
    )
    args = ap.parse_args()
    extra = json.loads(args.extra_json) if args.extra_json else None

    out_dir = Path(args.out).expanduser().resolve() / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [Path(p).expanduser().resolve() for p in args.image]
    if args.mode == "edit" and not image_paths:
        print("--mode edit needs at least one --image", file=sys.stderr)
        return 2

    models = requests.get(f"{args.base_url}/v1/models", timeout=30).json()
    print("server models:", json.dumps(models, indent=2)[:400])

    sampler = GpuSampler(args.gpu_index)
    sampler.start()

    warm = run_case(
        base_url=args.base_url,
        mode=args.mode,
        prompt=args.prompt,
        size=args.sizes[0],
        steps=args.steps[0],
        seed=args.seed,
        guidance=args.guidance_scale,
        background=args.background,
        output_format=args.output_format,
        image_paths=image_paths,
        out_dir=out_dir,
        stem=f"warmup_{args.mode}_{args.sizes[0]}_{args.steps[0]}",
        timeout=args.timeout,
        extra=extra,
    )
    print("warmup:", json.dumps(warm, indent=2))

    rows = []
    for size in args.sizes:
        for steps in args.steps:
            for rep in range(args.reps):
                stem = f"{args.mode}_{size}_{steps}_r{rep}"
                row = run_case(
                    base_url=args.base_url,
                    mode=args.mode,
                    prompt=args.prompt,
                    size=size,
                    steps=steps,
                    seed=args.seed + rep,
                    guidance=args.guidance_scale,
                    background=args.background,
                    output_format=args.output_format,
                    image_paths=image_paths,
                    out_dir=out_dir,
                    stem=stem,
                    timeout=args.timeout,
                    extra=extra,
                )
                rows.append(row)
                print(json.dumps(row))

    vram = sampler.stop()
    summary = summarize(rows)

    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "tag": args.tag,
                "extra": extra,
                "vram": vram,
                "runs": rows,
                "summary": summary,
            },
            indent=2,
        )
    )
    with open(out_dir / "results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== summary ===")
    for row in summary:
        print(json.dumps(row))
    print("\ngpu2 vram:", json.dumps(vram))
    print("artifacts:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
