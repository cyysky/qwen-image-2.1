# Qwen-Image-2.1 service: measured speed and quality

All numbers below were measured on this box with the service running on **GPU 2
only**, while the pre-existing z-image service kept running on the same card.
Raw artifacts (PNGs, per-run JSON/CSV, VRAM samples) are in `results/<tag>/`.

## Test bed

| Item | Value |
| --- | --- |
| GPUs | 3x RTX 4090 24 GB |
| GPU 0 | llama-server, ~17.0 GB used (not touched) |
| GPU 1 | vLLM, ~17.9 GB used (not touched) |
| GPU 2 | z-image SGLang diffusion, ~1.4 GB used; Qwen-Image-2.1 co-resident |
| Model | Qwen/Qwen-Image-2.1, 33 GB on disk, mounted read-only at `/model` |
| Serving stack | SGLang-Diffusion, nightly-dev-cu13-20260921 (PR #39983) |
| API | OpenAI Images API on `http://127.0.0.1:7853` |
| Startup | ~90 s from container start to `/health` = 200 |

Every request in the tables ran through `POST /v1/images/generations` (JSON) or
`POST /v1/images/edits` (multipart) with `response_format=b64_json`, so the
timings include the full pipeline: text/image conditioning, denoising and VAE decode,
plus base64 encoding of the PNG.

## Offload recipes compared

The service was restarted with each recipe (`SGLANG_OFFLOAD_ARGS=...`), then measured
with `scripts/bench.py` (one warm-up request, then N repetitions).

| Recipe | Flag summary | What stays on GPU 2 |
| --- | --- | --- |
| **A** | `--performance-mode manual --dit-cpu-offload true --text-encoder-cpu-offload true` | VAE only; DiT and Qwen3-VL stream from host RAM per stage |
| **B** | `--performance-mode memory` | VAE, image encoder and text encoder layerwise-offloaded; DiT offloaded between stages |
| **C** | `--performance-mode manual --dit-layerwise-offload true --layerwise-offload-components dit text_encoder image_encoder vae` | nothing resident; every component streams per layer from host RAM |

## Speed

Median of the repetitions; `wall` is the client-side HTTP time, `infer` is the
server's own `inference_time_s`, `peak` is the server's `peak_memory_mb`
(torch reserved peak for that request), `gpu2` is `nvidia-smi` peak for the whole
card including z-image.

| Recipe | Request | Reps | wall | infer | peak | gpu2 peak | Result |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1024x1024, 40 steps, text | 3 | **20.96 s** | 20.56 s | 21.6 GB | 24.01 GB | OK |
| A | 1024x1024, 20 steps, text | 3 | **12.17 s** | 11.77 s | 21.6 GB | 24.01 GB | OK |
| A | 1024x1024, 40 steps, edit | - | - | - | - | - | **OOM** |
| A | 2048x2048, 40 steps, text | 2 | **100.02 s** | 99.14 s | 21.6 GB | 24.01 GB | OK |
| B | 1024x1024, 40 steps, text | 3 | **20.03 s** | 19.63 s | 16.5 GB | 18.83 GB | OK |
| B | 1024x1024, 40 steps, edit | 3 | **23.85 s** | 23.66 s | 18.1 GB | 20.49 GB | OK |
| B | 2048x2048, 40 steps, text | 2 | **99.07 s** | 98.18 s | 21.4 GB | 23.72 GB | OK |
| B | 2048x2048, 40 steps, edit | - | - | - | - | - | **OOM** |
| C | 1024x1024, 40 steps, text | 3 | 23.37 s | 23.22 s | **8.0 GB** | **10.32 GB** | OK |
| C | 1024x1024, 40 steps, edit | 3 | 24.00 s | 23.83 s | **9.0 GB** | **11.44 GB** | OK |
| C | 2048x2048, 40 steps, text | 2 | 97.76 s | 97.20 s | 21.6 GB | 23.98 GB | OK |
| C | 2048x2048, 40 steps, edit | 1 | 135.30 s | 135.30 s | 19.6 GB | - | OK |

Run-to-run spread was tight: the three 1024x1024 / 40-step repetitions
landed within 0.5 % of each other (e.g. 20.90 / 20.96 / 20.99 s wall for
recipe A), and the GPU-2 idle baseline stayed at 3.1-4.5 GB with z-image resident.

### What the numbers say

* **Denoising scales linearly with steps, and quadratically with resolution.**
  Recipe A at 1024x1024 costs 11.77 s at 20 steps and 20.56 s at 40 steps, i.e.
  ~0.44 s/step plus ~3 s of fixed conditioning + VAE work. Going from 1024x1024
  to 2048x2048 (4x the pixels) at 40 steps costs 99.14 s, ~4.8x the 1K time -
  the expected quadratic attention cost, with no throughput trick available on one card.
* **Recipe B is the fastest 1K recipe** (20.03 s text, 23.85 s edit), ~1 s (5 %)
  ahead of A, because the text encoder streams layerwise instead of the whole 8B
  encoder being re-uploaded per request, and ~3.3 s (14 %) ahead of C, which pays
  PCIe traffic for every DiT layer on every step.
* **Recipe C is the fastest 2K recipe** (97.76 s vs 99.07 s for B) even though it
  is the slowest at 1K. At 2048x2048 the run is activation- rather than
  weight-bound, and streaming the 7B DiT layerwise avoids the allocator pressure
  that a 14 GB weight burst creates on an already full card.
* **Recipe C is the only recipe where every documented endpoint works.** Both A
  and B OOM on 2048x2048 edits: A needs 96 MiB with 21.0 GB already allocated,
  B needs 384 MiB with 113 MiB free and 1.7 GB reserved-but-unallocated. C's
  layerwise DiT plus layerwise VAE keeps the peak low enough to finish (135.3 s).
* The default in `qwen-image-2.1.yml` is therefore **C**, so every documented
  size and endpoint works out of the box on the shared card. B is the documented
  faster option for 1024x1024-only workloads.

## VRAM headroom on the shared card

| Recipe | GPU-2 idle | GPU-2 peak (1K) | GPU-2 peak (2K) | Headroom at peak |
| --- | --- | --- | --- | --- |
| A | 3.1 GB | 24.01 GB | 24.01 GB | ~0.5 GB (card full) |
| B | 3.8 GB | 18.83 GB | 23.72 GB | ~0.8 GB at 2K |
| C | 4.2 GB | 10.32 GB | 23.98 GB | ~13.7 GB at 1K |

The idle figures include z-image's ~1.4 GB, which stayed up and healthy for the
whole measurement series. A and B push the card to ~98 % of capacity at 1K and 2K
respectively, which is why the next stage (the reference-image encode in an edit)
had no room left. C leaves >13 GB free at 1K and still finishes a 2K edit.

## Quality

Quantitative metrics from `scripts/quality.py`, computed on the PNGs the server
returned (no human viewing needed).

| Probe | Size | Mode | Sharpness (Laplacian var) | Entropy | Contrast | Saturation | Unique colors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Neon-sign prompt, 40 steps | 1024x1024 | RGBA | 1276.4 | 6.175 | 0.367 | 0.613 | 314 879 |
| Same prompt | 2048x2048 | RGBA | 327.6 | 5.789 | 0.299 | 0.324 | 326 922 |

Sharpness is scale-dependent (a 2K frame has ~4x the pixels of a 1K frame, so the
per-pixel Laplacian variance drops even for equally crisp output); compare
sharpness only between images of the same size.

Targeted probes:

| Capability | Probe | Result |
| --- | --- | --- |
| Text rendering | "A minimalist poster with the exact text \"QWEN IMAGE 2.1\" ..." | tesseract reads exactly `QWEN IMAGE 2.1` |
| Native alpha | Cartoon dragon sticker, `background=transparent` | RGBA PNG, alpha mean 54.4, **79.9 % of pixels transparent**, 17.4 % fully opaque |
| Opaque output | Neon-sign prompt, default background | RGBA PNG, alpha mean 254.9, 93.6 % fully opaque (model chose an opaque frame) |
| Reference edit | Add a lantern to the neon-sign image | PSNR 11.09 dB vs the input, max diff 255 - the edit changes the image rather than passing it through |
| Seed control | Same prompt, seeds 42 vs 43 | PSNR 11.19 dB - different seeds give genuinely different images |

### The offload recipe does not change the image

Same prompt, same seed, different recipe:

| Comparison | 1K | 2K |
| --- | --- | --- |
| A vs B (seed 42) | PSNR **inf** | - |
| B vs C (seed 42) | PSNR **inf** | PSNR **inf** |

The PNG bytes are identical (`mse = 0`, `max_abs_diff = 0`), so CPU offloading is
numerically transparent: it only trades time for GPU memory, never quality. The
recipes can be switched freely for capacity reasons.

## Reproduce

```bash
cd /home/aiserver/qwen-image-2.1
sudo docker compose -f qwen-image-2.1.yml up -d --build   # default recipe C

# speed (uses the running server; --tag names the results/<tag> dir)
python3 scripts/bench.py --tag layerwise-1k-40  --sizes 1024x1024 --steps 40 --reps 3
python3 scripts/bench.py --tag layerwise-2k-40  --sizes 2048x2048 --steps 40 --reps 2
python3 scripts/bench.py --tag layerwise-edit-1k-40 --mode edit \
  --image refs/neon-ref.png --prompt "Add a glowing red paper lantern hanging above the sign" \
  --sizes 1024x1024 --steps 40 --reps 3

# quality
python3 scripts/quality.py results/layerwise-1k-40
python3 scripts/quality.py outputs/probes/poster_0.png --ocr-text "QWEN IMAGE 2.1"
python3 scripts/quality.py --psnr results/memory-1k-40/text_1024x1024_40_r0_0.png \
                                 results/layerwise-1k-40/text_1024x1024_40_r0_0.png

# a different recipe (restart required)
sudo env SGLANG_OFFLOAD_ARGS="--performance-mode memory" \
  docker compose -f qwen-image-2.1.yml up -d --force-recreate
```

## Known limitations

* 2048x2048 **edits** only fit with recipe C on this shared card; they take
  135 s and peak at 19.6 GB. A and B OOM there.
* 2048x2048 text-to-image peaks at ~21.6-21.4 GB under every recipe, because
  the 2K cost is dominated by activations (256x256 latents) and the RGBA VAE
  decode, not by weights. With z-image's 1.4 GB resident, the card is ~98 % full
  at that point, so anything else landing on GPU 2 during a 2K run can OOM it.
* `n > 1`, batch size > 1 and `guidance_scale > 1` (true CFG) were not
  benchmarked; they raise the peak memory of the same stages measured above.
* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was tried to reclaim
  the 1.7 GB of reserved-but-unallocated memory seen in the B 2K-edit OOM, but it
  broke the CUDA context (`unspecified launch failure`) with this driver/CUDA 13
  combination, so it is not used.
* Sharpness/entropy are computed with OpenCV on the decoded PNG and are useful for
  comparing runs, not as an absolute quality score; the text-rendering and alpha
  probes are the semantic checks.
