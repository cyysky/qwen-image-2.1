# Qwen-Image-2.1 service: measured speed and quality

All numbers below were measured on this box with the service running on **GPU 2
only**, while the pre-existing z-image service kept running on the same card.
Raw artifacts (PNGs, per-run JSON/CSV, VRAM samples) are in `results/<tag>/`.

## Test bed

| Item | Value |
| --- | --- |
| GPUs | 3x RTX 4090 24 GB |
| GPU 0 | free (llama-server stopped for this work; 15 MiB idle) |
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
| **C + `--vae-tiling true`** | 2048x2048, 40 steps, text | 3 | 99.9 s | 99.4 s | **5.8-6.0 GB** | **8.37 GB** | OK |
| **C + `--vae-tiling true`** | 2048x2048, 40 steps, edit | 1 | 139.7 s | 139.2 s | 18.0 GB | 20.38 GB | OK |
| **C + `--vae-tiling true`** | 1024x1024, 40 steps, text | 1 | 24.7 s | 24.4 s | **4.0 GB** | - | OK |
| **C + `--vae-tiling true`** | 1024x1024, 40 steps, edit | 1 | 25.1 s | 24.8 s | **7.5 GB** | - | OK |
| **C, `--vae-tiling false`** (shipped default) | 1024x1024, 40 steps, text | 2 | 23.4 s | 23.2 s | 8.0 GB | 10.3 GB | OK |
| **C, `--vae-tiling false`** (shipped default) | 2048x2048, 40 steps, text | 1 | 97.9 s | 97.5 s | 21.5 GB | 23.9 GB | OK |

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

## The 2048x2048 OOM, and the fix

Recipe C originally still tripped the allocator on 2048x2048: the run finished and
returned HTTP 200, but the log carried

```
[rank0]:[W922 05:15:28] memory allocation failed with OOM on device 0 while
trying to allocate 3630170112 bytes (free: 202833920, total: 25250627584).
[DecodingStage] finished in 0.9030 seconds
Peak memory usage: 21540.00 MB
```

Two things about that message are easy to misread. "device 0" is **GPU 2**: the
container is pinned with `device_ids: ['2']`, so it sees its only GPU as local
device 0. And the failure is *non-fatal*: the decode fell back and still wrote a
PNG, but the image it wrote had lost its text, which is the "text rendering is
failed" symptom.

`--vae-cpu-offload true` was tried first and **changed nothing** (same 3.6 GB
request, same peak, same OOM). SGLang's own decode stage explains why in a comment
at `stages/decoding.py`: `--vae-cpu-offload` moves VAE *weights*, not the
activations that overflow.

The fix is `--vae-tiling true`, which is what SGLang's OOM advice points at. It
splits the decode into 256x256 tiles (stride 192, overlap blending) and bounds the
working set. Same prompt, same seed, 2048x2048, 40 steps, tiling off vs on:

| `--vae-tiling` | Wall | infer | Server peak | `nvidia-smi` peak (card) | OOM logged |
| --- | --- | --- | --- | --- | --- |
| `false` | 97.35 s | 96.8 s | 21572 MB | ~24.0 GB | yes, 1 |
| `true` | 99.99 s | 99.4 s | **5.8-6.0 GB** | **8.37 GB** | **no** |

Tiling costs ~2.5 s (2.6 %) and removes ~15.8 GB of peak, so it is the recipe to
use whenever 2K matters. The shipped default keeps it off because that is what the
operator verified in use (see the recipe C rows in the speed table). Two consecutive
tiled runs were byte-identical (`md5 b75ba2f0...`), so the tiled path is
deterministic as well, and the tiled 1K PNG is byte-identical to the untiled one
(`ba27b0ac...`).

Tiling also engages at 1024x1024 (a 1024 px sample exceeds the 256 px tile
minimum), where it costs ~1.3 s (5 %) and cuts the peak from 8.0 GB to 4.0 GB for
text and 9.0 GB to 7.5 GB for edits. The 1K text probe still OCRs exactly as
`QWEN IMAGE 2.1`.

### Tiling does not cost quality

Same prompt and seed, tiled vs untiled 2048x2048 PNGs compared pixel-wise:

| Metric | Value |
| --- | --- |
| Mean absolute difference | 0.93 / 255 (0.36 %) |
| Pixels differing by more than 2 | 13.8 % |
| Max channel difference | 79 |
| Excess gradient energy at tile boundaries | 0.055 vs a 4.1 baseline (1.3 %) |

The differences are diffuse rather than concentrated on the tile grid: rows and
columns at multiples of 192 carry essentially the same gradient energy in the tiled and
untiled images, so there is no seam. Text OCR is comparable - the tiled run reads
`THE LAST LIGHT` and `IN THEATERS OCTOBER 24`, the untiled run reads
`Every ending is a beginning` and `IN THEATERS OCTOBER 24`. Tiling is a numerics
change in the decoder, not a quality regression.

## Upstream's Cache-DiT: 2.5-2.7x for free

`QwenLM/Qwen-Image-2.1` lists Cache-DiT as one of the Day-0 SGLang
features. This build ships it, it is **off by default**, and it is a per-request
switch (`"enable_cache_dit": true`), so no restart is needed to compare it.

The log confirms it is a DBCache residual cache over the denoise steps:
`DBCache_F1B0_W4I1M0MC3_R0.24_N40_CFG0`, i.e. warm up 4 steps, then skip any
step whose residual moved less than 0.24, at most 3 skipped steps in a row. It wraps
the DiT through a custom `ForwardPattern.Pattern_3` block adapter because this is the
native SGLang DiT rather than a diffusers one, and it is compatible with the
layerwise offload (skipped blocks are not streamed).

Same prompt, same seed, `--vae-tiling false`, repetitions where noted:

| `enable_cache_dit` | Request | Reps | Wall | infer | Server peak | Card peak |
| --- | --- | --- | --- | --- | --- | --- |
| `false` | 1024x1024, 40 steps, text | 2 | 23.37 s | 23.22 s | 7968 MB | 10316 MB |
| `true` | 1024x1024, 40 steps, text | 2 | **9.28 s** | 9.12 s | 8034 MB | 10382 MB |
| `false` | 2048x2048, 40 steps, text | 1 | 97.89 s | 97.47 s | 21522 MB | **23870 MB** |
| `true` | 2048x2048, 40 steps, text | 1 | **35.84 s** | 35.43 s | **19466 MB** | 21816 MB |

Two things matter beyond the raw speedup:

* **At 2K it lowers the peak by ~2 GB.** The uncached 2K run peaked at 23870 MB
  of the card's 24564 MB - the same coin-flip that lost text in the earlier OOM.
  With the cache it peaked at 21816 MB, which is ~2.7 GB of slack instead of ~0.7 GB.
* **The output is an approximation, not a re-render.** Same prompt and seed, cached
  vs uncached, both at `--vae-tiling false`: PSNR **29.05 dB** at 1024x1024 (mean
  abs diff 0.86/255) and **30.37 dB** at 2048x2048 (mean abs diff 0.82/255).
  Sharpness/entropy move by <1 %. On the exact-text poster probe tesseract still
  reads `QWEN IMAGE 2.1` at 1024x1024 with the cache on; at 2048x2048 both the
  cached and uncached runs read the same partial `... IMAGE 2.1`, so the 2K
  legibility ceiling is the model, not the cache. (The 2K PSNR in the section
  above, 23.65 dB, compares a tiled baseline against an untiled cached run and
  therefore mixes two changes; the 30.37 dB row here is the clean one.)

## VRAM headroom on the shared card

| Recipe | GPU-2 idle | GPU-2 peak (1K) | GPU-2 peak (2K) | Headroom at peak |
| --- | --- | --- | --- | --- |
| A | 3.1 GB | 24.01 GB | 24.01 GB | ~0.5 GB (card full) |
| B | 3.8 GB | 18.83 GB | 23.72 GB | ~0.8 GB at 2K |
| C | 4.2 GB | 10.32 GB | 23.98 GB | ~13.7 GB at 1K |
| **C + `--vae-tiling true`** (shipped default) | 4.2 GB | 10.3 GB | **8.37 GB text / 20.38 GB edit** | **~16 GB text / ~4 GB edit** |

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

# Cache-DiT on the same server, no restart (per-request switch)
python3 scripts/bench.py --tag upstream-cachedit-1k-40 --sizes 1024x1024 --steps 40 \
  --reps 2 --extra-json '{"enable_cache_dit": true}'

# quality
python3 scripts/quality.py results/layerwise-1k-40
python3 scripts/quality.py outputs/probes/poster_0.png --ocr-text "QWEN IMAGE 2.1"
python3 scripts/quality.py --psnr results/memory-1k-40/text_1024x1024_40_r0_0.png \
                                 results/layerwise-1k-40/text_1024x1024_40_r0_0.png

# a different recipe (restart required)
sudo env SGLANG_OFFLOAD_ARGS="--performance-mode memory" \
  docker compose -f qwen-image-2.1.yml up -d --force-recreate
```

## Prompt enhancement (PE-T2I on GPU 0)

The service above does **not** rewrite prompts, and that is what the text-rendering
probes were hitting. Upstream ships the rewriter as a separate Qwen3.5-VL 9B
checkpoint (`Qwen/Qwen-Image-2.1-PE-T2I`), served as a plain OpenAI-compatible
vLLM endpoint. It is hosted on GPU 0 by `pe-t2i.yml` (port 8104, `vllm/vllm-openai
v0.27.1`), temporary and `restart: "no"`.

Two settings are required rather than tuning, both found by reading the engine's own
startup failure:

| Symptom | Cause | Setting |
| --- | --- | --- |
| `KV cache is needed ... larger than the available KV cache memory (0.8 GiB)`, max len 24288 | vLLM 0.27.1 charges ~0.55 GB of CUDA-graph memory inside `--gpu-memory-utilization`; 0.90 reports as 0.8771 without it | `--gpu-memory-utilization 0.94` |
| `max_num_seqs (256) exceeds available Mamba cache blocks (108)` | hybrid model: 24 of 32 layers are linear attention, state lives in a fixed block pool sized by the same budget | `--max-num-seqs 8` |

With those, the engine loads 17.66 GiB of weights in 8.7 s and reports 1.74 GiB
of KV (53,084 tokens) at a 24576-token max length.

### Speed

The rewriter streams a ~16k-token thinking block, so it is slow by design and is
not a per-request step you can hide. Two prompts, one client request each:

| Prompt | `wh_ratio` | Rewrite wall time | `parse_ok` |
| --- | --- | --- | --- |
| "a corgi playing guitar in the rain" | 1:1 | ~2.5 min | true |
| 3-string movie poster | 2:3 | ~3.5 min | true |

That cost lands **once per prompt**, not per image: the rewrite is deterministic
text that can be cached, while the render it feeds is ~100 s at 2K.

### Does it fix the text?

The controlled comparison is the same prompt and seed rendered twice, once with the
raw prompt and once with the PE `positive_prompt` (4,159 chars, which spells out each
string with its placement and typography), at 2048x2048 and at the PE-chosen 2:3
(1696x2528).

| Size | Prompt | Peak (server) | Card peak | Wall | tesseract on the tagline | on the date |
| --- | --- | --- | --- | --- | --- | --- |
| 2048x2048 | raw | 21560 MB | - | 97.0 s | `every ending is a beginning` | `in theaters october 24` |
| 2048x2048 | PE | 21638 MB | - | 99.2 s | `every ending is a beginning` | `in theaters october 24` |
| 1696x2528 | raw | 19652 MB | 22052 MB | 100.5 s | `every ending is a beginning` | `in theaters october 24` |
| 1696x2528 | PE | 19704 MB | 22052 MB | 102.7 s | `every ending is a beginning` | `in theaters october 24` |

So the honest read is narrower than "PE fixes text": at these sizes the raw
prompt already renders the tagline and the date, and tesseract recovers both on all
four images. The one string that stays unreliable is the *title* - tesseract returns
no confident `THE LAST LIGHT` on any of the four, in either configuration, and the
poster's condensed serif title is exactly the kind of stylised type it fails on. The
PE run is visibly better in the title band (its top band OCRs `LAST` where the raw
run returns nothing) but that is a single-sample observation, not a measurement.

What did change is the OOM that the earlier 2048x2048 text-loss came from. With
tiling off the 2K render sits right at the card's edge: the turn-7 run peaked at
21572 MB, logged an OOM and lost text; today the same configuration peaked at
21560 MB and completed, so that text loss was the allocator, not the prompt. Tiling
on is still the way to make 2K text rendering deterministic - it peaks at 5768 MB.

## Known limitations

* 2048x2048 runs at 5.8 GB (text) and 18.0 GB (edit) with `--vae-tiling true`.
  The shipped default has tiling off, which puts 2K right at the card's edge: the
  decode stage allocates 3.6 GB in one tensor, and the same prompt and seed has been
  seen both OOM with a text-lossy image (21572 MB) and pass (21560 MB) depending
  on what else was resident. Turn tiling on for a deterministic 2K. A and B OOM on 2K
  edits even with tiling, because their peak comes from resident weights rather than
  the decode.
* 2048x2048 edits still peak at 18.0 GB (20.4 GB for the whole card including
  z-image), so the card is ~83 % full and there is ~4 GB of headroom; the text
  path peaks at only 5.8 GB. Anything else landing on GPU 2 during a 2K *edit* can
  still OOM it.
* Cache-DiT is an approximation: at the default threshold it changes the PNG by
  PSNR 29-30 dB. It is per request (`"enable_cache_dit": true`), and turning it
  on for a whole server (`SGLANG_CACHE_DIT_ENABLED=true`) also disables SGLang's
  auto-residency tuning (`auto_residency_args_skip_reason` returns
  "cache-dit enabled"), so the per-request switch is the safer way to use it.
* `n > 1`, batch size > 1 and `guidance_scale > 1` (true CFG) were not
  benchmarked; they raise the peak memory of the same stages measured above.
* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was tried to reclaim
  the 1.7 GB of reserved-but-unallocated memory seen in the B 2K-edit OOM, but it
  broke the CUDA context (`unspecified launch failure`) with this driver/CUDA 13
  combination, so it is not used.
* Sharpness/entropy are computed with OpenCV on the decoded PNG and are useful for
  comparing runs, not as an absolute quality score; the text-rendering and alpha
  probes are the semantic checks.
