# Qwen-Image-2.1 OpenAI-compatible image service

Self-contained service folder for **Qwen-Image-2.1**
(<https://github.com/QwenLM/Qwen-Image-2.1>) served through
SGLang-Diffusion on **GPU 2 only**, with every component streamed from host RAM
(CPU offload), so it co-exists with the z-image service on the same card.

The service exposes the OpenAI Images API, exactly like the existing z-image
service (`/home/aiserver/z-image-turbo.yml`), so LiteLLM / OpenAI SDK clients can
call it without any change beyond the base URL.

## Layout

```
qwen-image-2.1/
  Dockerfile          SGLang image with native Qwen-Image-2.1 support + b64 default
  qwen-image-2.1.yml  docker compose service (GPU 2, CPU offload, port 7853)
  pe-t2i.yml          docker compose service (GPU 0, prompt enhancer, port 8104)
  models/             Qwen/Qwen-Image-2.1 checkpoint (mounted read-only at /model)
  outputs/            generated PNGs written by the server (also saved by the bench)
  results/            benchmark JSON/CSV/PNG artifacts
  refs/               reference images used by the edit benchmarks
  patches/            build-time source patches applied by the Dockerfile
  RESULTS.md          measured speed, VRAM and quality numbers
  scripts/bench.py    speed + quality benchmark over the OpenAI Images API
  scripts/quality.py  pixel metrics (sharpness, entropy, alpha, OCR, PSNR)
  scripts/smoke.sh    one-shot health + generation check
  logs/               download / docker build / serve logs
```

## Model

Qwen-Image-2.1 is a unified text-to-image + image-editing model:

| Component | Detail |
| --- | --- |
| Transformer (DiT) | 32 single-stream block-causal layers, ~7B params, bf16 |
| Text encoder | Qwen3-VL 8B (text + reference images in one representation) |
| VAE | 64-channel RGBA, 16x spatial compression, native transparency |
| Scheduler | Flow matching, Euler discrete, dynamic shifting |
| Defaults | 1024x1024, 40 steps, CFG 1 (guidance_scale 1.0) |
| Native sizes | 2K: 1:1 2048x2048, 16:9 2752x1536, 9:16 1536x2752, ... |

Capabilities: text-to-image, single/multi reference editing (up to 10 images),
transparent RGBA output, prefix-KV reuse for the condition prefix.

## Why the base image is a nightly

SGLang added Qwen-Image-2.1 in
[PR #39983](https://github.com/sgl-project/sglang/pull/39983), merged
2026-09-20. The `v0.5.20` release tag and the `lmsysorg/sglang:latest`
image (0.5.19) predate that merge and do not ship
`multimodal_gen/runtime/pipelines/qwen_image21.py`, so this folder builds on
`lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5` (built after the merge).
The Dockerfile fails the build if that pipeline file is missing.

## Download the model

The weights are not in git (31 GB). The compose file mounts
`models/Qwen-Image-2.1` read-only at `/model` in the container, so download
them into exactly that path before the first start:

```bash
cd /home/aiserver/qwen-image-2.1
hf download Qwen/Qwen-Image-2.1 --local-dir models/Qwen-Image-2.1
```

28 files / ~31 GB; ~5 min at ~100 MB/s on this box. `hf` is the Hugging Face
CLI (`pip install -U huggingface_hub` if `hf` is missing). Set `HF_TOKEN` if you
hit rate limits, or point `HF_ENDPOINT` at a mirror (for example
`HF_ENDPOINT=https://hf-mirror.com`). Re-running the command resumes and only
fetches what is missing.

A complete download leaves `model_index.json` plus the `transformer/`,
`text_encoder/`, `vae/`, `processor/` and `scheduler/` subfolders. The model must
be present before `docker compose up`, otherwise the container starts against an
empty `/model` and fails; `scripts/smoke.sh` fails loudly in that case.

## Build and run

```bash
cd /home/aiserver/qwen-image-2.1
sudo docker compose -f qwen-image-2.1.yml up -d --build
sudo docker logs -f qwen-image-2.1        # wait for "server is ready"
```

The model is 33 GB on disk; the first start also fills the JIT/kernel cache.

## Offload recipes

GPU 2 is shared with the existing z-image service (~1.4 GB), and a 2048x2048
request peaks near the card's capacity, so the default streams every component
from host RAM (measured 4.0 GB peak at 1024x1024, 24.7 s text / 25.1 s edit):

```bash
# default (full layerwise CPU offload, VAE tiling off)
--performance-mode manual --dit-layerwise-offload true \
  --layerwise-offload-components dit text_encoder image_encoder vae \
  --vae-tiling false
```

`--vae-tiling true` is the lever that pulls 2048x2048 well clear of the card's
limit. Without it the decode stage allocates ~3.6 GB in one tensor at 2K, which sits
right at the edge of the free memory on the shared card. Measured on the same prompt
and seed: tiling off peaks at **21572 MB** with a logged OOM and text lost, tiling on
peaks at **5768 MB** in 99.4 s with none. `--vae-cpu-offload true` does not help
here: it moves VAE *weights*, not the activations that overflow, and was measured
to change nothing.

The shipped default is tiling **off** because that is what the operator verified in
use. It leaves little headroom, so a 2K request is a coin flip depending on what
else is resident on GPU 2 - a later 2048x2048 run peaked at 21560 MB and passed.
Override it back on when you want the safety margin:

```bash
sudo env SGLANG_OFFLOAD_ARGS="--performance-mode manual --dit-layerwise-offload true --layerwise-offload-components dit text_encoder image_encoder vae --vae-tiling true" \
  sudo docker compose -f qwen-image-2.1.yml up -d --force-recreate
```

`--performance-mode memory` is ~3 s (14 %) faster at 1024x1024 (20.0 s text /
23.9 s edit) but peaks ~5 GB higher and OOMs on 2048x2048 edits. Switch recipes
with an environment variable, no file edit needed (note `sudo env`: plain `sudo`
drops the variable):

```bash
sudo env SGLANG_OFFLOAD_ARGS="--performance-mode memory" \
  sudo docker compose -f qwen-image-2.1.yml up -d --force-recreate
```

Other knobs worth trying: `--dit-layerwise-resident-layers N`,
`--layerwise-prefetch-size text_encoder=2`,
`--vae-config.tile-sample-min-height 512` (larger tiles, less overlap work).
See `RESULTS.md` for the full recipe comparison and measurements.

## OpenAI-compatible API

Both endpoints live on `http://127.0.0.1:7853` and return `b64_json` by default.

Text-to-image:

```bash
curl -s http://127.0.0.1:7853/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen-Image-2.1",
    "prompt": "a capybara reading a book by candlelight",
    "size": "1024x1024",
    "num_inference_steps": 40,
    "seed": 42
  }' | python3 -c 'import sys,json,base64;d=json.load(sys.stdin);print("infer_s",d.get("inference_time_s"));open("out.png","wb").write(base64.b64decode(d["data"][0]["b64_json"]))'
```

Image edit (multipart, one `-F image=` per reference, up to 10):

```bash
curl -s http://127.0.0.1:7853/v1/images/edits \
  -F 'model=Qwen-Image-2.1' \
  -F 'prompt=Add a glowing red paper lantern hanging above the sign' \
  -F 'size=1024x1024' \
  -F 'num_inference_steps=40' \
  -F 'seed=42' \
  -F 'image=@/home/aiserver/qwen-image-2.1/refs/neon-ref.png;type=image/png' \
  | python3 -c 'import sys,json,base64;d=json.load(sys.stdin);print("infer_s",d.get("inference_time_s"));open("edit.png","wb").write(base64.b64decode(d["data"][0]["b64_json"]))'
```

Other requests:

```bash
# models
curl -s http://127.0.0.1:7853/v1/models

# 2K text-to-image
curl -s http://127.0.0.1:7853/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen-Image-2.1","prompt":"a panoramic mountain landscape",
       "size":"2048x2048","num_inference_steps":40,"seed":42}'

# transparent RGBA sticker
curl -s http://127.0.0.1:7853/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen-Image-2.1","prompt":"This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent.",
       "size":"1024x1024","num_inference_steps":40,"background":"transparent","output_format":"png"}'
```

Responses look like the OpenAI images API plus SGLang's own telemetry:

```json
{
  "id": "...", "created": 1790000000,
  "data": [{"b64_json": "iVBORw0...", "url": null}],
  "peak_memory_mb": 12345.6,
  "inference_time_s": 12.34,
  "usage": {"prompt_tokens": 123, "image_count": 1}
}
```

Requests also accept the SGLang extensions `width`/`height` (multiples of 32),
`guidance_scale`, `negative_prompt`, `n`, `output_format`, `background`.

Notes that matter in practice:

* `size` must be divisible by 32, otherwise the request 500s (this is what the
  720x720 startup warm-up tripped over). Native sizes: `1024x1024`,
  `2048x2048`, `2752x1536`, `1536x2752`, `2400x1792`, `1792x2400`,
  `2528x1696`, `1696x2528`.
* `background: "transparent"` returns a real RGBA cutout; `output_format`
  defaults to `png` (see the Dockerfile patch) so alpha survives.
* 2048x2048 edits only fit with the default full-offload recipe and take
  ~140 s and peak at 20.4 GB for the whole card; 1024x1024 generation is ~25 s
  and peaks at 4.0 GB.
* `guidance_scale` defaults to 1.0 (CFG off) and `n` defaults to 1.
* OpenAI SDK / LiteLLM clients only need the base URL `http://127.0.0.1:7853/v1`.

## Text rendering and prompt rewriting

This SGLang service does **not** rewrite prompts. `--enable-prompt-rewrite`
exists in SGLang but is wired only for LongCat-Image and Ernie-Image; for
Qwen-Image-2.1 the prompt goes straight into the Qwen3-VL text encoder. Short
quoted strings usually render exactly (tesseract reads `HELLO WORLD` and
`QWEN IMAGE 2.1` back verbatim at 1024x1024), while longer multi-string
layouts (title + tagline + date) often lose text.

The upstream repo ships the missing piece as a separate model, not as part of the
serving stack: `prompt_rewrite/` with two Qwen3.5-VL 9B checkpoints, ~18.8 GB each
in bf16.

| Checkpoint | Purpose |
| --- | --- |
| `Qwen/Qwen-Image-2.1-PE-T2I` | expands a text-to-image prompt |
| `Qwen/Qwen-Image-2.1-PE-I2I` | rewrites an edit instruction |

They return JSON `{"rewritten_prompt": ..., "wh_ratio": "16:9"}` and are served as
an OpenAI-compatible vLLM endpoint (`prompt_rewrite/serve.sh`, port 8100). `wh_ratio`
maps to a canvas via `WH_RATIO_TO_SIZE` (1:1 2048x2048, 4:3 2400x1792, 3:4
1792x2400, 3:2 2528x1696, 2:3 1696x2528, 16:9 2752x1536, 9:16 1536x2752).

That service cannot run inside this container: `prompt_rewrite/requirements.txt`
pins `vllm==0.19.1` and `transformers==5.4.0`, while this container ships
transformers 5.12.1 and has no `vllm` module. It runs as a second container
instead. `pe-t2i.yml` hosts the T2I checkpoint on the now-free GPU 0 as a plain
vLLM OpenAI endpoint on port **8104**, with both checkpoints in
`models/Qwen-Image-2.1-PE-T2I` and `models/Qwen-Image-2.1-PE-I2I` (gitignored);
see `logs/pe-download.log`.

Two vLLM settings there are required rather than tuning. `--gpu-memory-utilization
0.94` because vLLM 0.27.1 charges ~0.55 GB of CUDA-graph memory against the
same budget, so 0.90 left 0.8 GB of KV against the 0.81 GB a 24576-token
sequence needs and the engine refused to start. `--max-num-seqs 8` because this is
a hybrid model - 24 of 32 layers are linear attention, whose state lives in a fixed
pool of "Mamba cache blocks" - and only 108 blocks fit at that budget, below
vLLM's default of 256, so the engine aborted with `max_num_seqs (256) exceeds
available Mamba cache blocks (108)`.

```bash
sudo docker compose -f pe-t2i.yml up -d
curl -sf http://127.0.0.1:8104/health && echo ok
```

The server holds the weights but **not** the task prompt, so the caller must send
it. Use the upstream client (it is what produces the JSON record):

```bash
cd /path/to/Qwen-Image-2.1/prompt_rewrite
python3 client.py --task t2i --model Qwen/Qwen-Image-2.1-PE-T2I \
  --system-prompt prompts/system_prompt_t2i.txt --port 8104 \
  "a corgi playing guitar in the rain"
```

It streams a ~16k-token thinking block, so expect minutes per prompt, and returns
`{positive_prompt, negative_prompt, wh_ratio, parse_ok}`. Feed `positive_prompt` to
`/v1/images/generations` and take `size` from `wh_ratio`. This host is temporary:
`restart: "no"`, and the bonsai unit on GPU 0 is still `enable`d, so a reboot
reclaims the card.

## Host GPU budget

GPU 2 is 24 GB and shared with z-image (~1.4 GB resident). The 2K text path now
peaks at 8.4 GB for the whole card and the 2K edit path at 20.4 GB, so 2K edits
still leave only ~4 GB of slack. GPU 0 (llama-server `bonsai-2-27b`, ~17 GB) was
stopped for this work and now hosts the prompt enhancer (`qwen-pe-t2i`, ~21 GB,
`pe-t2i.yml`); it is still `systemctl enable`d, so `sudo systemctl start
bonsai-2-27b` brings it back and it will also return on reboot, which would
collide with the prompt enhancer on GPU 0. GPU 1 (vLLM, ~17.9 GB) was not
touched.

## Measurement

```bash
python3 scripts/bench.py --tag layerwise-1k-40 --sizes 1024x1024 --steps 40 --reps 3
```

`bench.py` warms up once, then repeats each (size, steps) cell, recording
client wall time, server `inference_time_s`, server `peak_memory_mb`, PNG size and
GPU-2 VRAM sampled with `nvidia-smi`. Artifacts land in `results/<tag>/`.

`scripts/quality.py` scores those PNGs (sharpness, entropy, alpha, OCR, PSNR), and
`scripts/smoke.sh` is the one-shot health + generation check.

See `RESULTS.md` for the measured speed, VRAM and quality numbers; the raw
artifacts are the `results/<tag>/` directories.

The prompt-enhancer comparison in `RESULTS.md` was run by hand: the PE client
prints the rewrite record, then the `positive_prompt` goes to
`POST http://127.0.0.1:7853/v1/images/generations` at the size implied by
`wh_ratio`, and `tesseract` reads the strings back out of the PNG.
