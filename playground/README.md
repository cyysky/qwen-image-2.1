# Qwen-Image-2.1 Playground

A small FastAPI backend plus a no-build frontend for testing prompts and edits
against Qwen-Image-2.1, with a **selectable prompt enhancer** - `ds4-flash` on
`router.pixel-space.co`, the **Qwen-Image-2.1-PE-T2I** text-to-image checkpoint,
or the **Qwen-Image-2.1-PE-I2I** edit checkpoint - rewriting prompts at the UI
level before anything is submitted to the image endpoint.

Every endpoint, model, key and default lives in `.env` - no code changes needed to
re-point the app.

## Quick start

```powershell
pip install -r requirements.txt
python app.py                 # http://127.0.0.1:7860
```

Then open the printed URL. `.\run.ps1` does the same, and
`.\run.ps1 -Check` prints the resolved config and probes every endpoint.

## What the UI does

- **Generate tab** - prompt, model, size/steps/guidance/seed/n/format/background,
  negative prompt. `Ctrl+Enter` runs, `Ctrl+Shift+Enter` enhances.
- **Edit tab** - an edit instruction plus up to `MAX_REFERENCE_IMAGES` references
  (drag and drop or browse; one `-F image=` per reference), or reuse any generated
  image straight from the gallery via *Use as reference*.
- **Enhance with enhancer** - sends the prompt (and the selected preset plus any
  extra instruction) to the selected enhancer engine, shows the rewrite next to the
  original, and lets you *Use* / *Revert* it. Tick *Enhance the prompt before
  submitting* to have the backend rewrite and submit in one request.
- **Enhancer engine picker** - `ds4-flash` (plain-text rewrite) or
  `pe-t2i` / `pe-i2i` (Qwen-Image-2.1-PE checkpoints). A PE engine also
  returns a `wh_ratio`, so the panel shows the ratio and offers *Apply suggested
  size* (the matching native canvas) plus its negative prompt. `pe-i2i` reads the
  reference images and may answer with `ratio_follow` (`<imageN>`) instead, keeping
  that reference's shape. Each checkpoint carries only one task prompt, so `pe-t2i`
  is greyed out on the Edit tab and `pe-i2i` on the Generate tab.
- **Results** - images with SGLang telemetry (`inference_time_s`,
  `peak_memory_mb`, tokens), a collapsed raw JSON view, download (single or all),
  click-to-zoom. Results are stored in **IndexedDB**, so they survive a refresh;
  the prompt history stays in `localStorage`.
- **Copy as curl** - a ready-to-paste curl for whatever is currently in the form.

The size field warns before submitting if a value is not a multiple of 32, which is
what makes SGLang 500 (the old `720x720` warm-up).

## Configuration (`.env`)

Copy `.env.example` to `.env` and edit. Blank values fall back to the built-in
defaults, so deleting a line never breaks the app.

| Variable | Meaning |
| --- | --- |
| `IMAGE_ENDPOINTS` | Comma-separated `name\|base_url\|api_key\|label`. A missing `/v1` is appended automatically. |
| `API_KEY_<NAME>` | Per-endpoint key, e.g. `API_KEY_PIXEL_SPACE`. Overrides the inline key. |
| `DEFAULT_ENDPOINT` | Endpoint selected on first load. |
| `APPEND_V1` | Set `false` to stop auto-appending `/v1`. |
| `IMAGE_MODEL` / `IMAGE_MODELS` | Model sent by default / offered in the picker. |
| `SIZES` / `DEFAULT_SIZE` | Sizes offered in the UI / the one preselected. |
| `DEFAULT_STEPS`, `DEFAULT_GUIDANCE_SCALE`, `DEFAULT_OUTPUT_FORMAT` | Generation defaults. |
| `IMAGE_RESPONSE_FORMAT` | `b64_json` (default) or `url`. |
| `MAX_REFERENCE_IMAGES` | Cap on edit references. |
| `IMAGE_TIMEOUT_S` | Image request timeout; 900s leaves room for 2048x2048 edits (~135s). |
| `ENHANCER_ENABLED` | Master switch for the prompt enhancer. |
| `ENHANCER_ENGINES` | Comma-separated enhancer engine keys, in UI order. The first is the default. |
| `ENHANCER_ENGINE` | Engine selected on first load. |
| `ENHANCER_ENGINE_<KEY>_*` | Per-engine settings: `LABEL`, `KIND` (`chat`, `pe` or `pe-i2i`), `BASE_URL`, `CHAT_PATH`, `MODEL`, `API_KEY`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `MIN_P`, `PRESENCE_PENALTY`, `MAX_TOKENS`, `TIMEOUT_S`, `SEED`, `ENABLE_THINKING`, `IMAGE_INPUTS`, `SYSTEM_PROMPT`, `EDIT_SYSTEM_PROMPT`, `SYSTEM_PROMPT_FILE`, `EDIT_SYSTEM_PROMPT_FILE`. `<KEY>` is the engine key uppercased with dashes turned into underscores. |
| `ENHANCER_BASE_URL`, `ENHANCER_CHAT_PATH`, `ENHANCER_MODEL`, `ENHANCER_API_KEY`, `ENHANCER_TEMPERATURE`, `ENHANCER_MAX_TOKENS`, `ENHANCER_TIMEOUT_S`, `ENHANCER_SYSTEM_PROMPT`, `ENHANCER_EDIT_SYSTEM_PROMPT` | Legacy single-engine overrides, kept for older `.env` files. They apply to the first engine in `ENHANCER_ENGINES` only. |
| `ENHANCER_PRESETS`, `ENHANCER_PRESET_<KEY>` | Preset keys and the instruction behind each key. |
| `ENHANCER_APPLY_BY_DEFAULT` | Pre-tick *Enhance the prompt before submitting*. |
| `APP_HOST`, `APP_PORT`, `CORS_ORIGINS` | Where the UI is served. |

### Enhancer engines

Two contracts are built in; both are ordinary OpenAI chat-completions calls, so
any compatible server works:

- `chat` (default key `ds4-flash`) - replies with the rewritten prompt as plain
  text. Used for text-to-image and edit.
- `pe` (default key `pe-t2i`) - a Qwen-Image-2.1-PE checkpoint served by vLLM.
  It answers with `{"rewritten_prompt": ..., "wh_ratio": ...}`, sampled with
  upstream's t2i profile (`temperature 1.0`, `top_p 0.95`, `top_k 20`,
  `presence_penalty 1.5`, `seed 42`). The PE checkpoints ship their task prompt
  next to the weights rather than in a repo, so a copy of the T2I prompt is bundled
  at `prompts/pe_t2i_system_prompt.txt` and loaded via
  `ENHANCER_ENGINE_PE_T2I_SYSTEM_PROMPT_FILE`.
- `pe-i2i` (default key `pe-i2i`) - the Qwen-Image-2.1-PE-I2I edit checkpoint,
  the same contract with the reference images in the user message
  (`IMAGE_INPUTS=true`). It answers with `{"rewritten_prompt": ...,
  "wh_ratio": ...}` or, when the render should keep a source image's shape,
  `{"rewritten_prompt": ..., "ratio_follow": "<imageN>"}`. The edit profile is
  upstream's (`temperature 1.0`, `top_p 0.95`, `top_k 20`, `presence_penalty 0.0`,
  `max_tokens 24000`, `seed 42`), the bundled prompt is
  `prompts/pe_i2i_system_prompt.txt` and `ENABLE_THINKING` is left blank so the
  host's `--reasoning-parser qwen3` decides.

`wh_ratio` maps to a native canvas: `1:1 2048x2048`, `4:3 2400x1792`,
`3:4 1792x2400`, `3:2 2528x1696`, `2:3 1696x2528`, `16:9 2752x1536`,
`9:16 1536x2752`. If a PE answer is not a clean JSON record the raw answer is
shown with a warning instead of being dropped.

`ratio_follow` names a reference image (`<image1>`..`<imageN>`), so the UI sends
each reference's real pixel size alongside the images and the backend maps the name
to that canvas (snapped to a multiple of 32). When the answer suggests a size and the
caller did not pin one, `/api/generate` and `/api/edit` follow it.

Edit `.env` while the server runs, then hit **Reload .env** in the header (or
`POST /api/config/reload`). Keys are never sent to the browser - the UI only learns
whether one is set.

## HTTP API

| Route | Purpose |
| --- | --- |
| `GET /api/config` | Endpoints, models, sizes, defaults, presets. |
| `POST /api/config/reload` | Re-read `.env`. |
| `GET /api/health?endpoint=` | Probe one endpoint, or all of them. |
| `GET /api/models?endpoint=` | Proxy the upstream `/v1/models`. |
| `POST /api/enhance` | Rewrite a prompt with the selected `engine` (`is_edit` for the edit task). |
| `POST /api/generate` | Text-to-image. |
| `POST /api/edit` | Image edit (multipart, one `image` part per reference). |

`/api/generate` and `/api/edit` return the upstream payload unchanged plus
`request` (endpoint, model, prompt, original prompt, params) and `wall_time_s`, so
the UI can show what was actually sent.

```bash
curl -s http://127.0.0.1:7860/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"endpoint":"remote","prompt":"a capybara reading a book by candlelight",
       "size":"1024x1024","num_inference_steps":40,"seed":42}'
```

## Tests

`tools/smoke_test.py` boots a mock upstream (no GPU needed) plus the app on spare
ports and checks every route, including enhancer rewriting, the multiple-of-32 guard,
transparent output, multipart edits and reference reuse:

```powershell
python tools/smoke_test.py
```

`tools/mock_upstream.py` is that stand-in on its own, handy for poking at the UI
offline:

```powershell
python tools/mock_upstream.py --port 7899
```

## Notes

- `size` must be divisible by 32; native sizes are `1024x1024`, `2048x2048`,
  `2752x1536`, `1536x2752`, `2400x1792`, `1792x2400`, `2528x1696`, `1696x2528`.
- `background: transparent` returns a real RGBA cutout, so `output_format` is forced
  to `png`.
- `guidance_scale` defaults to 1.0 (CFG off) and `n` to 1.
