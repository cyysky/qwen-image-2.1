"""FastAPI app: serves the playground UI and proxies the image endpoints."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import mimetypes
import time

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import enhancer as enhancer_module
from . import image_api
from .config import FRONTEND_DIR, Settings, get_settings, reload_settings
from .errors import BadRequest, UpstreamError
from .schemas import EnhanceRequest, GenerateRequest

app = FastAPI(
    title="Qwen-Image-2.1 Playground",
    version="1.0.0",
    description=(
        "Prompt testing UI for Qwen-Image-2.1 text-to-image and image-edit "
        "endpoints, with ds4-flash prompt enhancement."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(UpstreamError)
async def upstream_error_handler(_: Request, exc: UpstreamError) -> JSONResponse:
    status = exc.status if 400 <= exc.status < 600 else 502
    return JSONResponse(status_code=status, content=exc.to_dict())


@app.exception_handler(BadRequest)
async def bad_request_handler(_: Request, exc: BadRequest) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": str(exc)})


# --------------------------------------------------------------------------
# config / health
# --------------------------------------------------------------------------
@app.get("/api/config")
async def api_config() -> dict:
    """Everything the UI needs: endpoints, models, sizes, defaults, presets."""
    return get_settings().to_public_dict()


@app.post("/api/config/reload")
async def api_config_reload() -> dict:
    """Re-read `.env` so endpoints/models can change without a restart."""
    settings = reload_settings()
    return {"ok": True, "config": settings.to_public_dict()}


async def _probe(settings: Settings, name: str) -> dict:
    endpoint = settings.endpoint(name)
    started = time.perf_counter()
    try:
        data, _, _ = await image_api.list_models(
            settings, endpoint, timeout_s=settings.health_timeout_s
        )
    except UpstreamError as exc:
        return {
            "name": endpoint.name,
            "label": endpoint.display,
            "base_url": endpoint.base_url,
            "ok": False,
            "status": exc.status,
            "error": exc.message,
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }
    models = [
        item.get("id")
        for item in (data.get("data") or [])
        if isinstance(item, dict) and item.get("id")
    ]
    return {
        "name": endpoint.name,
        "label": endpoint.display,
        "base_url": endpoint.base_url,
        "ok": True,
        "status": 200,
        "models": models,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }


@app.get("/api/health")
async def api_health(endpoint: str | None = None) -> dict:
    """Probe one endpoint (or all of them) for reachability and model list."""
    settings = get_settings()
    if endpoint:
        return {"results": [await _probe(settings, endpoint)]}
    results = await asyncio.gather(
        *(_probe(settings, item.name) for item in settings.endpoints)
    )
    return {"results": list(results)}


@app.get("/api/models")
async def api_models(endpoint: str | None = None) -> dict:
    """Proxy the upstream /v1/models list for one endpoint."""
    settings = get_settings()
    target = settings.endpoint(endpoint)
    data, elapsed, url = await image_api.list_models(
        settings, target, timeout_s=settings.health_timeout_s
    )
    return {
        "endpoint": target.name,
        "base_url": target.base_url,
        "elapsed_s": round(elapsed, 2),
        "url": url,
        "models": data.get("data") if isinstance(data, dict) else data,
        "raw": data,
    }


# --------------------------------------------------------------------------
# prompt enhancement
# --------------------------------------------------------------------------
@app.post("/api/enhance")
async def api_enhance(req: EnhanceRequest) -> dict:
    settings = get_settings()
    result = await enhancer_module.enhance_prompt(
        settings,
        prompt=req.prompt,
        mode=req.mode,
        instruction=req.instruction,
        is_edit=req.is_edit,
        negative_prompt=req.negative_prompt,
        size=req.size,
        model=req.model,
        engine=req.engine,
        images=req.images,
        image_sizes=req.image_sizes,
    )
    return result


def _suggested_size(meta: dict | None) -> str:
    """Canvas a PE rewrite asked for, when it names one and the caller did not."""
    return (meta or {}).get("suggested_size") or ""


def _data_url(mime: str, blob: bytes) -> str:
    """Inline a reference image as a `data:` URL for a multimodal enhancer."""
    encoded = base64.b64encode(blob).decode("ascii")
    return f"data:{mime or 'image/png'};base64,{encoded}"


def _parse_sizes(raw: str | None) -> list[str]:
    """Split a `1024x1024,1696x2528` form field into a list."""
    return [item.strip() for item in (raw or "").replace("\n", ",").split(",") if item.strip()]


# --------------------------------------------------------------------------
# text-to-image
# --------------------------------------------------------------------------
@app.post("/api/generate")
async def api_generate(req: GenerateRequest) -> dict:
    settings = get_settings()
    endpoint = settings.endpoint(req.endpoint)

    prompt = req.prompt.strip()
    original_prompt = prompt
    enhanced_meta: dict | None = None

    if req.enhance:
        enhanced_meta = await enhancer_module.enhance_prompt(
            settings,
            prompt=prompt,
            mode=req.enhance_mode,
            instruction=req.enhance_instruction,
            is_edit=False,
            negative_prompt=req.negative_prompt,
            size=req.size or settings.default_size,
            model=req.enhance_model,
            engine=req.enhance_engine,
        )
        prompt = enhanced_meta["prompt"]
        if not req.size and _suggested_size(enhanced_meta):
            # The PE checkpoint picked a canvas and the caller did not, so follow
            # it instead of letting the server fall back to its own default.
            req.size = _suggested_size(enhanced_meta)

    payload = image_api.build_generation_payload(
        settings,
        prompt=prompt,
        model=req.model,
        negative_prompt=req.negative_prompt,
        size=req.size,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        n=req.n,
        output_format=req.output_format,
        background=req.background,
    )

    data, elapsed, url = await image_api.generate_image(settings, endpoint, payload)
    return image_api.normalize_image_response(
        data,
        endpoint=endpoint,
        model=payload["model"],
        prompt=prompt,
        original_prompt=original_prompt,
        params={k: v for k, v in payload.items() if k not in {"prompt", "model"}},
        elapsed_s=elapsed,
        url=url,
        enhanced=enhanced_meta,
    )


# --------------------------------------------------------------------------
# image edit
# --------------------------------------------------------------------------
def _content_type(upload: UploadFile) -> str:
    if upload.content_type and upload.content_type != "application/octet-stream":
        return upload.content_type
    guessed, _ = mimetypes.guess_type(upload.filename or "")
    return guessed or "image/png"


async def _collect_references(
    settings: Settings, images: list[UploadFile], reference_b64: str | None
) -> list[tuple[str, tuple[str, bytes, str]]]:
    """Merge uploaded files and base64 payloads into multipart parts."""
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for index, upload in enumerate(images or [], start=1):
        blob = await upload.read()
        if not blob:
            continue
        name = upload.filename or f"reference-{index}.png"
        files.append(("image", (name, blob, _content_type(upload))))

    if reference_b64:
        try:
            payload = json.loads(reference_b64)
        except json.JSONDecodeError as exc:
            raise BadRequest("reference_b64 must be a JSON array of base64 images") from exc
        if not isinstance(payload, list):
            payload = [payload]
        for index, item in enumerate(payload, start=len(files) + 1):
            if not isinstance(item, str) or not item.strip():
                continue
            raw = item.strip()
            mime = "image/png"
            if raw.startswith("data:"):
                header, _, raw = raw.partition(",")
                mime = header[5:].split(";")[0] or mime
            try:
                blob = base64.b64decode(raw, validate=False)
            except (binascii.Error, ValueError) as exc:
                raise BadRequest(f"Reference image {index} is not valid base64") from exc
            files.append(("image", (f"reference-{index}.png", blob, mime)))

    if not files:
        raise BadRequest("An edit needs at least one reference image")
    if len(files) > settings.max_reference_images:
        raise BadRequest(
            f"Too many reference images: {len(files)} "
            f"(MAX_REFERENCE_IMAGES={settings.max_reference_images})"
        )
    return files


@app.post("/api/edit")
async def api_edit(
    prompt: str = Form(...),
    endpoint: str | None = Form(None),
    model: str | None = Form(None),
    negative_prompt: str | None = Form(None),
    size: str | None = Form(None),
    width: int | None = Form(None),
    height: int | None = Form(None),
    num_inference_steps: int | None = Form(None),
    guidance_scale: float | None = Form(None),
    seed: int | None = Form(None),
    n: int | None = Form(None),
    output_format: str | None = Form(None),
    background: str | None = Form(None),
    enhance: bool = Form(False),
    enhance_mode: str | None = Form(None),
    enhance_instruction: str | None = Form(None),
    enhance_model: str | None = Form(None),
    enhance_engine: str | None = Form(None),
    reference_b64: str | None = Form(None),
    reference_sizes: str | None = Form(None),
    image: list[UploadFile] = File(default=[]),
    images: list[UploadFile] = File(default=[]),
) -> dict:
    """Image edit: multipart upload with up to MAX_REFERENCE_IMAGES references."""
    settings = get_settings()
    target = settings.endpoint(endpoint)

    uploads = [item for item in (list(image or []) + list(images or []))]
    files = await _collect_references(settings, uploads, reference_b64)

    prompt = (prompt or "").strip()
    if not prompt:
        raise BadRequest("An edit prompt is required")
    original_prompt = prompt
    enhanced_meta: dict | None = None

    if enhance:
        enhanced_meta = await enhancer_module.enhance_prompt(
            settings,
            prompt=prompt,
            mode=enhance_mode,
            instruction=enhance_instruction,
            is_edit=True,
            negative_prompt=negative_prompt,
            size=size or settings.default_size,
            model=enhance_model,
            engine=enhance_engine,
            images=[_data_url(mime, blob) for _, (_, blob, mime) in files],
            image_sizes=_parse_sizes(reference_sizes),
        )
        prompt = enhanced_meta["prompt"]
        if not size and _suggested_size(enhanced_meta):
            # The edit checkpoint kept a source image's shape, so render at the
            # matching canvas rather than the server's default.
            size = _suggested_size(enhanced_meta)

    form = image_api.build_edit_form(
        settings,
        prompt=prompt,
        model=model,
        negative_prompt=negative_prompt,
        size=size,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        n=n,
        output_format=output_format,
        background=background,
    )

    data, elapsed, url = await image_api.edit_image(settings, target, form, files)
    return image_api.normalize_image_response(
        data,
        endpoint=target,
        model=form["model"],
        prompt=prompt,
        original_prompt=original_prompt,
        params={k: v for k, v in form.items() if k not in {"prompt", "model"}},
        elapsed_s=elapsed,
        url=url,
        enhanced=enhanced_meta,
    )


# --------------------------------------------------------------------------
# static frontend
# --------------------------------------------------------------------------
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico")
async def favicon() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "favicon.svg", media_type="image/svg+xml")


if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
