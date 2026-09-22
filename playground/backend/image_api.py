"""Thin async client for the SGLang / OpenAI-compatible image endpoints."""

from __future__ import annotations

import json
import time

import httpx

from .config import ImageEndpoint, Settings
from .errors import BadRequest, UpstreamError


def _headers(endpoint: ImageEndpoint, extra: dict | None = None) -> dict:
    headers = dict(extra or {})
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    return headers


def _timeout(seconds: float) -> httpx.Timeout:
    return httpx.Timeout(seconds, connect=min(30.0, seconds))


async def _post(
    url: str,
    *,
    endpoint: ImageEndpoint,
    timeout_s: float,
    json_body: dict | None = None,
    data: dict | None = None,
    files: list | None = None,
) -> tuple[dict, float]:
    """POST to an endpoint, returning (parsed json, elapsed seconds)."""
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_timeout(timeout_s)) as client:
            response = await client.post(
                url,
                json=json_body,
                data=data,
                files=files,
                headers=_headers(endpoint),
            )
    except httpx.TimeoutException as exc:
        raise UpstreamError(
            504,
            f"{endpoint.display} timed out after {timeout_s:.0f}s",
            url=url,
        ) from exc
    except httpx.RequestError as exc:
        raise UpstreamError(
            502, f"Cannot reach {endpoint.display} ({url}): {exc}", url=url
        ) from exc

    elapsed = time.perf_counter() - started

    if response.status_code >= 400:
        raise UpstreamError(
            response.status_code,
            f"{endpoint.display} returned HTTP {response.status_code}",
            url=url,
            body=response.text,
        )

    try:
        return response.json(), elapsed
    except json.JSONDecodeError as exc:
        raise UpstreamError(
            502,
            f"{endpoint.display} returned a non-JSON response",
            url=url,
            body=response.text,
        ) from exc


def _validate_size(size: str | None, width: int | None, height: int | None) -> None:
    """SGLang 500s on sizes that are not multiples of 32, so catch it early."""
    for label, value in (("size", size), ("width", width), ("height", height)):
        if value is None:
            continue
        if label == "size":
            parts = str(value).lower().replace(" ", "").split("x")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise BadRequest(f"size must look like WIDTHxHEIGHT, got '{value}'")
            numbers = [int(part) for part in parts]
        else:
            numbers = [int(value)]
        for number in numbers:
            if number <= 0 or number % 32 != 0:
                raise BadRequest(
                    f"{label} {number} is not a multiple of 32 - SGLang rejects it "
                    "(try 1024x1024, 2048x2048 or 2752x1536)"
                )


def build_generation_payload(
    settings: Settings,
    *,
    prompt: str,
    model: str | None = None,
    negative_prompt: str | None = None,
    size: str | None = None,
    width: int | None = None,
    height: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    n: int | None = None,
    output_format: str | None = None,
    background: str | None = None,
) -> dict:
    """Build the /v1/images/generations body, dropping unset fields."""
    _validate_size(size, width, height)
    payload: dict = {
        "model": model or settings.image_model,
        "prompt": prompt,
        "response_format": settings.response_format,
    }
    optional = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "n": n,
        "output_format": output_format or settings.default_output_format,
        "background": background,
    }
    if width is not None and height is not None:
        payload["width"] = width
        payload["height"] = height
    elif size:
        payload["size"] = size
    for key, value in optional.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        payload[key] = value
    return payload


def build_edit_form(
    settings: Settings,
    *,
    prompt: str,
    model: str | None = None,
    negative_prompt: str | None = None,
    size: str | None = None,
    width: int | None = None,
    height: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    n: int | None = None,
    output_format: str | None = None,
    background: str | None = None,
) -> dict[str, str]:
    """Build the multipart fields for /v1/images/edits."""
    _validate_size(size, width, height)
    form: dict[str, str] = {
        "model": model or settings.image_model,
        "prompt": prompt,
        "response_format": settings.response_format,
        "output_format": output_format or settings.default_output_format,
    }
    optional = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "n": n,
        "background": background,
    }
    if width is not None and height is not None:
        form["width"] = str(width)
        form["height"] = str(height)
    elif size:
        form["size"] = size
    for key, value in optional.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        form[key] = str(value)
    return form


async def generate_image(
    settings: Settings, endpoint: ImageEndpoint, payload: dict
) -> tuple[dict, float, str]:
    """POST /v1/images/generations."""
    url = f"{endpoint.base_url}/images/generations"
    data, elapsed = await _post(
        url, endpoint=endpoint, timeout_s=settings.image_timeout_s, json_body=payload
    )
    return data, elapsed, url


async def edit_image(
    settings: Settings,
    endpoint: ImageEndpoint,
    form: dict[str, str],
    files: list[tuple[str, tuple[str, bytes, str]]],
) -> tuple[dict, float, str]:
    """POST /v1/images/edits with one `image` part per reference."""
    url = f"{endpoint.base_url}/images/edits"
    data, elapsed = await _post(
        url,
        endpoint=endpoint,
        timeout_s=settings.image_timeout_s,
        data=form,
        files=files,
    )
    return data, elapsed, url


async def list_models(
    settings: Settings, endpoint: ImageEndpoint, timeout_s: float | None = None
) -> tuple[dict, float, str]:
    """GET /v1/models, used by the UI's model picker and the health probe."""
    url = f"{endpoint.base_url}/models"
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            timeout=_timeout(timeout_s or settings.health_timeout_s)
        ) as client:
            response = await client.get(url, headers=_headers(endpoint))
    except httpx.TimeoutException as exc:
        raise UpstreamError(
            504, f"{endpoint.display} timed out on /models", url=url
        ) from exc
    except httpx.RequestError as exc:
        raise UpstreamError(
            502, f"Cannot reach {endpoint.display} ({url}): {exc}", url=url
        ) from exc

    elapsed = time.perf_counter() - started
    if response.status_code >= 400:
        raise UpstreamError(
            response.status_code,
            f"{endpoint.display} returned HTTP {response.status_code} on /models",
            url=url,
            body=response.text,
        )
    try:
        return response.json(), elapsed, url
    except json.JSONDecodeError as exc:
        raise UpstreamError(
            502, f"{endpoint.display} returned a non-JSON response", url=url,
            body=response.text,
        ) from exc


def normalize_image_response(
    data: dict,
    *,
    endpoint: ImageEndpoint,
    model: str,
    prompt: str,
    original_prompt: str | None,
    params: dict,
    elapsed_s: float,
    url: str,
    enhanced: dict | None = None,
) -> dict:
    """Merge SGLang telemetry with the request echo the UI relies on."""
    if not isinstance(data, dict):
        raise UpstreamError(
            502, "Upstream returned an unexpected payload shape", url=url
        )
    items = data.get("data")
    if not isinstance(items, list) or not items:
        raise UpstreamError(
            502, "Upstream returned no images", url=url, body=json.dumps(data)[:2000]
        )

    images = []
    for item in items:
        if not isinstance(item, dict):
            continue
        images.append(
            {
                "b64_json": item.get("b64_json"),
                "url": item.get("url"),
                "revised_prompt": item.get("revised_prompt"),
            }
        )
    if not images:
        raise UpstreamError(
            502, "Upstream returned no usable images", url=url, body=json.dumps(data)[:2000]
        )

    result = dict(data)
    result["data"] = images
    result["request"] = {
        "endpoint": endpoint.name,
        "endpoint_label": endpoint.display,
        "base_url": endpoint.base_url,
        "model": model,
        "prompt": prompt,
        "original_prompt": original_prompt if original_prompt != prompt else None,
        "enhanced": enhanced,
        "params": params,
    }
    result["wall_time_s"] = round(elapsed_s, 2)
    return result
