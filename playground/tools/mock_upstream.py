#!/usr/bin/env python3
"""A tiny stand-in for the SGLang image endpoint and the router.

Useful to exercise the playground UI (and the smoke test) without touching the
real GPUs:

    python tools/mock_upstream.py --port 7899

It implements just enough of the OpenAI surface:

    GET  /v1/models
    POST /v1/images/generations
    POST /v1/images/edits
    POST /v1/chat/completions     (canned prompt; PE models get the PE contract)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import time
import zlib

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock Qwen-Image upstream")

MAX_SIDE = int(os.getenv("MOCK_MAX_SIDE", "256"))

#: `wh_ratio` values a Qwen-Image-2.1-PE checkpoint may answer with.
PE_RATIOS = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16")


def is_pe_model(model: str) -> bool:
    return "PE-T2I" in (model or "").upper()


def is_pe_i2i_model(model: str) -> bool:
    return "PE-I2I" in (model or "").upper()


def user_text(content) -> str:
    """Flatten a chat message's content, which may be a multimodal part list."""
    if isinstance(content, list):
        for part in reversed(content):
            if isinstance(part, dict) and part.get("type") == "text":
                return str(part.get("text") or "")
        return ""
    return str(content or "")


def reference_count(content) -> int:
    if not isinstance(content, list):
        return 0
    return len([part for part in content if isinstance(part, dict) and part.get("type") == "image_url"])


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def png_bytes(width: int, height: int, rgb: tuple[int, int, int], alpha: int | None = None) -> bytes:
    """Encode a solid colour PNG without needing Pillow."""
    channels = 4 if alpha is not None else 3
    color_type = 6 if alpha is not None else 2
    pixel = bytes(rgb) + (bytes([alpha]) if alpha is not None else b"")
    raw = b"".join(b"\x00" + pixel * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )


def parse_size(value: str | None, width: int | None, height: int | None) -> tuple[int, int]:
    if width and height:
        return min(int(width), MAX_SIDE), min(int(height), MAX_SIDE)
    if value and "x" in value.lower():
        left, _, right = value.lower().partition("x")
        if left.strip().isdigit() and right.strip().isdigit():
            return min(int(left), MAX_SIDE), min(int(right), MAX_SIDE)
    return 128, 128


def image_response(prompt: str, size: tuple[int, int], background: str | None, count: int, seed: int | None) -> dict:
    width, height = size
    transparent = (background or "").lower() == "transparent"
    alpha = 0 if transparent else None
    # Tint by prompt length so different prompts look different.
    rgb = (60 + len(prompt) % 120, 90, 200)
    blob = png_bytes(width, height, rgb, alpha)
    payload = base64.b64encode(blob).decode("ascii")
    return {
        "id": f"mock-{int(time.time())}",
        "created": int(time.time()),
        "data": [{"b64_json": payload, "url": None} for _ in range(max(1, count))],
        "peak_memory_mb": 12345.6,
        "inference_time_s": 1.25,
        "usage": {"prompt_tokens": len(prompt.split()), "image_count": max(1, count)},
        "mock": {"size": f"{width}x{height}", "background": background, "seed": seed},
    }


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": "Qwen-Image-2.1", "object": "model", "owned_by": "mock"},
            {"id": "ds4-flash", "object": "model", "owned_by": "mock"},
            {"id": "Qwen/Qwen-Image-2.1-PE-T2I", "object": "model", "owned_by": "mock"},
            {"id": "Qwen/Qwen-Image-2.1-PE-I2I", "object": "model", "owned_by": "mock"},
        ],
    }


@app.post("/v1/images/generations")
async def generations(request: Request) -> JSONResponse:
    body = await request.json()
    raw_size = body.get("size")
    if raw_size:
        # Mirror SGLang: a size that is not divisible by 32 is a hard error.
        left, _, right = str(raw_size).lower().partition("x")
        try:
            invalid = int(left) % 32 or int(right) % 32
        except ValueError:
            invalid = True
        if invalid:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": f"size {raw_size} must be divisible by 32"}},
            )
    size = parse_size(raw_size, body.get("width"), body.get("height"))
    return JSONResponse(
        image_response(
            body.get("prompt", ""),
            size,
            body.get("background"),
            body.get("n", 1),
            body.get("seed"),
        )
    )


@app.post("/v1/images/edits")
async def edits(
    prompt: str = Form(...),
    size: str | None = Form(None),
    width: int | None = Form(None),
    height: int | None = Form(None),
    background: str | None = Form(None),
    n: int | None = Form(None),
    seed: int | None = Form(None),
    image: list[UploadFile] = File(default=[]),
) -> JSONResponse:
    data = image_response(prompt, parse_size(size, width, height), background, n or 1, seed)
    data["mock"]["references"] = len(image or [])
    return JSONResponse(data)


@app.post("/v1/chat/completions")
async def chat(request: Request) -> JSONResponse:
    body = await request.json()
    messages = body.get("messages") or []
    model = body.get("model", "ds4-flash")
    user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    text = user_text(user)
    original = text.split("Original")[-1].split(":", 1)[-1].strip() or text
    enhanced = (
        f"{original}, rendered with dramatic side lighting, shallow depth of field, "
        "fine surface texture, muted teal and amber palette, 35mm film grain, "
        "wide establishing composition, atmospheric haze in the background"
    )
    if is_pe_i2i_model(model):
        # The edit checkpoint reads the reference images next to the instruction and
        # answers with `ratio_follow` instead of `wh_ratio` when the render should
        # keep a source image's shape.
        references = reference_count(user)
        record = {
            "rewritten_prompt": (
                f"{original}, integrated with the existing scene: matching perspective, "
                "lighting and reflections, keeping the current composition and palette"
            ),
            "wh_ratio": "" if references else PE_RATIOS[len(original) % len(PE_RATIOS)],
            "ratio_follow": "<image1>" if references else "",
        }
        return JSONResponse(
            {
                "id": f"chatcmpl-mock-pe-i2i-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "\n\n" + json.dumps(record, ensure_ascii=False),
                            "reasoning": (
                                f"The user supplied {references} reference image(s) and an edit "
                                "instruction. I should keep the scene and follow <image1>."
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2400, "completion_tokens": 1400, "total_tokens": 3800},
            }
        )
    if is_pe_model(model):
        # A PE checkpoint answers with a thinking block plus one JSON record at the
        # very end of the answer, exactly like Qwen-Image-2.1-PE-T2I.
        ratio = PE_RATIOS[len(original) % len(PE_RATIOS)]
        record = {"rewritten_prompt": enhanced, "wh_ratio": ratio}
        return JSONResponse(
            {
                "id": f"chatcmpl-mock-pe-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "\n\n" + json.dumps(record, ensure_ascii=False),
                            "reasoning": (
                                "The user supplied a short image brief. I should keep the "
                                "subject, add lighting, composition and palette detail, "
                                f"and answer on a {ratio} canvas."
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2400, "completion_tokens": 1400, "total_tokens": 3800},
            }
        )
    return JSONResponse(
        {
            "id": f"chatcmpl-mock-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": enhanced},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
        }
    )


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7899)
    args = parser.parse_args()
    print(f"mock upstream -> http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
