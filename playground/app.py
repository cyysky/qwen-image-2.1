#!/usr/bin/env python3
"""Entry point for the Qwen-Image-2.1 playground.

    python app.py                  # serve the UI on APP_HOST:APP_PORT
    python app.py --check          # print config and probe every image endpoint
    python app.py --check-enhancer  # also send one tiny enhancer request
    python app.py --reload         # auto-reload while editing the backend
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import struct
import sys
import zlib

import httpx

from backend.config import get_settings


def print_config() -> None:
    settings = get_settings()
    print("image endpoints:")
    for endpoint in settings.endpoints:
        key = "key set" if endpoint.api_key else "no key"
        print(f"  - {endpoint.name:<12} {endpoint.base_url:<40} {key}  ({endpoint.display})")
    print(f"  default        : {settings.default_endpoint}")
    print(f"image models   : {', '.join(settings.image_models or (settings.image_model,))}")
    print(f"sizes          : {', '.join(settings.sizes)}")
    print(
        "defaults       : "
        f"size={settings.default_size} steps={settings.default_steps} "
        f"guidance={settings.default_guidance_scale} format={settings.default_output_format}"
    )
    print(f"image timeout  : {settings.image_timeout_s:.0f}s")
    print("enhancer:")
    print(f"  enabled      : {settings.enhancer_enabled}")
    print(f"  default      : {settings.default_enhancer or '-'}")
    for engine in settings.enhancer_engines:
        key = "key set" if engine.api_key else "no key"
        tasks = "+".join(
            name
            for name, supported in (("t2i", engine.supports_t2i), ("edit", engine.supports_edit))
            if supported
        ) or "no prompt"
        if engine.image_inputs:
            tasks += " (image inputs)"
        print(
            f"  - {engine.key:<12} {engine.kind:<5} {engine.model:<28} "
            f"{key:<7} {tasks:<22} {engine.url()}"
        )
    print(f"  presets      : {', '.join(settings.enhancer_presets) or '-'}")
    print(f"web server     : http://{settings.host}:{settings.port}")


async def probe_endpoints() -> bool:
    settings = get_settings()
    healthy = True
    async with httpx.AsyncClient(timeout=settings.health_timeout_s) as client:
        for endpoint in settings.endpoints:
            url = f"{endpoint.base_url}/models"
            headers = {}
            if endpoint.api_key:
                headers["Authorization"] = f"Bearer {endpoint.api_key}"
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                count = len((response.json() or {}).get("data") or [])
                print(f"  ok    {endpoint.name:<12} {url} ({count} model(s))")
            except httpx.HTTPStatusError as exc:
                healthy = False
                print(f"  HTTP {exc.response.status_code} {endpoint.name:<12} {url}")
            except Exception as exc:  # noqa: BLE001 - report anything that failed
                healthy = False
                print(f"  FAIL  {endpoint.name:<12} {url} -> {exc}")
    return healthy


def _probe_reference(side: int = 64) -> str:
    """A tiny solid PNG as a data URL, so an edit enhancer has a reference."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    pixel = bytes((90, 140, 200))
    raw = b"".join(b"\x00" + pixel * side for _ in range(side))
    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(blob).decode("ascii")


async def probe_enhancer() -> bool:
    settings = get_settings()
    if not settings.enhancer_enabled or not settings.enhancer_engines:
        print("  enhancer disabled or unconfigured, skipping")
        return True

    from backend.enhancer import enhance_prompt

    healthy = True
    for engine in settings.enhancer_engines:
        # An edit checkpoint only rewrites an instruction next to a reference
        # image, so hand it a tiny solid one instead of the bare text prompt.
        extra_kwargs = {}
        if engine.image_inputs:
            extra_kwargs = {"images": [_probe_reference()], "image_sizes": ["64x64"]}
        try:
            result = await enhance_prompt(
                settings,
                prompt=(
                    "add a glowing red paper lantern above the sign"
                    if engine.image_inputs
                    else "a capybara reading a book by candlelight"
                ),
                engine=engine.key,
                is_edit=engine.image_inputs,
                **extra_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - report anything that failed
            healthy = False
            print(f"  FAIL  {engine.key:<12} {engine.url()} -> {exc}")
            continue
        extra = ""
        if result.get("kind") in {"pe", "pe-i2i"}:
            extra = (
                f" ratio={result.get('wh_ratio') or result.get('ratio_follow') or '-'}"
                f" size={result.get('suggested_size') or '-'}"
                f" parse_ok={result.get('parse_ok')}"
            )
        print(f"  ok    {engine.key:<12} {engine.url()}")
        print(
            f"        model={result['model']} elapsed={result['elapsed_s']}s{extra}"
        )
        print(f"        prompt={result['prompt'][:160]}")
    return healthy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="print config and probe every image endpoint")
    parser.add_argument("--check-enhancer", action="store_true", help="also send one enhancer request")
    parser.add_argument("--reload", action="store_true", help="auto-reload on file changes (development)")
    args = parser.parse_args()

    settings = get_settings()

    if args.check or args.check_enhancer:
        print_config()
        print("\nprobing image endpoints:")
        ok = asyncio.run(probe_endpoints())
        if args.check_enhancer:
            print("\nprobing enhancer:")
            ok = asyncio.run(probe_enhancer()) and ok
        return 0 if ok else 1

    import uvicorn

    print(f"Qwen-Image-2.1 playground -> http://{settings.host}:{settings.port}")
    print(f"endpoints: {', '.join(e.name for e in settings.endpoints)}")
    print(
        "enhancer : "
        + ", ".join(f"{e.key} ({e.model})" for e in settings.enhancer_engines)
    )
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
