#!/usr/bin/env python3
"""End-to-end smoke test: mock upstream + the playground app.

Boots `tools/mock_upstream.py` and `app.py` on spare ports, points the app at the
mock (image endpoint *and* enhancer) and exercises every route:

    python tools/smoke_test.py

Exits non-zero on the first failure, so it works as a CI check.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_PORT = 7899
APP_PORT = 7898


def free_port(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def wait_for(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                response.read()
                return
        except Exception:  # noqa: BLE001 - keep polling until the deadline
            time.sleep(0.4)
    raise RuntimeError(f"{url} never came up")


def request(method: str, url: str, payload=None, headers=None):
    data = None
    hdrs = dict(headers or {})
    if payload is not None:
        if isinstance(payload, (bytes, bytearray)):
            data = bytes(payload)
        else:
            data = json.dumps(payload).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"raw": body}


def multipart(fields: dict, files: list[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    boundary = "----smoke" + str(int(time.time() * 1000))
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    for name, filename, blob, mime in files:
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'
            ).encode()
            + blob
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {label}{(' -> ' + detail) if detail else ''}")
    if not condition:
        raise SystemExit(1)


def balanced_div(html: str, marker: str) -> str:
    """Return the full `<div>...</div>` block whose opening tag contains `marker`.

    Used to assert DOM nesting: a panel placed inside a hidden tab panel is
    un-hideable, which is exactly how the Edit tab's enhanced-prompt box broke.
    """
    at = html.index(marker)
    start = html.rindex("<div", 0, at)
    depth = 0
    pos = start
    while pos < len(html):
        open_at = html.find("<div", pos)
        close_at = html.find("</div>", pos)
        if close_at == -1:
            break
        if open_at != -1 and open_at < close_at:
            depth += 1
            pos = open_at + 4
        else:
            depth -= 1
            pos = close_at + 6
            if depth == 0:
                return html[start:pos]
    return html[start:]


def main() -> int:
    if not free_port(UPSTREAM_PORT) or not free_port(APP_PORT):
        print("ports 7898/7899 are busy; stop whatever is using them first")
        return 1

    env = dict(os.environ)
    env.update(
        {
            "IMAGE_ENDPOINTS": f"mock|http://127.0.0.1:{UPSTREAM_PORT}/v1||Mock upstream",
            "DEFAULT_ENDPOINT": "mock",
            "ENHANCER_BASE_URL": f"http://127.0.0.1:{UPSTREAM_PORT}/v1",
            "ENHANCER_MODEL": "ds4-flash",
            "ENHANCER_API_KEY": "sk-smoke",
            "ENHANCER_ENGINES": "ds4-flash,pe-t2i,pe-i2i",
            "ENHANCER_ENGINE": "ds4-flash",
            "ENHANCER_ENGINE_PE_T2I_LABEL": "Qwen-Image-2.1-PE-T2I (mock)",
            "ENHANCER_ENGINE_PE_T2I_KIND": "pe",
            "ENHANCER_ENGINE_PE_T2I_BASE_URL": f"http://127.0.0.1:{UPSTREAM_PORT}/v1",
            "ENHANCER_ENGINE_PE_T2I_MODEL": "Qwen/Qwen-Image-2.1-PE-T2I",
            "ENHANCER_ENGINE_PE_I2I_LABEL": "Qwen-Image-2.1-PE-I2I (mock)",
            "ENHANCER_ENGINE_PE_I2I_KIND": "pe-i2i",
            "ENHANCER_ENGINE_PE_I2I_BASE_URL": f"http://127.0.0.1:{UPSTREAM_PORT}/v1",
            "ENHANCER_ENGINE_PE_I2I_MODEL": "Qwen/Qwen-Image-2.1-PE-I2I",
            "APP_PORT": str(APP_PORT),
            "APP_HOST": "127.0.0.1",
            "HEALTH_TIMEOUT_S": "10",
            "IMAGE_TIMEOUT_S": "120",
            "ENHANCER_APPLY_BY_DEFAULT": "false",
        }
    )

    upstream = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "mock_upstream.py"), "--port", str(UPSTREAM_PORT)],
        cwd=str(ROOT),
    )
    app = subprocess.Popen([sys.executable, str(ROOT / "app.py")], cwd=str(ROOT), env=env)

    base = f"http://127.0.0.1:{APP_PORT}"
    try:
        wait_for(f"{base}/api/config")
        print(f"app up on {base}, upstream on {UPSTREAM_PORT}\n")

        status, config = request("GET", f"{base}/api/config")
        check("GET /api/config", status == 200 and config["endpoints"][0]["name"] == "mock")
        check("  enhancer url", config["enhancer"]["url"].endswith("/chat/completions"), config["enhancer"]["url"])
        check("  sizes from .env", "1024x1024" in config["sizes"])
        engine_keys = [engine["key"] for engine in config["enhancer"]["engines"]]
        check(
            "  enhancer engines from .env",
            engine_keys == ["ds4-flash", "pe-t2i", "pe-i2i"],
            str(engine_keys),
        )
        pe_engine = config["enhancer"]["engines"][1]
        check("  pe engine is text-to-image only", pe_engine["supports_edit"] is False)
        check("  pe engine supports text-to-image", pe_engine["supports_t2i"] is True)
        check("  pe engine takes no image inputs", pe_engine["image_inputs"] is False)
        i2i_engine = config["enhancer"]["engines"][2]
        check("  pe-i2i engine is edit only", i2i_engine["supports_t2i"] is False)
        check("  pe-i2i engine supports edits", i2i_engine["supports_edit"] is True)
        check("  pe-i2i engine takes image inputs", i2i_engine["image_inputs"] is True)
        check("  pe engine exposes a ratio map", "16:9" in config["enhancer"]["ratio_sizes"])
        check("  pe engine is not the default", config["enhancer"]["default"] == "ds4-flash")

        status, health = request("GET", f"{base}/api/health")
        check("GET /api/health", status == 200 and health["results"][0]["ok"] is True)

        status, models = request("GET", f"{base}/api/models?endpoint=mock")
        check("GET /api/models", status == 200 and len(models["models"]) >= 1)

        status, enhanced = request(
            "POST",
            f"{base}/api/enhance",
            {"prompt": "a capybara reading a book", "mode": "cinematic", "instruction": "keep it cosy"},
        )
        check("POST /api/enhance", status == 200 and len(enhanced["prompt"]) > 40)
        check("  preset applied", enhanced["instruction"].startswith("Give it a cinematic"))
        check("  chat engine echoed", enhanced["engine"] == "ds4-flash" and enhanced["kind"] == "chat")

        status, pe_enhanced = request(
            "POST",
            f"{base}/api/enhance",
            {"prompt": "a capybara reading a book", "engine": "pe-t2i"},
        )
        check("POST /api/enhance (PE engine)", status == 200, pe_enhanced.get("error", ""))
        check("  pe kind", pe_enhanced["kind"] == "pe")
        ratio = pe_enhanced["wh_ratio"]
        check("  pe wh_ratio", ratio in config["enhancer"]["ratio_sizes"], ratio)
        check(
            "  pe suggested size",
            pe_enhanced["suggested_size"] == config["enhancer"]["ratio_sizes"][ratio],
            pe_enhanced["suggested_size"],
        )
        check("  pe parse_ok", pe_enhanced["parse_ok"] is True)
        check(
            "  pe thinking kept out of the prompt",
            "The user supplied" not in pe_enhanced["prompt"] and len(pe_enhanced["thinking"]) > 0,
        )
        check(
            "  pe system prompt came from the bundled file",
            pe_enhanced["system_prompt_source"].endswith("pe_t2i_system_prompt.txt"),
            pe_enhanced["system_prompt_source"],
        )

        status, pe_edit = request(
            "POST",
            f"{base}/api/enhance",
            {"prompt": "add a lantern", "engine": "pe-t2i", "is_edit": True},
        )
        check("PE engine has no edit prompt -> 400", status == 400, str(pe_edit.get("error", ""))[:60])

        status, i2i_no_image = request(
            "POST",
            f"{base}/api/enhance",
            {"prompt": "add a lantern", "engine": "pe-i2i", "is_edit": True},
        )
        check(
            "PE-I2I engine without a reference -> 400",
            status == 400,
            str(i2i_no_image.get("error", ""))[:60],
        )

        status, unknown = request(
            "POST", f"{base}/api/enhance", {"prompt": "x", "engine": "nope"}
        )
        check("unknown engine -> 400", status == 400, unknown.get("error", ""))

        status, generated = request(
            "POST",
            f"{base}/api/generate",
            {
                "endpoint": "mock",
                "model": "Qwen-Image-2.1",
                "prompt": "a capybara reading a book by candlelight",
                "size": "1024x1024",
                "num_inference_steps": 40,
                "seed": 42,
            },
        )
        check("POST /api/generate", status == 200 and generated["data"][0]["b64_json"])
        check("  telemetry", generated["inference_time_s"] == 1.25)
        check("  request echo", generated["request"]["prompt"].startswith("a capybara"))
        image = base64.b64decode(generated["data"][0]["b64_json"])
        check("  png magic", image[:8] == b"\x89PNG\r\n\x1a\n", f"{len(image)} bytes")

        reference = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        status, i2i_enhanced = request(
            "POST",
            f"{base}/api/enhance",
            {
                "prompt": "add a lantern",
                "engine": "pe-i2i",
                "is_edit": True,
                "images": [reference],
                "image_sizes": ["1696x2528"],
            },
        )
        check("POST /api/enhance (PE-I2I engine)", status == 200, i2i_enhanced.get("error", ""))
        check("  pe-i2i kind", i2i_enhanced["kind"] == "pe-i2i")
        check(
            "  pe-i2i read the reference",
            i2i_enhanced["image_inputs"] is True and i2i_enhanced["image_count"] == 1,
        )
        check("  pe-i2i ratio_follow", i2i_enhanced["ratio_follow"] == "<image1>")
        check(
            "  pe-i2i suggested size follows the reference",
            i2i_enhanced["suggested_size"] == "1696x2528",
            i2i_enhanced["suggested_size"],
        )
        check("  pe-i2i parse_ok", i2i_enhanced["parse_ok"] is True)
        check(
            "  pe-i2i system prompt came from the bundled file",
            i2i_enhanced["system_prompt_source"].endswith("pe_i2i_system_prompt.txt"),
            i2i_enhanced["system_prompt_source"],
        )

        status, i2i_generate = request(
            "POST",
            f"{base}/api/generate",
            {"prompt": "a capybara", "enhance": True, "enhance_engine": "pe-i2i"},
        )
        check(
            "PE-I2I engine has no t2i prompt -> 400",
            status == 400,
            str(i2i_generate.get("error", ""))[:60],
        )

        status, enhanced_gen = request(
            "POST",
            f"{base}/api/generate",
            {
                "prompt": "a capybara reading a book",
                "size": "1024x1024",
                "enhance": True,
                "enhance_mode": "photoreal",
            },
        )
        check("POST /api/generate + enhance", status == 200)
        check(
            "  prompt was rewritten",
            enhanced_gen["request"]["original_prompt"] == "a capybara reading a book",
        )

        status, pe_gen = request(
            "POST",
            f"{base}/api/generate",
            {
                "prompt": "a capybara reading a book",
                "size": "1024x1024",
                "enhance": True,
                "enhance_engine": "pe-t2i",
            },
        )
        check("POST /api/generate + PE enhance", status == 200, str(pe_gen.get("error", ""))[:80])
        pe_meta = pe_gen["request"]["enhanced"]
        check("  PE engine recorded", pe_meta["engine"] == "pe-t2i", str(pe_meta["engine"]))
        check("  PE ratio recorded", pe_meta["wh_ratio"] in config["enhancer"]["ratio_sizes"])

        status, rejected = request(
            "POST", f"{base}/api/generate", {"prompt": "x", "size": "720x720"}
        )
        check("size not divisible by 32 -> 400", status == 400, rejected.get("error", ""))

        status, transparent = request(
            "POST",
            f"{base}/api/generate",
            {
                "prompt": "a cute cartoon dragon sticker",
                "size": "1024x1024",
                "background": "transparent",
                "output_format": "png",
            },
        )
        check("transparent request", status == 200 and transparent["data"][0]["b64_json"])

        body, content_type = multipart(
            {
                "endpoint": "mock",
                "prompt": "Add a glowing red paper lantern above the sign",
                "size": "1024x1024",
                "seed": "42",
            },
            [("image", "ref.png", image, "image/png")],
        )
        status, edited = request(
            "POST", f"{base}/api/edit", body, {"Content-Type": content_type}
        )
        check("POST /api/edit", status == 200 and edited["data"][0]["b64_json"])
        check("  reference counted", edited["mock"]["references"] == 1)
        check("  params echoed", edited["request"]["params"]["seed"] == "42")

        body, content_type = multipart(
            {
                "endpoint": "mock",
                "prompt": "Add a glowing red paper lantern above the sign",
                "enhance": "true",
                "enhance_engine": "pe-i2i",
                "reference_sizes": "1696x2528",
            },
            [("image", "ref.png", image, "image/png")],
        )
        status, i2i_edited = request(
            "POST", f"{base}/api/edit", body, {"Content-Type": content_type}
        )
        check("POST /api/edit + PE-I2I enhance", status == 200, str(i2i_edited.get("error", ""))[:80])
        check("  reference counted", i2i_edited["mock"]["references"] == 1)
        check(
            "  suggested size applied to the render",
            i2i_edited["request"]["params"]["size"] == "1696x2528",
            str(i2i_edited["request"]["params"].get("size")),
        )

        body, content_type = multipart(
            {"endpoint": "mock", "prompt": "no reference attached"}, []
        )
        status, _ = request("POST", f"{base}/api/edit", body, {"Content-Type": content_type})
        check("edit without a reference -> 400", status == 400)

        status, reloaded = request("POST", f"{base}/api/config/reload", {})
        check("POST /api/config/reload", status == 200 and reloaded["ok"] is True)

        with urllib.request.urlopen(f"{base}/", timeout=10) as response:
            html = response.read().decode("utf-8")
        check("GET / serves the UI", "Qwen-Image-2.1 Playground" in html)
        with urllib.request.urlopen(f"{base}/static/app.js", timeout=10) as response:
            js = response.read().decode("utf-8")
        check("GET /static/app.js", "curlForCurrentForm" in js)

        generate_panel = balanced_div(html, 'id="panel-generate"')
        check(
            "enhanced-prompt panel sits outside the generate tab",
            'id="enhance-panel"' not in generate_panel,
        )
        edit_panel = balanced_div(html, 'id="panel-edit"')
        check("edit tab has its own enhance button", 'id="edit-enhance-btn"' in edit_panel)
        check("engine selector in the UI", 'id="enhance-engine-select"' in html)
        check("ratio row in the UI", 'id="enhance-ratio"' in html)
        check("app.js restores results from IndexedDB", "indexedDB" in js)
        check("app.js sends the selected engine", "enhance_engine" in js)
        check("app.js gates engines per tab", "supportsT2i" in js and "supportsEdit" in js)
        check("app.js sends the reference sizes", "reference_sizes" in js)
        check("app.js reads the references for the enhancer", "readAsDataUrl" in js)
        check("app.js keeps the enhance ratio row", "ratio_follow" in js)

        print("\nall checks passed")
        return 0
    finally:
        for process in (app, upstream):
            process.terminate()
        for process in (app, upstream):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    sys.exit(main())
