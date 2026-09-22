#!/usr/bin/env bash
# Quick health + OpenAI-API smoke test for the Qwen-Image-2.1 service.
# Usage: scripts/smoke.sh [base_url] [out_dir]
set -euo pipefail

base_url="${1:-http://127.0.0.1:7853}"
out_dir="${2:-/home/aiserver/qwen-image-2.1/outputs/smoke}"
mkdir -p "$out_dir"

echo "== /health =="
curl -sf "$base_url/health"; echo

echo "== /v1/models =="
curl -sf "$base_url/v1/models"; echo

echo "== /v1/images/generations (no response_format -> expect b64_json) =="
curl -sf "$base_url/v1/images/generations" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen-Image-2.1",
    "prompt": "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement",
    "size": "1024x1024",
    "num_inference_steps": 40,
    "seed": 42
  }' > "$out_dir/response.json"

python3 - "$out_dir" <<'PY'
import base64, json, sys
from pathlib import Path

out = Path(sys.argv[1])
payload = json.loads((out / "response.json").read_text())
print("id:", payload.get("id"))
print("inference_time_s:", payload.get("inference_time_s"))
print("peak_memory_mb:", payload.get("peak_memory_mb"))
print("usage:", payload.get("usage"))
for i, item in enumerate(payload.get("data", [])):
    b64 = item.get("b64_json")
    print(f"data[{i}] url={item.get('url')!r} b64_len={len(b64) if b64 else 0}")
    if b64:
        path = out / f"smoke_{i}.png"
        path.write_bytes(base64.b64decode(b64))
        print("saved:", path)
PY
