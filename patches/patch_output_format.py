#!/usr/bin/env python3
"""Make SGLang's OpenAI image API safe for Qwen-Image-2.1's RGBA VAE.

Qwen-Image-2.1's VAE emits 4-channel RGBA. SGLang's shared default picks
JPEG when a request omits `output_format`, and PIL then aborts the request with

    OSError: cannot write mode RGBA as JPEG

so every generation without an explicit output_format failed. Two edits:

1. `choose_output_image_ext` (used by both /v1/images/generations and
   /v1/images/edits) now defaults to PNG, which is also OpenAI's own default
   and the only lossless option that can carry the alpha channel.
2. `_save_image_frame` composites alpha over white before handing the frame to
   the JPEG/WebP encoder, so an explicit `output_format=jpeg` also succeeds
   instead of returning HTTP 500.

Every edit asserts on the exact upstream text, so the build fails loudly if
SGLang changes these files instead of silently shipping a broken service.
"""

from pathlib import Path

OPENAI_UTILS = Path(
    "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py"
)
ENTRYPOINT_UTILS = Path(
    "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/utils.py"
)

# 1. choose_output_image_ext: fall back to png, not jpg.
old_ext = """    if (background or "auto").lower() == "transparent":
        return "png"
    return "jpg"
"""
new_ext = """    # PNG is OpenAI's default and the only lossless format that can carry the
    # RGBA output of the Qwen-Image-2.1 VAE; the upstream "jpg" fallback made
    # every request that omitted output_format fail with
    # "cannot write mode RGBA as JPEG".
    return "png"
"""

# 2. _save_image_frame: composite RGBA over white for non-PNG formats.
old_save = """    else:
        imageio.imwrite(path, frame, quality=quality)
"""
new_save = """    else:
        # The JPEG/WebP encoders cannot take 4 channels; composite alpha over
        # white so an explicit output_format=jpeg/webp still succeeds.
        if frame.ndim == 3 and frame.shape[-1] == 4:
            alpha = frame[..., 3:4].astype("float32") / 255.0
            rgb = frame[..., :3].astype("float32")
            frame = (rgb * alpha + 255.0 * (1.0 - alpha)).round().astype("uint8")
        imageio.imwrite(path, frame, quality=quality)
"""


def patch(path: Path, old: str, new: str, label: str) -> None:
    src = path.read_text()
    assert new not in src, f"{label}: already patched"
    assert old in src, f"{label}: upstream text changed, refusing to patch"
    path.write_text(src.replace(old, new, 1))
    print(f"patched {label}")


patch(OPENAI_UTILS, old_ext, new_ext, "choose_output_image_ext")
patch(ENTRYPOINT_UTILS, old_save, new_save, "_save_image_frame")
print("output-format patches applied")
