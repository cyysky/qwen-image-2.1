# SGLang + Qwen-Image-2.1 service image
#
# Purpose: image for the `sglang-qwen-image-21` service in qwen-image-2.1.yml.
# Modelled on the existing z-image service (/home/aiserver/sglang-dockerfile).
#
# Base: the first published SGLang image built after upstream merged native
# Qwen-Image-2.1 support (sgl-project/sglang PR #39983, merged 2026-09-20
# 01:46 UTC). The v0.5.20 release tag and the `latest` image predate it, so the
# nightly-dev image is used instead of a release image.
#
# Changes vs the base image:
#   * fail the build loudly if the base image predates Qwen-Image-2.1 support
#   * make sure the `diffusion` extras are installed (diffusers, cache-dit, ...)
#   * default the OpenAI-compatible image API to b64_json instead of url, so
#     clients (including requests routed through LiteLLM, which drops
#     response_format) get inline base64 images without extra parameters.
#
# Build: sudo docker build -t sglang-qwen-image-21:latest /home/aiserver/qwen-image-2.1
FROM lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5

WORKDIR /sgl-workspace/sglang

# Native Qwen-Image-2.1 pipeline is what this service exists for; a base image
# without it must fail the build instead of silently serving the wrong model.
RUN test -f \
        /sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines/qwen_image21.py

# Diffusion extras. Idempotent when the base image already ships them.
RUN pip install -e "python[diffusion]"

# Patch the JSON API schema default: response_format "url" -> "b64_json".
# The greps fail the build loudly if upstream changes the file, instead of
# silently producing an image without the patch.
RUN grep -q 'response_format: Optional\[str\] = "url"' \
        /sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py \
    && sed -i 's/response_format: Optional\[str\] = "url"/response_format: Optional[str] = "b64_json"/' \
        /sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py \
    && grep -q 'response_format: Optional\[str\] = "b64_json"' \
        /sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py

# Patch the OpenAI image API defaults for an RGBA model (see
# patches/patch_output_format.py for the exact edits and assertions).
COPY patches/ /tmp/qwen-image-21-patches/
RUN python3 /tmp/qwen-image-21-patches/patch_output_format.py

# Runtime stays identical: `sglang serve` is launched by the compose command,
# the model comes from /model, and outputs go to /sgl-workspace/sglang/outputs.
