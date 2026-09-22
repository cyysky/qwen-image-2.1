"""Prompt enhancement through an OpenAI-compatible chat model.

Two engine contracts are supported, chosen per engine in `.env`:

* ``chat`` - an ordinary chat model (ds4-flash on router.pixel-space) that
  replies with the rewritten prompt as plain text.
* ``pe`` / ``pe-i2i`` - a Qwen-Image-2.1-PE checkpoint served by vLLM, which
  replies with a JSON record carrying ``rewritten_prompt``, ``wh_ratio`` and
  ``parse_ok``. The edit checkpoint (``pe-i2i``) also takes the reference
  images in the user message and may answer with ``ratio_follow`` instead of
  ``wh_ratio``, asking the render to inherit one reference image's shape.

The enhancer lives purely at the UI level: the browser asks for a rewritten
prompt, shows it to the user, and only submits to the image endpoint once the
user is happy with it.
"""

from __future__ import annotations

import json
import re
import time

import httpx

from .config import EnhancerEngine, Settings, env_var_suffix
from .errors import BadRequest, UpstreamError

# Fences or wrapping quotes some models like to add around their answer.
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?(?P<body>.*?)\n?\s*```\s*$", re.DOTALL)
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019"))
_PREAMBLE_RE = re.compile(
    r"^(here('| i)s|here is|sure[,!]?|certainly[,!]?|enhanced prompt)\b[^\n]*:\s*",
    re.IGNORECASE,
)

#: PE answers are one JSON object at the very end of the answer section.
#: `positive_prompt` is what upstream's client normalizes the record to.
_PE_REWRITE_KEYS = ("rewritten_prompt", "rewrited_prompt", "positive_prompt")


def _clean(text: str) -> str:
    """Strip markdown fences, wrapping quotes and 'here is your prompt' preambles."""
    text = (text or "").strip()
    if not text:
        return ""

    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group("body").strip()

    # Drop a leading label such as "Enhanced prompt:".
    first_line, _, rest = text.partition("\n")
    if rest and first_line.strip().rstrip(":").lower() in {
        "enhanced prompt",
        "prompt",
        "image prompt",
        "rewritten prompt",
    }:
        text = rest.strip()

    text = _PREAMBLE_RE.sub("", text, count=1).strip()

    for opener, closer in _QUOTE_PAIRS:
        if len(text) > 1 and text.startswith(opener) and text.endswith(closer):
            text = text[1:-1].strip()

    # Collapse hard wraps inside a paragraph, keep blank-line paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_thinking(text: str) -> tuple[str, str]:
    """Split a PE answer into `(thinking, answer)`.

    The chat template pre-fills `` thinking``, so a server started without
    ``--reasoning-parser qwen3`` leaves the thinking inline and the answer
    starts after ``</think>``. With the parser on, vLLM strips it for us and
    this is a no-op.
    """
    text = text or ""
    if "\u003c/think\u003e" in text:
        think, _, answer = text.partition("\u003c/think\u003e")
        if "\u003cthink\u003e" in think:
            think = think.partition("\u003cthink\u003e")[2]
        return think.strip(), answer.strip()
    if "\u003cthink\u003e" in text:
        # Unterminated thinking block: the generation hit the token budget.
        return text.partition("\u003cthink\u003e")[2].strip(), ""
    return "", text.strip()


def _balanced_spans(answer: str) -> list[str]:
    """Every balanced top-level `{...}` span, in order.

    A greedy regex is not enough: a brace in prose after the object stretches
    the match past its real end and the parse fails silently. Braces inside JSON
    string literals are skipped, so a rewrite containing `{` is still safe.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for index, char in enumerate(answer):
        if in_str:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_str = False
            continue
        if char == '"':
            in_str = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(answer[start:index + 1])
    return spans


def parse_pe_answer(answer: str) -> dict:
    """Parse a PE answer section into `positive_prompt` / `wh_ratio` / `parse_ok`.

    The wire key is `rewritten_prompt` (upstream also accepts the
    `rewrited_prompt` typo); `positive_prompt` is the internal name.
    Mirrors upstream `pe_core.parse_answer`: on failure `positive_prompt` falls
    back to the raw answer so the model's output is never lost, and `parse_ok` is
    the only way to tell that fallback from a clean parse.

    `wh_ratio` and `ratio_follow` are mutually exclusive: the t2i checkpoint
    picks a new composition with `wh_ratio`, while the edit checkpoint keeps a
    source image's shape with `ratio_follow` (`<image1>`..`<imageN>`).
    """
    answer = (answer or "").strip()
    for candidate in reversed(_balanced_spans(answer)):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        rewritten = None
        for key in _PE_REWRITE_KEYS:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                rewritten = value.strip()
                break
        if not rewritten:
            continue
        return {
            "positive_prompt": rewritten,
            "negative_prompt": str(obj.get("negative_prompt") or "").strip(),
            "wh_ratio": str(obj.get("wh_ratio") or "").strip(),
            "ratio_follow": str(obj.get("ratio_follow") or "").strip(),
            "parse_ok": True,
        }
    return {
        "positive_prompt": answer,
        "negative_prompt": "",
        "wh_ratio": "",
        "ratio_follow": "",
        "parse_ok": False,
    }


def _snap_dimension(value: int) -> int:
    return max(32, int(round(value / 32.0)) * 32)


def snap_size(size: str) -> str:
    """Snap "WxH" to the multiple of 32 SGLang accepts, or "" if unparsable."""
    match = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", size or "", re.IGNORECASE)
    if not match:
        return ""
    return f"{_snap_dimension(int(match.group(1)))}x{_snap_dimension(int(match.group(2)))}"


def ratio_follow_size(ratio_follow: str, image_sizes: list[str] | None) -> str:
    """Canvas implied by a PE answer's `ratio_follow` (`<image1>`..`<imageN>`).

    The edit checkpoint sets this instead of `wh_ratio` when the render should
    inherit a source image's shape; the caller supplies those source sizes, so
    the size can be offered without decoding the references here.
    """
    match = re.search(r"image\s*(\d+)", ratio_follow or "", re.IGNORECASE)
    if not match or not image_sizes:
        return ""
    index = int(match.group(1)) - 1
    if index < 0 or index >= len(image_sizes):
        return ""
    return snap_size(image_sizes[index])


def _data_url(image: str) -> str:
    """Accept a `data:` URL or bare base64 and return a `data:` URL."""
    text = (image or "").strip()
    if not text:
        return ""
    if text.startswith("data:"):
        return text
    return f"data:image/png;base64,{text}"


def build_instruction(
    settings: Settings,
    *,
    mode: str | None,
    instruction: str | None,
    is_edit: bool,
) -> str | None:
    """Combine a preset mode and a free-form instruction into one directive."""
    parts: list[str] = []
    if mode:
        preset = settings.enhancer_presets.get(mode)
        if preset is None:
            match = next(
                (key for key in settings.enhancer_presets if key.lower() == mode.lower()),
                None,
            )
            preset = settings.enhancer_presets[match] if match else None
        if preset is None:
            raise BadRequest(f"Unknown enhancer preset: {mode}")
        parts.append(preset)
    if instruction and instruction.strip():
        parts.append(instruction.strip())
    if not parts:
        return None
    return " ".join(parts)


def build_messages(
    system: str,
    *,
    prompt: str,
    instruction: str | None,
    is_edit: bool = False,
    negative_prompt: str | None = None,
    size: str | None = None,
) -> list[dict]:
    """Assemble the chat messages sent to the enhancer model."""
    context_lines: list[str] = []
    if instruction:
        context_lines.append(f"Enhancement directive: {instruction}")
    if size:
        context_lines.append(f"Target output resolution: {size}")
    if negative_prompt and negative_prompt.strip():
        context_lines.append(
            "Things to avoid in the result (do not add a negative prompt section, "
            f"just steer away from them): {negative_prompt.strip()}"
        )
    kind = "edit instruction" if is_edit else "image idea"
    context_lines.append(f"Original {kind}: {prompt.strip()}")
    context_lines.append(
        "Reply with the rewritten prompt only."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(context_lines)},
    ]


def _pe_payload(engine: EnhancerEngine, model: str | None, messages: list[dict]) -> dict:
    """Body for a PE checkpoint: upstream's production sampling profile."""
    payload: dict = {
        "model": model or engine.model,
        "messages": messages,
        "temperature": engine.temperature,
        "top_p": engine.top_p,
        "presence_penalty": engine.presence_penalty,
        "max_tokens": engine.max_tokens,
        "stream": False,
    }
    if engine.enable_thinking is not None:
        # vLLM extension. Upstream's t2i client sends it so the chat template
        # does not suppress the think block the checkpoints were trained with;
        # the edit host runs with --reasoning-parser qwen3 and needs no hint.
        payload["chat_template_kwargs"] = {"enable_thinking": bool(engine.enable_thinking)}
    # top_k / min_p are vLLM extensions too, and go at the top level.
    if engine.top_k:
        payload["top_k"] = engine.top_k
    if engine.min_p:
        payload["min_p"] = engine.min_p
    if engine.seed is not None:
        payload["seed"] = engine.seed
    return payload


def _chat_payload(engine: EnhancerEngine, model: str | None, messages: list[dict]) -> dict:
    """Body for an ordinary chat model; optional fields only when non-default."""
    payload: dict = {
        "model": model or engine.model,
        "messages": messages,
        "temperature": engine.temperature,
        "max_tokens": engine.max_tokens,
        "stream": False,
    }
    if engine.top_p and engine.top_p != 1.0:
        payload["top_p"] = engine.top_p
    if engine.presence_penalty:
        payload["presence_penalty"] = engine.presence_penalty
    if engine.seed is not None:
        payload["seed"] = engine.seed
    return payload


async def enhance_prompt(
    settings: Settings,
    *,
    prompt: str,
    mode: str | None = None,
    instruction: str | None = None,
    is_edit: bool = False,
    negative_prompt: str | None = None,
    size: str | None = None,
    model: str | None = None,
    engine: str | None = None,
    images: list[str] | None = None,
    image_sizes: list[str] | None = None,
) -> dict:
    """Rewrite `prompt` with the selected enhancer engine.

    `images` are the reference images (a `data:` URL or bare base64 each) and
    `image_sizes` their "WxH" sizes, both in the order the references were
    sent. An engine that takes image inputs needs at least one of them; a PE
    answer that asks to follow a reference image uses the matching size to
    suggest a canvas, since the checkpoint only names `<imageN>`.
    """
    if not settings.enhancer_enabled:
        raise BadRequest("The prompt enhancer is disabled (ENHANCER_ENABLED=false)")
    if not (prompt or "").strip():
        raise BadRequest("Cannot enhance an empty prompt")
    try:
        target = settings.enhancer_engine(engine)
    except LookupError as exc:
        raise BadRequest(str(exc)) from exc
    if not target.base_url:
        raise BadRequest(
            f"No base URL configured for enhancer '{target.key}' - set "
            f"ENHANCER_ENGINE_{env_var_suffix(target.key)}_BASE_URL"
        )

    directive = build_instruction(
        settings, mode=mode, instruction=instruction, is_edit=is_edit
    )
    try:
        system_prompt, prompt_source = target.system_prompt_for(is_edit)
    except LookupError as exc:
        raise BadRequest(str(exc)) from exc
    if not system_prompt:
        name = "edit system prompt" if is_edit else "system prompt"
        variable = "EDIT_SYSTEM_PROMPT" if is_edit else "SYSTEM_PROMPT"
        raise BadRequest(
            f"Enhancer '{target.key}' has no {name} - set "
            f"ENHANCER_ENGINE_{env_var_suffix(target.key)}_{variable}"
        )

    if target.is_pe:
        # The PE checkpoints are trained on the bare brief: the system prompt
        # already carries the whole task, so no resolution or negative hints.
        user_text = prompt.strip()
        if directive:
            user_text = f"{user_text}\n\nEnhancement directive: {directive}"
        if target.image_inputs:
            # The edit checkpoint is multimodal: the user message is the
            # reference images followed by the instruction, exactly the shape
            # upstream's `--task edit` client sends.
            parts = [
                {"type": "image_url", "image_url": {"url": url}}
                for url in (_data_url(image) for image in (images or []))
                if url
            ]
            if not parts:
                raise BadRequest(
                    f"Enhancer '{target.key}' rewrites edits, so it needs at least "
                    "one reference image"
                )
            parts.append({"type": "text", "text": user_text})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": parts},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]
        payload = _pe_payload(target, model, messages)
    else:
        messages = build_messages(
            system_prompt,
            prompt=prompt,
            instruction=directive,
            is_edit=is_edit,
            negative_prompt=negative_prompt,
            size=size,
        )
        payload = _chat_payload(target, model, messages)

    url = target.url()
    headers = {"Content-Type": "application/json"}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"

    started = time.perf_counter()
    timeout = httpx.Timeout(target.timeout_s, connect=min(30.0, target.timeout_s))
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise UpstreamError(
            504, f"Enhancer timed out after {target.timeout_s:.0f}s", url=url
        ) from exc
    except httpx.RequestError as exc:
        raise UpstreamError(502, f"Cannot reach enhancer at {url}: {exc}", url=url) from exc

    elapsed = time.perf_counter() - started

    if response.status_code >= 400:
        raise UpstreamError(
            response.status_code,
            f"Enhancer returned HTTP {response.status_code}",
            url=url,
            body=response.text,
        )

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise UpstreamError(
            502, "Enhancer returned a non-JSON response", url=url, body=response.text
        ) from exc

    if isinstance(data, dict) and data.get("error"):
        error = data["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise UpstreamError(
            502, f"Enhancer error: {message}", url=url, body=json.dumps(data)[:2000]
        )

    choices = data.get("choices") or []
    content = ""
    thinking = ""
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        # vLLM 0.27.1 names the think block `reasoning`; `--reasoning-parser
        # qwen3` names it `reasoning_content`. Either way it is not the answer.
        thinking = message.get("reasoning") or message.get("reasoning_content") or ""
    if not thinking and content:
        # No reasoning parser on the server: the think block is inline.
        thinking, content = _split_thinking(content)

    if target.is_pe:
        parsed = parse_pe_answer(content)
        enhanced = parsed["positive_prompt"]
        if not enhanced:
            raise UpstreamError(
                502, "Enhancer returned an empty prompt", url=url,
                body=response.text[:2000],
            )
        suggested = settings.enhancer_ratio_sizes.get(parsed["wh_ratio"], "")
        if not suggested and parsed["ratio_follow"]:
            # The edit checkpoint keeps a source image's shape instead of
            # naming a ratio, so the suggestion comes from that reference.
            suggested = ratio_follow_size(parsed["ratio_follow"], image_sizes)
        return {
            "prompt": enhanced,
            "original_prompt": prompt.strip(),
            "model": payload["model"],
            "base_url": target.base_url,
            "engine": target.key,
            "kind": target.kind,
            "instruction": directive,
            "negative_prompt": parsed["negative_prompt"],
            "wh_ratio": parsed["wh_ratio"],
            "ratio_follow": parsed["ratio_follow"],
            "suggested_size": suggested,
            "parse_ok": parsed["parse_ok"],
            "thinking": thinking,
            "answer": content,
            "system_prompt_source": prompt_source,
            "image_inputs": target.image_inputs,
            "image_count": len(images or []) if target.image_inputs else 0,
            "elapsed_s": round(elapsed, 2),
            "usage": data.get("usage"),
        }

    enhanced = _clean(content)
    if not enhanced:
        raise UpstreamError(
            502, "Enhancer returned an empty prompt", url=url, body=response.text[:2000]
        )

    return {
        "prompt": enhanced,
        "original_prompt": prompt.strip(),
        "model": payload["model"],
        "base_url": target.base_url,
        "engine": target.key,
        "kind": target.kind,
        "instruction": directive,
        "negative_prompt": "",
        "wh_ratio": "",
        "ratio_follow": "",
        "suggested_size": "",
        "parse_ok": True,
        "thinking": thinking,
        "answer": content,
        "system_prompt_source": prompt_source,
        "image_inputs": False,
        "image_count": 0,
        "elapsed_s": round(elapsed, 2),
        "usage": data.get("usage"),
    }
