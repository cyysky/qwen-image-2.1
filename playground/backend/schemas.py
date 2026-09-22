"""Request/response models for the playground API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Body of POST /api/generate (text-to-image)."""

    prompt: str = Field(min_length=1)
    endpoint: str | None = None
    model: str | None = None
    negative_prompt: str | None = None
    size: str | None = None
    width: int | None = None
    height: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None
    n: int | None = None
    output_format: str | None = None
    background: str | None = None
    # playground conveniences
    enhance: bool = False
    enhance_instruction: str | None = None
    enhance_model: str | None = None
    enhance_mode: str | None = None
    enhance_engine: str | None = None


class EnhanceRequest(BaseModel):
    """Body of POST /api/enhance (prompt rewriting only)."""

    prompt: str = Field(min_length=1)
    mode: str | None = None
    instruction: str | None = None
    model: str | None = None
    engine: str | None = None
    is_edit: bool = False
    negative_prompt: str | None = None
    size: str | None = None
    #: Reference images (a data: URL or bare base64 each) for an enhancer that
    #: rewrites edits, plus their "WxH" sizes so a `ratio_follow` answer can
    #: name a canvas without the backend decoding the images.
    images: list[str] = Field(default_factory=list)
    image_sizes: list[str] = Field(default_factory=list)


class EnhanceResult(BaseModel):
    prompt: str
    original_prompt: str
    model: str
    base_url: str
    engine: str | None = None
    kind: str = "chat"
    instruction: str | None = None
    negative_prompt: str = ""
    wh_ratio: str = ""
    ratio_follow: str = ""
    suggested_size: str = ""
    parse_ok: bool = True
    thinking: str = ""
    answer: str = ""
    system_prompt_source: str = ""
    image_inputs: bool = False
    image_count: int = 0
    elapsed_s: float
    usage: dict | None = None
