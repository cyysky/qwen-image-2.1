"""Environment driven configuration.

Everything the playground talks to - image endpoints, image models, the prompt
enhancer model and every default - is read from the environment, so the whole app
can be re-pointed by editing `.env` alone. Nothing outside of the fallback
defaults in this module is hardcoded.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

load_dotenv(ROOT_DIR / ".env")
load_dotenv(ROOT_DIR / ".env.local", override=True)


# --------------------------------------------------------------------------
# raw env helpers
# --------------------------------------------------------------------------
def env_str(name: str, default: str = "") -> str:
    """Read a string, treating blank values as "unset"."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def env_bool(name: str, default: bool = False) -> bool:
    value = env_str(name, "")
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on", "y"}


def env_int(name: str, default: int) -> int:
    try:
        return int(float(env_str(name, "")))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, ""))
    except ValueError:
        return default


def env_list(name: str, default: list[str] | None = None) -> list[str]:
    raw = env_str(name, "")
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def env_var_suffix(name: str) -> str:
    """Turn a human name into a safe env-var suffix, e.g. pixel-space -> PIXEL_SPACE."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


# --------------------------------------------------------------------------
# image endpoints
# --------------------------------------------------------------------------
DEFAULT_ENDPOINTS_RAW = (
    "remote|http://10.0.151.2:7853/v1||Qwen-Image-2.1 10.0.151.2:7853,"
    "pixel-space|https://router.pixel-space.co/||Pixel-Space Router"
)


@dataclass(frozen=True)
class ImageEndpoint:
    """One OpenAI-compatible image endpoint."""

    name: str
    base_url: str
    api_key: str = ""
    label: str = ""

    @property
    def display(self) -> str:
        return self.label or self.name

    def to_public_dict(self) -> dict:
        """Never expose the key itself, only whether one is set."""
        return {
            "name": self.name,
            "label": self.display,
            "base_url": self.base_url,
            "has_key": bool(self.api_key),
        }


def normalize_base_url(url: str, append_v1: bool = True) -> str:
    """Strip trailing slashes and append `/v1` when the caller forgot it."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return url
    if append_v1 and "/v1" not in url:
        url = f"{url}/v1"
    return url


def parse_endpoints(raw: str, append_v1: bool = True) -> tuple[ImageEndpoint, ...]:
    """Parse `name|base_url|api_key|label` entries separated by commas.

    The api_key may also be supplied out of band as `API_KEY_<NAME>`, which wins.
    """
    endpoints: list[ImageEndpoint] = []
    seen: set[str] = set()
    for chunk in (raw or "").replace("\n", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.split("|")]
        name = parts[0]
        if not name or name in seen:
            continue
        base_url = normalize_base_url(parts[1] if len(parts) > 1 else "", append_v1)
        if not base_url:
            continue
        api_key = parts[2] if len(parts) > 2 else ""
        label = parts[3] if len(parts) > 3 else ""
        api_key = env_str(f"API_KEY_{env_var_suffix(name)}", api_key)
        seen.add(name)
        endpoints.append(ImageEndpoint(name=name, base_url=base_url, api_key=api_key, label=label))
    return tuple(endpoints)


# --------------------------------------------------------------------------
# enhancer defaults
# --------------------------------------------------------------------------
DEFAULT_ENHANCER_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for the Qwen-Image-2.1 text-to-image diffusion model. "
    "Rewrite the user's rough idea into one single, vivid, production-ready image prompt. "
    "Rules: reply with the prompt text only - no preamble, no explanations, no markdown, "
    "no bullet points, no quotation marks around the whole prompt, no negative-prompt section. "
    "Write one dense paragraph of 60 to 140 words. Always cover: subject and action, "
    "composition and camera angle, lighting, materials and textures, colour palette, mood and "
    "atmosphere, environment or background, and rendering style. Prefer concrete visual nouns over "
    "abstract adjectives. Keep any text that must appear in the image exactly as written, wrapped in "
    "double quotes. Never add watermarks, signatures, borders, extra limbs or duplicated subjects. "
    "If transparency or a sticker is requested, state that the image is an RGBA image with "
    "transparency and a transparent background. Preserve the user's intent: never change the subject, "
    "the subject count, or a style that was explicitly requested."
)

DEFAULT_ENHANCER_EDIT_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for the Qwen-Image-2.1 image-editing model. The user supplies "
    "one or more reference images plus an edit instruction. Rewrite that instruction into one single, "
    "precise, production-ready edit prompt. Rules: reply with the prompt text only - no preamble, no "
    "explanations, no markdown, no quotation marks around the whole prompt. State the change to apply, "
    "exactly where it goes, and how it integrates with the existing scene: perspective, lighting, shadows, "
    "scale, materials and reflections. State clearly what must stay unchanged. Keep any in-image text in "
    "double quotes. 60 to 120 words."
)

DEFAULT_ENHANCER_PRESETS: dict[str, str] = {
    "enhance": (
        "Maximise overall fidelity: enrich subject detail, lighting, composition, "
        "materials and atmosphere while keeping the original intent intact."
    ),
    "photoreal": (
        "Make it photorealistic: natural lighting, true-to-life materials, shallow "
        "depth of field, subtle surface imperfections, 85mm lens look."
    ),
    "cinematic": (
        "Give it a cinematic film-still look: dramatic key light, volumetric haze, "
        "anamorphic framing, rich colour grade, movie-poster mood."
    ),
    "illustration": (
        "Render it as a polished digital illustration: clean line work, flat shaded "
        "colour, expressive shapes, storybook charm."
    ),
    "product": (
        "Turn it into a premium studio product shot: seamless backdrop, soft-box "
        "lighting, crisp reflections, catalogue-ready composition."
    ),
    "sticker": (
        "Make it a cute sticker design: bold clean outlines, flat vibrant colours, "
        "simple shapes, subject isolated on a transparent background."
    ),
}


def load_presets() -> dict[str, str]:
    """Presets from `ENHANCER_PRESET_<KEY>`, ordered by `ENHANCER_PRESETS`."""
    presets = dict(DEFAULT_ENHANCER_PRESETS)
    for key in list(presets):
        presets[key] = env_str(f"ENHANCER_PRESET_{env_var_suffix(key)}", presets[key])

    requested = env_list("ENHANCER_PRESETS")
    if requested:
        ordered: dict[str, str] = {}
        for key in requested:
            suffix = env_var_suffix(key)
            match = next((k for k in presets if env_var_suffix(k) == suffix), None)
            if match:
                ordered[match] = presets[match]
                continue
            value = env_str(f"ENHANCER_PRESET_{suffix}", "")
            if value:
                ordered[key] = value
        return ordered or presets
    return presets


# --------------------------------------------------------------------------
# enhancer engines
# --------------------------------------------------------------------------
#: `wh_ratio` -> native canvas, byte-for-byte the upstream WH_RATIO_TO_SIZE map.
#: A PE engine picks the ratio; the UI offers the matching size as one click.
WH_RATIO_TO_SIZE: dict[str, str] = {
    "1:1": "2048x2048",
    "4:3": "2400x1792",
    "3:4": "1792x2400",
    "3:2": "2528x1696",
    "2:3": "1696x2528",
    "16:9": "2752x1536",
    "9:16": "1536x2752",
}

#: The PE checkpoints ship their system prompt next to the weights, not in any
#: repo, so a copy of the T2I prompt is bundled under `prompts/` and this is the
#: fallback path. `ENHANCER_ENGINE_<KEY>_SYSTEM_PROMPT_FILE` overrides it.
DEFAULT_PE_T2I_SYSTEM_PROMPT_FILE = "prompts/pe_t2i_system_prompt.txt"

#: Same for the edit checkpoint: `Qwen-Image-2.1-PE-I2I/system_prompt.txt` is the
#: 18 KB edit prompt, bundled verbatim under `prompts/`.
DEFAULT_PE_I2I_SYSTEM_PROMPT_FILE = "prompts/pe_i2i_system_prompt.txt"

#: Built-in per-engine defaults. `ENHANCER_ENGINE_<KEY>_<FIELD>` overrides any
#: of them, and the legacy `ENHANCER_*` variables still override the first
#: engine, so an existing `.env` keeps working unchanged.
DEFAULT_ENHANCER_ENGINES: dict[str, dict] = {
    "ds4-flash": {
        "label": "ds4-flash (router)",
        "kind": "chat",
        "base_url": "https://router.pixel-space.co/v1",
        "chat_path": "/chat/completions",
        "model": "ds4-flash",
        "temperature": 0.7,
        "max_tokens": 700,
        "timeout_s": 120.0,
        "system_prompt": DEFAULT_ENHANCER_SYSTEM_PROMPT,
        "edit_system_prompt": DEFAULT_ENHANCER_EDIT_SYSTEM_PROMPT,
    },
    "pe-t2i": {
        "label": "Qwen-Image-2.1-PE-T2I",
        "kind": "pe",
        "base_url": "http://10.0.151.2:8104/v1",
        "chat_path": "/chat/completions",
        "model": "Qwen/Qwen-Image-2.1-PE-T2I",
        # Upstream t2i production profile (pe_core.Profile): presence_penalty
        # 1.5 is what the checkpoint was trained with; 0 is the edit profile.
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "max_tokens": 16256,
        "timeout_s": 900.0,
        "seed": 42,
        "enable_thinking": True,
        "system_prompt_file": DEFAULT_PE_T2I_SYSTEM_PROMPT_FILE,
    },
    "pe-i2i": {
        "label": "Qwen-Image-2.1-PE-I2I",
        "kind": "pe-i2i",
        "base_url": "http://10.0.151.2:8105/v1",
        "chat_path": "/chat/completions",
        "model": "Qwen/Qwen-Image-2.1-PE-I2I",
        # Upstream *edit* profile: presence_penalty 0.0 and a 24000-token
        # budget, because an edit answer plus its thinking is longer than t2i's.
        # The host runs with --reasoning-parser qwen3 and leaves thinking on,
        # so unlike the t2i host no chat_template_kwargs are sent.
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 24000,
        "timeout_s": 900.0,
        "seed": 42,
        # The edit checkpoint is multimodal: the user message carries the
        # reference images next to the instruction.
        "image_inputs": True,
        "edit_system_prompt_file": DEFAULT_PE_I2I_SYSTEM_PROMPT_FILE,
    },
}

DEFAULT_ENHANCER_ENGINES_RAW = "ds4-flash,pe-t2i,pe-i2i"


@dataclass(frozen=True)
class EnhancerEngine:
    """One selectable prompt enhancer.

    `kind` decides the contract: `chat` is an ordinary chat model that replies
    with the rewritten prompt as plain text, while the Qwen-Image-2.1-PE
    checkpoints reply with a JSON record carrying `rewritten_prompt`, `wh_ratio`,
    `ratio_follow` and a `parse_ok` flag. `pe` is the text-to-image checkpoint
    and `pe-i2i` the edit one, which also takes the reference images in the
    user message.
    """

    key: str
    label: str = ""
    kind: str = "chat"
    base_url: str = ""
    chat_path: str = "/chat/completions"
    model: str = ""
    api_key: str = ""
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 700
    timeout_s: float = 120.0
    seed: int | None = None
    #: vLLM `chat_template_kwargs.enable_thinking`; None omits it and lets the
    #: checkpoint's chat template decide, which is what upstream's edit body does.
    enable_thinking: bool | None = None
    #: Whether the user message carries the reference images (PE-I2I).
    image_inputs: bool = False
    system_prompt: str = ""
    edit_system_prompt: str = ""
    system_prompt_file: str = ""
    edit_system_prompt_file: str = ""

    @property
    def display(self) -> str:
        return self.label or self.key

    @property
    def is_pe(self) -> bool:
        return self.kind.strip().lower() in {"pe", "pe-i2i"}

    @property
    def supports_t2i(self) -> bool:
        """Whether this engine has a rewrite prompt for the text-to-image task."""
        return bool(self.system_prompt or self.system_prompt_file)

    @property
    def supports_edit(self) -> bool:
        """Whether this engine has a rewrite prompt for the edit task."""
        return bool(self.edit_system_prompt or self.edit_system_prompt_file)

    def url(self) -> str:
        path = self.chat_path or "/chat/completions"
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url.rstrip('/')}{path}"

    def resolve_system_prompt_file(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else ROOT_DIR / path

    def system_prompt_for(self, is_edit: bool) -> tuple[str, str]:
        """Return `(prompt_text, source)` for the task, loading a file if set."""
        relative = self.edit_system_prompt_file if is_edit else self.system_prompt_file
        inline = self.edit_system_prompt if is_edit else self.system_prompt
        if relative:
            path = self.resolve_system_prompt_file(relative)
            if path.is_file():
                return path.read_text(encoding="utf-8").strip(), str(path)
            if not inline:
                suffix = env_var_suffix(self.key)
                raise LookupError(
                    f"enhancer '{self.key}': system prompt file {path} is missing - "
                    f"set ENHANCER_ENGINE_{suffix}_SYSTEM_PROMPT or "
                    f"ENHANCER_ENGINE_{suffix}_SYSTEM_PROMPT_FILE"
                )
        return inline, ("inline" if inline else "")

    def to_public_dict(self) -> dict:
        """Browser-safe view: never the key itself, only whether one is set."""
        return {
            "key": self.key,
            "label": self.display,
            "kind": self.kind,
            "model": self.model,
            "base_url": self.base_url,
            "url": self.url() if self.base_url else "",
            "has_key": bool(self.api_key),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
            "seed": self.seed,
            "supports_t2i": self.supports_t2i,
            "supports_edit": self.supports_edit,
            "image_inputs": self.image_inputs,
            "system_prompt_chars": len(self.system_prompt or ""),
            "system_prompt_file": self.system_prompt_file,
        }


def _engine_env(suffix: str, name: str, legacy: str = "") -> str:
    """Read `ENHANCER_ENGINE_<SUFFIX>_<NAME>`, then the legacy variable."""
    value = env_str(f"ENHANCER_ENGINE_{suffix}_{name}", "")
    if value:
        return value
    return env_str(legacy, "") if legacy else ""


def _engine_float(suffix: str, name: str, legacy: str, fallback: float) -> float:
    text = _engine_env(suffix, name, legacy)
    if text:
        try:
            return float(text)
        except ValueError:
            pass
    return float(fallback)


def _engine_int(suffix: str, name: str, legacy: str, fallback: int) -> int:
    text = _engine_env(suffix, name, legacy)
    if text:
        try:
            return int(float(text))
        except ValueError:
            pass
    return int(fallback)


def _engine_seed(suffix: str, defaults: dict) -> int | None:
    """Seed is optional: unset (and no built-in default) means "do not send"."""
    if not _engine_env(suffix, "SEED", "") and "seed" not in defaults:
        return None
    value = _engine_int(suffix, "SEED", "", int(defaults.get("seed", -1)))
    return value if value >= 0 else None


def _engine_bool(suffix: str, name: str, fallback: bool) -> bool:
    """A boolean engine setting whose unset value falls back to the default."""
    text = _engine_env(suffix, name, "")
    if not text:
        return bool(fallback)
    return text.strip().lower() in {"1", "true", "yes", "on", "y"}


def _engine_optional_bool(suffix: str, name: str, fallback: bool | None) -> bool | None:
    """A tri-state setting: unset (or blank) keeps `fallback`.

    `None` means "omit the field and let the checkpoint's chat template
    decide", which is what upstream's edit body does.
    """
    text = env_str(f"ENHANCER_ENGINE_{suffix}_{name}", "")
    if not text:
        return fallback
    return text.strip().lower() in {"1", "true", "yes", "on", "y"}


def load_engines(append_v1: bool = True) -> tuple[EnhancerEngine, ...]:
    """Engines from `ENHANCER_ENGINES`, ordered; the first one is the default."""
    keys = env_list("ENHANCER_ENGINES", list(DEFAULT_ENHANCER_ENGINES))
    engines: list[EnhancerEngine] = []
    seen: set[str] = set()
    for index, raw_key in enumerate(keys):
        key = raw_key.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        defaults = DEFAULT_ENHANCER_ENGINES.get(key, {})
        suffix = env_var_suffix(key)
        legacy = (lambda name: f"ENHANCER_{name}") if index == 0 else (lambda name: "")

        kind = _engine_env(suffix, "KIND", "").lower() or defaults.get("kind", "chat")
        if kind not in {"chat", "pe", "pe-i2i"}:
            kind = "chat"
        base_url = normalize_base_url(
            _engine_env(suffix, "BASE_URL", legacy("BASE_URL"))
            or defaults.get("base_url", ""),
            append_v1,
        )
        engines.append(
            EnhancerEngine(
                key=key,
                label=_engine_env(suffix, "LABEL", "") or defaults.get("label", key),
                kind=kind,
                base_url=base_url,
                chat_path=_engine_env(suffix, "CHAT_PATH", legacy("CHAT_PATH"))
                or defaults.get("chat_path", "/chat/completions"),
                model=_engine_env(suffix, "MODEL", legacy("MODEL"))
                or defaults.get("model", key),
                api_key=_engine_env(suffix, "API_KEY", legacy("API_KEY"))
                or defaults.get("api_key", ""),
                temperature=_engine_float(
                    suffix, "TEMPERATURE", legacy("TEMPERATURE"), defaults.get("temperature", 0.7)
                ),
                top_p=_engine_float(suffix, "TOP_P", "", defaults.get("top_p", 1.0)),
                top_k=_engine_int(suffix, "TOP_K", "", defaults.get("top_k", 0)),
                min_p=_engine_float(suffix, "MIN_P", "", defaults.get("min_p", 0.0)),
                presence_penalty=_engine_float(
                    suffix, "PRESENCE_PENALTY", "", defaults.get("presence_penalty", 0.0)
                ),
                max_tokens=_engine_int(
                    suffix, "MAX_TOKENS", legacy("MAX_TOKENS"), defaults.get("max_tokens", 700)
                ),
                timeout_s=_engine_float(
                    suffix, "TIMEOUT_S", legacy("TIMEOUT_S"), defaults.get("timeout_s", 120.0)
                ),
                seed=_engine_seed(suffix, defaults),
                enable_thinking=_engine_optional_bool(
                    suffix, "ENABLE_THINKING", defaults.get("enable_thinking")
                ),
                image_inputs=_engine_bool(
                    suffix, "IMAGE_INPUTS", defaults.get("image_inputs", kind == "pe-i2i")
                ),
                system_prompt=_engine_env(suffix, "SYSTEM_PROMPT", legacy("SYSTEM_PROMPT"))
                or defaults.get("system_prompt", ""),
                edit_system_prompt=_engine_env(
                    suffix, "EDIT_SYSTEM_PROMPT", legacy("EDIT_SYSTEM_PROMPT")
                )
                or defaults.get("edit_system_prompt", ""),
                system_prompt_file=_engine_env(suffix, "SYSTEM_PROMPT_FILE", "")
                or defaults.get("system_prompt_file", ""),
                edit_system_prompt_file=_engine_env(suffix, "EDIT_SYSTEM_PROMPT_FILE", "")
                or defaults.get("edit_system_prompt_file", ""),
            )
        )
    return tuple(engines)


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    # web server
    host: str = "127.0.0.1"
    port: int = 7860
    cors_origins: tuple[str, ...] = ("*",)
    health_timeout_s: float = 15.0
    image_timeout_s: float = 900.0

    # image endpoints / models
    endpoints: tuple[ImageEndpoint, ...] = ()
    default_endpoint: str = ""
    image_model: str = "Qwen-Image-2.1"
    image_models: tuple[str, ...] = ()
    sizes: tuple[str, ...] = ()
    max_reference_images: int = 10

    # generation defaults
    default_size: str = "1024x1024"
    default_steps: int = 40
    default_guidance_scale: float = 1.0
    default_output_format: str = "png"
    response_format: str = "b64_json"
    enhance_by_default: bool = False

    # enhancer
    enhancer_enabled: bool = True
    enhancer_engines: tuple[EnhancerEngine, ...] = ()
    default_enhancer: str = ""
    enhancer_ratio_sizes: dict[str, str] = field(default_factory=dict)
    enhancer_presets: dict[str, str] = field(default_factory=dict)

    # ---- lookups ---------------------------------------------------------
    def endpoint(self, name: str | None = None) -> ImageEndpoint:
        """Resolve an endpoint by name, falling back to the default one."""
        wanted = (name or "").strip() or self.default_endpoint
        for endpoint in self.endpoints:
            if endpoint.name == wanted:
                return endpoint
        if self.endpoints:
            return self.endpoints[0]
        raise LookupError("No image endpoints configured - check IMAGE_ENDPOINTS in .env")

    def enhancer_engine(self, key: str | None = None) -> EnhancerEngine:
        """Resolve an enhancer by key.

        An unknown key is an error rather than a silent fallback to the first
        engine: quietly rewriting with a different model than the one the user
        picked would be a surprise.
        """
        wanted = (key or "").strip() or self.default_enhancer
        for engine in self.enhancer_engines:
            if engine.key == wanted:
                return engine
        for engine in self.enhancer_engines:
            if engine.key.lower() == wanted.lower():
                return engine
        if not self.enhancer_engines:
            raise LookupError(
                "No prompt enhancer engines configured - check ENHANCER_ENGINES in .env"
            )
        raise LookupError(
            f"Unknown prompt enhancer engine '{wanted}' - "
            "check ENHANCER_ENGINES in .env"
        )

    def enhancer_url(self, key: str | None = None) -> str:
        try:
            return self.enhancer_engine(key).url()
        except LookupError:
            return ""

    def to_public_dict(self) -> dict:
        """Browser-safe view of the config (no secrets)."""
        return {
            "endpoints": [endpoint.to_public_dict() for endpoint in self.endpoints],
            "default_endpoint": self.default_endpoint
            or (self.endpoints[0].name if self.endpoints else ""),
            "models": list(self.image_models or (self.image_model,)),
            "default_model": self.image_model,
            "sizes": list(self.sizes),
            "max_reference_images": self.max_reference_images,
            "defaults": {
                "size": self.default_size,
                "num_inference_steps": self.default_steps,
                "guidance_scale": self.default_guidance_scale,
                "output_format": self.default_output_format,
                "background": "auto",
                "n": 1,
                "enhance": self.enhance_by_default,
            },
            "enhancer": {
                "enabled": self.enhancer_enabled and bool(self.enhancer_engines),
                "engines": [engine.to_public_dict() for engine in self.enhancer_engines],
                "default": self.default_enhancer
                or (self.enhancer_engines[0].key if self.enhancer_engines else ""),
                # Legacy single-engine view, kept so older clients keep working.
                "model": self.enhancer_engines[0].model if self.enhancer_engines else "",
                "models": [engine.model for engine in self.enhancer_engines],
                "base_url": self.enhancer_engines[0].base_url if self.enhancer_engines else "",
                "url": self.enhancer_url(),
                "ratio_sizes": dict(self.enhancer_ratio_sizes),
                "presets": [
                    {"key": key, "instruction": instruction}
                    for key, instruction in self.enhancer_presets.items()
                ],
            },
        }


def load_settings() -> Settings:
    """Build a Settings object from the current environment."""
    append_v1 = env_bool("APPEND_V1", True)
    endpoints = parse_endpoints(env_str("IMAGE_ENDPOINTS", DEFAULT_ENDPOINTS_RAW), append_v1)

    default_endpoint = env_str("DEFAULT_ENDPOINT", "")
    if endpoints and not any(item.name == default_endpoint for item in endpoints):
        default_endpoint = endpoints[0].name

    image_model = env_str("IMAGE_MODEL", "Qwen-Image-2.1")
    engines = load_engines(append_v1)

    default_enhancer = env_str("ENHANCER_ENGINE", "")
    if engines and not any(item.key == default_enhancer for item in engines):
        default_enhancer = engines[0].key

    return Settings(
        host=env_str("APP_HOST", "127.0.0.1"),
        port=env_int("APP_PORT", 7860),
        cors_origins=tuple(env_list("CORS_ORIGINS", ["*"])),
        health_timeout_s=env_float("HEALTH_TIMEOUT_S", 15.0),
        image_timeout_s=env_float("IMAGE_TIMEOUT_S", 900.0),
        endpoints=endpoints,
        default_endpoint=default_endpoint,
        image_model=image_model,
        image_models=tuple(env_list("IMAGE_MODELS", [image_model])),
        sizes=tuple(
            env_list(
                "SIZES",
                [
                    "1024x1024",
                    "2048x2048",
                    "2752x1536",
                    "1536x2752",
                    "2400x1792",
                    "1792x2400",
                    "2528x1696",
                    "1696x2528",
                ],
            )
        ),
        max_reference_images=env_int("MAX_REFERENCE_IMAGES", 10),
        default_size=env_str("DEFAULT_SIZE", "1024x1024"),
        default_steps=env_int("DEFAULT_STEPS", 40),
        default_guidance_scale=env_float("DEFAULT_GUIDANCE_SCALE", 1.0),
        default_output_format=env_str("DEFAULT_OUTPUT_FORMAT", "png"),
        response_format=env_str("IMAGE_RESPONSE_FORMAT", "b64_json"),
        enhance_by_default=env_bool("ENHANCER_APPLY_BY_DEFAULT", False),
        enhancer_enabled=env_bool("ENHANCER_ENABLED", True),
        enhancer_engines=engines,
        default_enhancer=default_enhancer,
        enhancer_ratio_sizes=dict(WH_RATIO_TO_SIZE),
        enhancer_presets=load_presets(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reload_settings() -> Settings:
    """Re-read `.env` without restarting the server (POST /api/config/reload)."""
    load_dotenv(ROOT_DIR / ".env", override=True)
    load_dotenv(ROOT_DIR / ".env.local", override=True)
    get_settings.cache_clear()
    return get_settings()
