"""Role-based model routing.

Each pipeline stage declares a *role* (not a model). The model for that role
is resolved at call time from, in order: an env var override, the
`model_routes.json` config file, then a built-in default. Every resolution
records which of those three tiers it came from, so the caller can log it
on a trace span rather than guess after the fact.

This is a config cascade, not an error-handling fallback: if a role has no
usable model (unknown role, blank string in config), resolution raises
immediately. Nothing here ever catches a failed model call and silently
retries with a different model — a run either used the model it resolved,
or it errored out visibly.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_fireworks import ChatFireworks

ROUTES_PATH = Path(__file__).resolve().parent.parent / "model_routes.json"

# Built-in last-resort tier, mirrored in model_routes.json. Both slugs have been
# confirmed live against this project's Fireworks account (see README's Model
# routing section) — re-verify against your own catalog if you change them.
DEFAULT_ROUTES: dict[str, str] = {
    "generate": "accounts/fireworks/models/qwen3p8-max",          # strong model: writes the SQL
    "validate": "accounts/fireworks/models/deepseek-v4p1-flash",  # cheap/fast: schema-grounded lint before execution
    "summarize": "accounts/fireworks/models/deepseek-v4p1-flash", # cheap/fast: presentation layer only
    "judge": "accounts/fireworks/models/deepseek-v4p1-flash",     # cheap/fast: eval-only faithfulness check on the summary
}

ROLES = tuple(DEFAULT_ROUTES)

# generate's default model is a reasoning model: it spends tokens on a hidden
# reasoning trace before emitting the structured `sql` field, so it needs far
# more headroom than validate/summarize, which are short classification/
# paraphrase tasks. Too small a budget here doesn't make the model "fail" at
# SQL — it gets truncated before it ever writes the SQL field.
ROLE_MAX_TOKENS: dict[str, int] = {
    "generate": 4096,
    "validate": 1024,
    "summarize": 512,
    "judge": 512,
}


class UnknownRoleError(KeyError):
    """Raised when a caller asks for a role that isn't part of the routing table."""


class RouteConfigError(ValueError):
    """Raised when a role resolves to an empty/invalid model string."""


@dataclass(frozen=True)
class ModelResolution:
    role: str
    model: str
    source: str  # "override" | "env" | "config_file" | "default"


_config_cache: Optional[dict] = None


def _load_config_file() -> dict:
    global _config_cache
    if _config_cache is None:
        if ROUTES_PATH.exists():
            with open(ROUTES_PATH) as f:
                _config_cache = json.load(f)
        else:
            _config_cache = {}
    return _config_cache


def resolve_model(role: str, override: Optional[str] = None) -> ModelResolution:
    """Resolve the model for a role. Raises rather than guessing on misconfiguration."""
    if role not in ROLES:
        raise UnknownRoleError(
            f"Unknown routing role {role!r}. Known roles: {', '.join(ROLES)}"
        )

    if override:
        return ModelResolution(role=role, model=override, source="override")

    env_key = f"MODEL_ROUTE_{role.upper()}"
    env_value = os.environ.get(env_key)
    if env_value:
        return ModelResolution(role=role, model=env_value, source="env")

    config = _load_config_file()
    if role in config:
        value = config[role]
        if not value:
            raise RouteConfigError(f"model_routes.json has a blank entry for role {role!r}")
        return ModelResolution(role=role, model=value, source="config_file")

    default = DEFAULT_ROUTES[role]
    return ModelResolution(role=role, model=default, source="default")


_llm_cache: dict[tuple[str, str, int], ChatFireworks] = {}


def get_llm_for_role(
    role: str, override: Optional[str] = None, max_tokens: Optional[int] = None
) -> tuple[ChatFireworks, ModelResolution]:
    """Build (or reuse) the chat client for a role, alongside the resolution that produced it.

    `max_tokens`, if omitted, comes from ROLE_MAX_TOKENS for the role rather than one
    shared constant — see the comment above ROLE_MAX_TOKENS for why generate needs more.
    """
    resolution = resolve_model(role, override=override)
    tokens = max_tokens if max_tokens is not None else ROLE_MAX_TOKENS.get(role, 1024)

    cache_key = (resolution.role, resolution.model, tokens)
    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = ChatFireworks(
            model=resolution.model,
            api_key=os.environ["FIREWORKS_API_KEY"],
            temperature=0,
            max_tokens=tokens,
        )
    return _llm_cache[cache_key], resolution


def make_span(resolution: ModelResolution, stage: str, start: float, usage: Optional[dict] = None, retries: int = 0) -> dict:
    """Build a trace entry recording which model actually served a stage, and why."""
    span = {
        "stage": stage,
        "role": resolution.role,
        "model": resolution.model,
        "source": resolution.source,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "retries": retries,
    }
    if usage:
        span["input_tokens"] = usage.get("input_tokens")
        span["output_tokens"] = usage.get("output_tokens")
    return span
