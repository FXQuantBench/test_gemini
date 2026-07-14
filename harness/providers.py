"""harness/providers.py — LLM provider adapters with constrained/structured output.

Each adapter has the signature:
    (model_id: str, api_key: str, system_prompt: str, user_message: str) -> dict

The returned dict always has the shape:
    {
        "thoughts": str,
        "file_changes": [{"path": str, "content": str}, ...],
        "commands": [str, ...]
    }

Raises ValueError(raw_response) if the provider returns a malformed or empty response.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

# ---------------------------------------------------------------------------
# Retry / backoff helper
# ---------------------------------------------------------------------------

_RETRY_TIMEOUT_SECONDS = 120
_RETRY_BASE_DELAY = 1.0   # seconds
_RETRY_MAX_DELAY = 30.0   # seconds


def _call_with_backoff(fn, *args, **kwargs):
    """Call *fn* with exponential backoff until it succeeds or 120 s have elapsed.

    Retries on transient HTTP errors (rate-limits, server errors). Any other
    exception propagates immediately.
    """
    deadline = time.monotonic() + _RETRY_TIMEOUT_SECONDS
    delay = _RETRY_BASE_DELAY
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            # Classify as retryable based on exception type / message
            exc_str = str(exc).lower()
            retryable = any(
                token in exc_str
                for token in ("429", "rate limit", "rate_limit", "529",
                              "overloaded", "503", "502", "500",
                              "server error", "timeout", "timed out")
            )
            if not retryable:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"API call did not succeed within {_RETRY_TIMEOUT_SECONDS}s "
                    f"after {attempt + 1} attempt(s). Last error: {exc}"
                ) from exc
            sleep_for = min(delay, remaining)
            jitter = random.uniform(0, sleep_for * 0.2)
            time.sleep(sleep_for + jitter)
            delay = min(delay * 2, _RETRY_MAX_DELAY)
            attempt += 1

# ---------------------------------------------------------------------------
# JSON Schema shared across all providers
# ---------------------------------------------------------------------------

_FILE_CHANGE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}

_RESEARCH_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "hypothesis": {"type": "string"},
        "verdict": {"type": "string"},
    },
    "required": ["id"],
    "additionalProperties": False,
}

_RECORD_UPDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "release_note": {"type": "string"},
        "research_updates": {
            "type": "array",
            "items": _RESEARCH_UPDATE_SCHEMA,
        },
    },
    "additionalProperties": False,
}

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "thoughts": {"type": "string"},
        "file_changes": {
            "type": "array",
            "items": _FILE_CHANGE_SCHEMA,
        },
        "commands": {
            "type": "array",
            "items": {"type": "string"},
        },
        "record_updates": _RECORD_UPDATES_SCHEMA,
    },
    "required": ["thoughts", "file_changes", "commands"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_response(obj: Any, raw: str) -> dict:
    """Verify parsed object matches expected schema shape; raise ValueError on failure."""
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict response, got {type(obj).__name__}. Raw: {raw}")
    for key in ("thoughts", "file_changes", "commands"):
        if key not in obj:
            raise ValueError(f"Response missing key '{key}'. Raw: {raw}")
    if not isinstance(obj["thoughts"], str) or not obj["thoughts"].strip():
        raise ValueError(f"'thoughts' must be a non-empty string. Raw: {raw}")
    if not isinstance(obj["file_changes"], list):
        raise ValueError(f"'file_changes' must be a list. Raw: {raw}")
    if not isinstance(obj["commands"], list):
        raise ValueError(f"'commands' must be a list. Raw: {raw}")
    if "record_updates" in obj:
        updates = obj["record_updates"]
        if not isinstance(updates, dict):
            raise ValueError(f"'record_updates' must be an object. Raw: {raw}")
        allowed = {"release_note", "research_updates"}
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(
                f"'record_updates' contains unsupported keys {sorted(unknown)}. Raw: {raw}"
            )
        if "release_note" in updates and not isinstance(updates["release_note"], str):
            raise ValueError(f"'record_updates.release_note' must be a string. Raw: {raw}")
        if "research_updates" in updates:
            research_updates = updates["research_updates"]
            if not isinstance(research_updates, list):
                raise ValueError(
                    f"'record_updates.research_updates' must be a list. Raw: {raw}"
                )
            for update in research_updates:
                if not isinstance(update, dict) or not isinstance(update.get("id"), str):
                    raise ValueError(
                        "Each research update must be an object with a string 'id'. "
                        f"Raw: {raw}"
                    )
                if set(update) - {"id", "hypothesis", "verdict"}:
                    raise ValueError(f"Research update contains unsupported keys. Raw: {raw}")
                for key in ("hypothesis", "verdict"):
                    if key in update and not isinstance(update[key], str):
                        raise ValueError(f"'research_updates.{key}' must be a string. Raw: {raw}")
    return obj


# ---------------------------------------------------------------------------
# Adapter: OpenAI-compatible (OpenAI, Mistral, any OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

def openai_compatible(
    model_id: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
) -> dict:
    """Use JSON Schema response_format for strictly-structured output.

    Supports any OpenAI-compatible endpoint. Set the ``MODEL_BASE_URL``
    environment variable to override the default OpenAI API base URL
    (e.g. ``https://api.mistral.ai/v1`` or a local vLLM/Ollama endpoint).
    When ``MODEL_BASE_URL`` is not set, the official OpenAI API is used.
    """
    import os  # noqa: PLC0415
    import openai  # noqa: PLC0415

    base_url = os.environ.get("MODEL_BASE_URL", "").strip() or None
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = _call_with_backoff(
        client.chat.completions.create,
        model=model_id,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "strategy_response",
                "strict": True,
                "schema": _RESPONSE_SCHEMA,
            },
        },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content if response.choices else ""
    if not raw:
        raise ValueError(f"OpenAI returned empty content. Full response: {response}")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenAI response is not valid JSON: {exc}. Raw: {raw}") from exc

    return _validate_response(obj, raw)


# ---------------------------------------------------------------------------
# Adapter: Anthropic (tool-use forced call)
# ---------------------------------------------------------------------------

_ANTHROPIC_TOOL_DEF = {
    "name": "respond",
    "description": "Emit your structured response for this agentic loop iteration.",
    "input_schema": _RESPONSE_SCHEMA,
}


def anthropic_adapter(
    model_id: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
) -> dict:
    """Force the model to call the 'respond' tool; extract tool_use input block."""
    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic(api_key=api_key)
    response = _call_with_backoff(
        client.messages.create,
        model=model_id,
        max_tokens=8192,
        system=system_prompt,
        tools=[_ANTHROPIC_TOOL_DEF],
        tool_choice={"type": "tool", "name": "respond"},
        messages=[{"role": "user", "content": user_message}],
    )

    # Find the tool_use content block
    tool_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_block is None:
        raw = str(response.content)
        raise ValueError(f"Anthropic response contained no tool_use block. Raw: {raw}")

    obj = tool_block.input  # already a dict
    raw = json.dumps(obj)
    return _validate_response(obj, raw)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _strip_additional_properties(schema: dict) -> dict:
    """Return a deep copy of *schema* with all 'additionalProperties' keys removed.

    Gemini rejects this field; OpenAI and Anthropic handle it fine as-is.
    """
    import copy  # noqa: PLC0415

    out = copy.deepcopy(schema)

    def _strip(obj: Any) -> None:
        if isinstance(obj, dict):
            obj.pop("additionalProperties", None)
            for v in obj.values():
                _strip(v)
        elif isinstance(obj, list):
            for item in obj:
                _strip(item)

    _strip(out)
    return out


# ---------------------------------------------------------------------------
# Adapter: Google Gemini
# ---------------------------------------------------------------------------

def google_adapter(
    model_id: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
) -> dict:
    """Use response_mime_type + response_schema for structured JSON output."""
    from google import genai  # noqa: PLC0415
    from google.genai import types  # noqa: PLC0415

    client = genai.Client(api_key=api_key)

    response = _call_with_backoff(
        client.models.generate_content,
        model=model_id,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=_strip_additional_properties(_RESPONSE_SCHEMA),
        ),
    )

    raw = response.text if hasattr(response, "text") else ""
    if not raw:
        raise ValueError(f"Google returned empty response. Full response: {response}")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Google response is not valid JSON: {exc}. Raw: {raw}") from exc

    return _validate_response(obj, raw)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS: dict[str, Any] = {
    "openai": openai_compatible,
    "anthropic": anthropic_adapter,
    "google": google_adapter,
}
