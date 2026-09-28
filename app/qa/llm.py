"""Shared LLM interface for QA verification.

Provides ask_json for structured Pydantic outputs across Gemini Flash (reader)
and Claude Opus on Vertex AI (arbiter / reviewer), with automatic fallback to
Gemini 2.5 Pro on any Opus error.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any, Literal, Optional, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

try:
    from anthropic import AnthropicVertex
except ImportError:
    AnthropicVertex = None  # type: ignore

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_PROJECT = "cellular-cider-495602-r9"
DEFAULT_READER_MODEL = "gemini-2.5-flash"
DEFAULT_ARBITER_MODEL = "claude-opus-5-5"
DEFAULT_ARBITER_REGION = "global"
DEFAULT_FALLBACK_MODEL = "gemini-2.5-pro"


class LLMUnavailable(RuntimeError):
    """Raised when LLM calls are disabled via QA_OFFLINE=1 or unavailable."""
    pass


def llm_available() -> bool:
    """Check if LLM services are available (not offline)."""
    if os.environ.get("QA_OFFLINE") == "1":
        return False
    return True


def _get_gemini_client(location: str = "global") -> genai.Client:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
    return genai.Client(vertexai=True, project=project, location=location)


def _get_anthropic_client(region: Optional[str] = None) -> Any:
    if AnthropicVertex is None:
        raise LLMUnavailable("anthropic package not installed")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
    reg = region or os.environ.get("QA_ARBITER_REGION", DEFAULT_ARBITER_REGION)
    return AnthropicVertex(project_id=project, region=reg)


def _ask_gemini(
    prompt: str,
    schema: type[T],
    images: Optional[list[bytes]] = None,
    *,
    model: str,
    timeout_s: float = 90.0,
) -> T:
    client = _get_gemini_client(location="global")
    contents: list[Any] = []
    if images:
        for idx, img_bytes in enumerate(images, 1):
            mime = "image/png" if img_bytes.startswith(b"\x89PNG") else "image/jpeg"
            if len(images) > 1:
                contents.append(f"Image #{idx}:")
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
    contents.append(prompt)

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
        ),
    )
    raw = resp.text.strip() if resp.text else "{}"
    return schema.model_validate_json(raw)


def _extract_json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    else:
        # Look for outermost JSON object or array
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def _ask_opus(
    prompt: str,
    schema: type[T],
    images: Optional[list[bytes]] = None,
    *,
    model: str,
    region: Optional[str] = None,
    timeout_s: float = 90.0,
) -> T:
    client = _get_anthropic_client(region=region)
    content: list[dict[str, Any]] = []
    if images:
        for idx, img_bytes in enumerate(images, 1):
            mime = "image/png" if img_bytes.startswith(b"\x89PNG") else "image/jpeg"
            b64_str = base64.b64encode(img_bytes).decode("ascii")
            if len(images) > 1:
                content.append({"type": "text", "text": f"Image #{idx}:"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": b64_str,
                },
            })

    tool_name = "submit_result"
    tool = {
        "name": tool_name,
        "description": "Submit structured output",
        "input_schema": schema.model_json_schema(),
    }
    full_prompt = (
        f"{prompt}\n\nPlease submit your structured answer using the {tool_name} tool."
    )
    content.append({"type": "text", "text": full_prompt})

    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                system=f"You are a music sheet verification assistant. You must respond by calling the {tool_name} tool with the structured parameters.",
                tools=[tool],
                messages=[{"role": "user", "content": content}],
                timeout=timeout_s,
            )
            # 1. Check for tool_use block
            for block in resp.content:
                if block.type == "tool_use" and block.name == tool_name:
                    return schema.model_validate(block.input)

            # 2. Check for text blocks containing JSON
            for block in resp.content:
                if block.type == "text" and block.text:
                    parsed_dict = _extract_json_from_text(block.text)
                    return schema.model_validate(parsed_dict)

            raise ValueError(f"No tool_use or JSON text in Opus response: {resp.content}")
        except Exception as exc:
            last_exc = exc
            logger.warning("Opus attempt %d failed: %s", attempt + 1, exc)
            if attempt == 0:
                # Add explicit retry reminder
                content.append({
                    "type": "text",
                    "text": f"Error validating output: {exc}. Please call {tool_name} with valid parameters.",
                })

    raise last_exc or RuntimeError("Opus failed after retries")


def ask_json(
    prompt: str,
    schema: type[T],
    images: Optional[list[bytes]] = None,
    *,
    role: Literal["reader", "arbiter", "reviewer"] = "reader",
    model: Optional[str] = None,
    timeout_s: float = 90.0,
) -> T:
    """Query an LLM with structured Pydantic output.

    Args:
        prompt: Instruction string.
        schema: Target Pydantic model class.
        images: Optional list of raw image bytes (JPEG or PNG).
        role: 'reader' (Gemini Flash), 'arbiter' or 'reviewer' (Claude Opus with Gemini Pro fallback).
        model: Optional model name override.
        timeout_s: Timeout in seconds.

    Returns:
        Instance of schema.

    Raises:
        LLMUnavailable: When QA_OFFLINE=1 or services cannot be reached.
    """
    if os.environ.get("QA_OFFLINE") == "1":
        raise LLMUnavailable("QA_OFFLINE is set to 1; LLM operations are disabled.")

    if role == "reader":
        target_model = model or os.environ.get("QA_READER_MODEL", DEFAULT_READER_MODEL)
        try:
            return _ask_gemini(
                prompt,
                schema,
                images,
                model=target_model,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            logger.error("Reader LLM call failed: %s", exc)
            raise

    # arbiter or reviewer: Claude Opus on Vertex AI with Gemini Pro fallback
    target_opus = model or os.environ.get("QA_ARBITER_MODEL", DEFAULT_ARBITER_MODEL)
    target_region = os.environ.get("QA_ARBITER_REGION", DEFAULT_ARBITER_REGION)

    try:
        return _ask_opus(
            prompt,
            schema,
            images,
            model=target_opus,
            region=target_region,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        logger.warning(
            "Arbiter Opus model %s in %s failed: %s. Falling back to %s",
            target_opus,
            target_region,
            exc,
            DEFAULT_FALLBACK_MODEL,
        )
        try:
            return _ask_gemini(
                prompt,
                schema,
                images,
                model=os.environ.get("QA_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL),
                timeout_s=timeout_s,
            )
        except Exception as fallback_exc:
            logger.error("Fallback to %s failed: %s", DEFAULT_FALLBACK_MODEL, fallback_exc)
            raise fallback_exc from exc
