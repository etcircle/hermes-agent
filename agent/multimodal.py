"""Helpers for native multimodal user input handling.

Keeps gateway/runtime image passthrough logic small and reusable:
- local image file -> data URL conversion
- OpenAI chat content -> Responses API content conversion
- conservative native-vision capability detection
- text-only shadow messages for transcript persistence
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Optional, Tuple

from agent.models_dev import get_model_info, get_model_info_any_provider

_NATIVE_VISION_API_MODES = {"chat_completions", "codex_responses", "anthropic_messages"}
_IMAGE_MIME_FALLBACKS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}


def _candidate_model_ids(model: str) -> list[str]:
    candidates: list[str] = []
    raw = str(model or "").strip()
    if not raw:
        return candidates
    candidates.append(raw)
    if "/" in raw:
        suffix = raw.split("/", 1)[1].strip()
        if suffix and suffix not in candidates:
            candidates.append(suffix)
    return candidates


def _guess_image_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    return _IMAGE_MIME_FALLBACKS.get(path.suffix.lower(), "image/jpeg")


def image_file_to_data_url(image_path: str | Path) -> str:
    """Convert a local image file to a base64 data URL."""
    path = Path(image_path).expanduser()
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type = _guess_image_mime_type(path)
    return f"data:{mime_type};base64,{encoded}"


def build_multimodal_user_content(text: str, image_paths: list[str]) -> list[dict[str, Any]]:
    """Build OpenAI-compatible multimodal content blocks for a user message."""
    parts: list[dict[str, Any]] = []
    clean_text = str(text or "").strip()
    if clean_text:
        parts.append({"type": "text", "text": clean_text})
    for image_path in image_paths:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": image_file_to_data_url(image_path)},
            }
        )
    return parts


def build_persist_user_message(text: str, image_count: int) -> str:
    """Build a text-only shadow message safe for transcripts/session DB."""
    clean_text = str(text or "").strip()
    if image_count <= 0:
        return clean_text
    note = f"[User attached {image_count} image{'s' if image_count != 1 else ''}]"
    if clean_text:
        return f"{clean_text}\n\n{note}"
    return note


def prepend_text_to_user_content(content: Any, text: str) -> Any:
    """Prepend a plain text note to either string or multimodal content."""
    note = str(text or "").strip()
    if not note:
        return content
    if isinstance(content, list):
        return [{"type": "text", "text": note}, *content]
    if isinstance(content, str):
        return f"{note}\n\n{content}" if content else note
    return content


def convert_chat_content_to_responses(content: Any) -> Any:
    """Convert OpenAI chat content parts to Responses API content parts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""

    converted: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype == "text":
            converted.append({"type": "input_text", "text": part.get("text", "")})
        elif ptype == "image_url":
            image_data = part.get("image_url", {})
            url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data or "")
            entry: dict[str, Any] = {"type": "input_image", "image_url": url}
            detail = image_data.get("detail") if isinstance(image_data, dict) else None
            if detail:
                entry["detail"] = detail
            converted.append(entry)
        elif ptype in {"input_text", "input_image"}:
            converted.append(dict(part))
        else:
            text = part.get("text", "")
            if text:
                converted.append({"type": "input_text", "text": text})

    return converted or ""


def describe_user_payload_for_log(user_message: Any, limit: int = 80) -> str:
    """Human-readable preview for logs/debug output."""
    if isinstance(user_message, str):
        preview = user_message.replace("\n", " ")
        return (preview[:limit] + "...") if len(preview) > limit else preview
    if isinstance(user_message, list):
        image_count = 0
        text_count = 0
        for part in user_message:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"image_url", "input_image"}:
                image_count += 1
            elif part.get("type") in {"text", "input_text"}:
                text_count += 1
        preview = f"[multimodal user turn: {image_count} image{'s' if image_count != 1 else ''}, {text_count} text block{'s' if text_count != 1 else ''}]"
        return (preview[:limit] + "...") if len(preview) > limit else preview
    preview = str(user_message)
    return (preview[:limit] + "...") if len(preview) > limit else preview


def resolve_native_vision_support(
    provider: str,
    model: str,
    api_mode: str,
    base_url: str = "",
) -> Tuple[Optional[bool], str]:
    """Return whether native image passthrough is safe for this runtime.

    Returns:
      (True, reason)   -> native passthrough is supported
      (False, reason)  -> known unsupported
      (None, reason)   -> unknown / cannot determine safely
    """
    normalized_mode = str(api_mode or "").strip().lower()
    if normalized_mode not in _NATIVE_VISION_API_MODES:
        return False, f"api_mode '{api_mode or 'unknown'}' does not support native multimodal image input"

    normalized_provider = str(provider or "").strip().lower()
    model_candidates = _candidate_model_ids(model)
    if not model_candidates:
        return None, "No model configured for vision capability detection"

    for candidate in model_candidates:
        if normalized_provider:
            try:
                info = get_model_info(normalized_provider, candidate)
            except Exception:
                info = None
            if info is not None:
                return (
                    True if info.supports_vision() else False,
                    f"{info.id} {'supports' if info.supports_vision() else 'does not support'} vision",
                )

    for candidate in model_candidates:
        try:
            info = get_model_info_any_provider(candidate)
        except Exception:
            info = None
        if info is not None:
            return (
                True if info.supports_vision() else False,
                f"{info.id} {'supports' if info.supports_vision() else 'does not support'} vision",
            )

    base_label = str(base_url or "").strip() or "unknown endpoint"
    return None, (
        f"Could not determine vision support for provider={normalized_provider or 'unknown'}, "
        f"model={model_candidates[0]!r}, endpoint={base_label}"
    )
