"""Vision Preprocessor for downstream text-only agents.

This module isolates multimodal image handling from the core reasoning model.
It accepts raw image bytes, sends them to a configured OpenAI-compatible vision
endpoint, and returns a short natural-language summary that can be injected into
plain-text conversation context.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "你是一个视觉预处理组件，不是主聊天模型。"
    "请把图片转写成给纯文本大模型使用的简短中文摘要。"
    "只输出客观可见信息和适度的情绪/氛围线索：人物、表情、动作、场景、物体、关系。"
    "不要长篇分析，不要 Markdown，不要提及你看到了无法确认的细节。"
    "如果图片信息很少，就直接说明信息有限。"
)


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def resolve_vision_config(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict[str, str]:
    """Resolve the vision endpoint configuration.

    The defaults are intentionally loose so the caller can point this at Doubao,
    DeepSeek-compatible gateways, or any OpenAI-compatible multimodal endpoint.
    """
    resolved = {
        "provider": (provider or _env("QQ_VISION_PROVIDER", "doubao") or "doubao").strip(),
        "model": (model or _env("QQ_VISION_MODEL", "")).strip(),
        "base_url": (base_url or _env("QQ_VISION_BASE_URL", "")).rstrip("/"),
        "api_key": (api_key or _env("QQ_VISION_API_KEY", "")).strip(),
    }

    if not resolved["base_url"]:
        fallback = _env("QQ_VISION_FALLBACK_BASE_URL", "")
        if fallback:
            resolved["base_url"] = fallback.rstrip("/")

    return resolved


def _coerce_image_bytes(image_data: Any) -> bytes:
    if isinstance(image_data, bytes):
        return image_data
    if isinstance(image_data, bytearray):
        return bytes(image_data)
    if isinstance(image_data, memoryview):
        return image_data.tobytes()

    if isinstance(image_data, (str, Path)):
        raw = str(image_data).strip()
        if raw.startswith("data:"):
            try:
                _, payload = raw.split(",", 1)
                return base64.b64decode(payload)
            except Exception as exc:
                raise ValueError("Invalid data URL for image_data") from exc
        path = Path(raw).expanduser()
        if path.is_file():
            return path.read_bytes()

    raise TypeError(f"Unsupported image_data type: {type(image_data)!r}")


def _guess_mime_type(mime_type: str, image_name: str) -> str:
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    guessed, _ = mimetypes.guess_type(image_name)
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/jpeg"


def _build_messages(image_name: str, context_text: str, mime_type: str, image_bytes: bytes) -> list[dict[str, Any]]:
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    user_text = [
        "请生成一个给纯文本聊天模型使用的简短图片摘要。",
        "要求：中文、1-3 句、尽量具体、不要 Markdown、不要编号。",
    ]
    if image_name:
        user_text.append(f"图片文件名：{image_name}")
    if context_text:
        user_text.append(f"用户上下文：{context_text}")

    return [
        {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "\n".join(user_text)},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


async def summarize_image(
    image_data: Any,
    *,
    image_name: str = "",
    mime_type: str = "image/jpeg",
    context_text: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    session: Optional[aiohttp.ClientSession] = None,
) -> Optional[str]:
    """Summarize an image for downstream plain-text reasoning.

    Returns a compact Chinese summary or ``None`` on configuration/network failure.
    """
    cfg = resolve_vision_config(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    if not cfg["base_url"] or not cfg["api_key"]:
        logger.debug(
            "Vision preprocessor not configured (base_url=%s, api_key=%s)",
            bool(cfg["base_url"]),
            bool(cfg["api_key"]),
        )
        return None

    image_bytes = _coerce_image_bytes(image_data)
    if not image_bytes:
        return None

    resolved_model = cfg["model"] or _env("QQ_VISION_MODEL", "doubao-vision") or "doubao-vision"
    resolved_mime = _guess_mime_type(mime_type, image_name)
    messages = _build_messages(image_name, context_text, resolved_mime, image_bytes)
    url = f"{cfg['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
        "User-Agent": "her_agent-vision-preprocessor/1.0",
    }
    payload = {
        "model": resolved_model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 200,
    }

    try:
        if session is None:
            async with aiohttp.ClientSession(trust_env=False) as owned_session:
                return await _post_and_extract_summary(owned_session, url, headers, payload, timeout)
        return await _post_and_extract_summary(session, url, headers, payload, timeout)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError) as exc:
        logger.warning("Vision preprocessor failed: %s", exc)
        return None


async def _post_and_extract_summary(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> Optional[str]:
    timeout_cfg = aiohttp.ClientTimeout(total=timeout)
    async with session.post(url, headers=headers, json=payload, timeout=timeout_cfg) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"vision api error {resp.status}: {text[:300]}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"vision api returned non-JSON response: {text[:300]}") from exc

    return _extract_summary_text(data)


def _extract_summary_text(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    part_text = part.get("text") or part.get("content")
                    if isinstance(part_text, str) and part_text.strip():
                        text_parts.append(part_text.strip())
            if text_parts:
                return "\n".join(text_parts).strip()
        text = first.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()
    if isinstance(payload.get("text"), str) and payload["text"].strip():
        return payload["text"].strip()

    return None

