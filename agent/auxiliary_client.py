"""Shared auxiliary client router for side tasks.

Provides a single resolution chain so every consumer (context compression,
session search, web extraction, vision analysis, browser vision) picks up
the best available backend without duplicating fallback logic.

Resolution order for text tasks (auto mode):
  1. OpenRouter  (OPENROUTER_API_KEY)
  2. Nous Portal (~/.her_agent/auth.json active provider)
  3. Custom endpoint (cli-config.yaml model.base_url + OPENAI_API_KEY)
  4. Codex OAuth (Responses API via chatgpt.com with gpt-5.3-codex,
     wrapped to look like a chat.completions client)
  5. Native Anthropic
  6. Direct API-key providers (z.ai/GLM, Kimi/Moonshot, MiniMax, MiniMax-CN)
  7. None

Resolution order for vision/multimodal tasks (auto mode):
  1. Selected main provider, if it is one of the supported vision backends below
  2. OpenRouter
  3. Nous Portal
  4. Codex OAuth (gpt-5.3-codex supports vision via Responses API)
  5. Native Anthropic
  6. Custom endpoint (for local vision models: Qwen-VL, LLaVA, Pixtral, etc.)
  7. None

Per-task overrides are configured in cli-config.yaml under the ``auxiliary:`` section
(e.g. ``auxiliary.vision.provider``, ``auxiliary.compression.model``).
Default "auto" follows the chains above.

Payment / credit exhaustion fallback:
  When a resolved provider returns HTTP 402 or a credit-related error,
  call_llm() automatically retries with the next available provider in the
  auto-detection chain.  This handles the common case where a user depletes
  their OpenRouter balance but has Codex OAuth or another provider available.
"""

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path  # noqa: F401 — used by test mocks
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from openai import OpenAI

from agent.credential_pool import load_pool
from her_cli.config import get_hermes_home
from her_constants import OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)

# Module-level flag: only warn once per process about stale OPENAI_BASE_URL.
_stale_base_url_warned = False

_PROVIDER_ALIASES = {
    "google": "gemini",
    "google-gemini": "gemini",
    "google-ai-studio": "gemini",
    "x-ai": "xai",
    "x.ai": "xai",
    "grok": "xai",
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "zhipu": "zai",
    "kimi": "kimi-coding",
    "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn",
    "moonshot-cn": "kimi-coding-cn",
    "minimax-china": "minimax-cn",
    "minimax_cn": "minimax-cn",
    "claude": "anthropic",
    "claude-code": "anthropic",
}


def _normalize_aux_provider(provider: Optional[str]) -> str:
    normalized = (provider or "auto").strip().lower()
    if normalized.startswith("custom:"):
        suffix = normalized.split(":", 1)[1].strip()
        if not suffix:
            return "custom"
        normalized = suffix
    if normalized == "codex":
        return "openai-codex"
    if normalized == "main":
        # Resolve to the user's actual main provider so named custom providers
        # and non-aggregator providers (DeepSeek, Alibaba, etc.) work correctly.
        main_prov = _read_main_provider()
        if main_prov and main_prov not in ("auto", "main", ""):
            return main_prov
        return "custom"
    return _PROVIDER_ALIASES.get(normalized, normalized)


_FIXED_TEMPERATURE_MODELS: Dict[str, float] = {
    "kimi-for-coding": 0.6,
}

# Moonshot's kimi-for-coding endpoint (api.kimi.com/coding) documents:
# "k2.5 model will use a fixed value 1.0, non-thinking mode will use a fixed
# value 0.6.  Any other value will result in an error."  The same lock applies
# to the other k2.* models served on that endpoint.  Enumerated explicitly so
# non-coding siblings like `kimi-k2-instruct` (variable temperature, served on
# the standard chat API and third parties) are NOT clamped.
# Source: https://platform.kimi.ai/docs/guide/kimi-k2-5-quickstart
_KIMI_INSTANT_MODELS: frozenset = frozenset({
    "kimi-k2.5",
    "kimi-k2-turbo-preview",
    "kimi-k2-0905-preview",
})
_KIMI_THINKING_MODELS: frozenset = frozenset({
    "kimi-k2-thinking",
    "kimi-k2-thinking-turbo",
})

# Moonshot's public chat endpoint (api.moonshot.ai/v1) enforces a different
# temperature contract than the Coding Plan endpoint above.  Empirically,
# `kimi-k2.5` on the public API rejects 0.6 with HTTP 400
# "invalid temperature: only 1 is allowed for this model" — the Coding Plan
# lock (0.6 for non-thinking) does not apply.  `kimi-k2-turbo-preview` and the
# thinking variants already match the Coding Plan contract on the public
# endpoint, so we only override the models that diverge.
# Users hit this endpoint when `KIMI_API_KEY` is a legacy `sk-*` key (the
# `sk-kimi-*` prefix routes to api.kimi.com/coding/v1 instead — see
# her_cli/auth.py:_kimi_base_url_for_key).
_KIMI_PUBLIC_API_OVERRIDES: Dict[str, float] = {
    "kimi-k2.5": 1.0,
}


def _fixed_temperature_for_model(
    model: Optional[str],
    base_url: Optional[str] = None,
) -> Optional[float]:
    """Return a required temperature override for models with strict contracts.

    Moonshot's kimi-for-coding endpoint rejects any non-approved temperature on
    the k2.5 family.  Non-thinking variants require exactly 0.6; thinking
    variants require 1.0.  An optional ``vendor/`` prefix (e.g.
    ``moonshotai/kimi-k2.5``) is tolerated for aggregator routings.

    When ``base_url`` points to Moonshot's public chat endpoint
    (``api.moonshot.ai``), the contract changes for ``kimi-k2.5``: the public
    API only accepts ``temperature=1``, not 0.6.  That override takes precedence
    over the Coding Plan defaults above.

    Returns ``None`` for every other model, including ``kimi-k2-instruct*``
    which is the separate non-coding K2 family with variable temperature.
    """
    normalized = (model or "").strip().lower()
    bare = normalized.rsplit("/", 1)[-1]

    # Public Moonshot API has a stricter contract for some models than the
    # Coding Plan endpoint — check it first so it wins on conflict.
    if base_url and ("api.moonshot.ai" in base_url.lower() or "api.moonshot.cn" in base_url.lower()):
        public = _KIMI_PUBLIC_API_OVERRIDES.get(bare)
        if public is not None:
            logger.debug(
                "Forcing temperature=%s for %r on public Moonshot API", public, model
            )
            return public

    fixed = _FIXED_TEMPERATURE_MODELS.get(normalized)
    if fixed is not None:
        logger.debug("Forcing temperature=%s for model %r (fixed map)", fixed, model)
        return fixed
    if bare in _KIMI_THINKING_MODELS:
        logger.debug("Forcing temperature=1.0 for kimi thinking model %r", model)
        return 1.0
    if bare in _KIMI_INSTANT_MODELS:
        logger.debug("Forcing temperature=0.6 for kimi instant model %r", model)
        return 0.6
    return None

# Default auxiliary models for direct API-key providers (cheap/fast for side tasks)
_API_KEY_PROVIDER_AUX_MODELS: Dict[str, str] = {
    "gemini": "gemini-3-flash-preview",
    "zai": "glm-4.5-flash",
    "kimi-coding": "kimi-k2-turbo-preview",
    "kimi-coding-cn": "kimi-k2-turbo-preview",
    "minimax": "MiniMax-M2.7",
    "minimax-cn": "MiniMax-M2.7",
    "anthropic": "claude-haiku-4-5-20251001",
    "ai-gateway": "google/gemini-3-flash",
    "opencode-zen": "gemini-3-flash",
    "opencode-go": "glm-5",
    "kilocode": "google/gemini-3-flash-preview",
    "ollama-cloud": "nemotron-3-nano:30b",
}

# Vision-specific model overrides for direct providers.
# When the user's main provider has a dedicated vision/multimodal model that
# differs from their main chat model, map it here.  The vision auto-detect
# "exotic provider" branch checks this before falling back to the main model.
_PROVIDER_VISION_MODELS: Dict[str, str] = {
    "xiaomi": "mimo-v2-omni",
    "zai": "glm-5v-turbo",
}

# OpenRouter app attribution headers
_OR_HEADERS = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-OpenRouter-Title": "Hermes Agent",
    "X-OpenRouter-Categories": "productivity,cli-agent",
}

# Nous Portal extra_body for product attribution.
# Callers should pass this as extra_body in chat.completions.create()
# when the auxiliary client is backed by Nous Portal.
NOUS_EXTRA_BODY = {"tags": ["product=her_agent-agent"]}

# Set at resolve time — True if the auxiliary client points to Nous Portal
auxiliary_is_nous: bool = False

# Default auxiliary models per provider
_OPENROUTER_MODEL = "google/gemini-3-flash-preview"
_NOUS_MODEL = "google/gemini-3-flash-preview"
_NOUS_FREE_TIER_VISION_MODEL = "xiaomi/mimo-v2-omni"
_NOUS_FREE_TIER_AUX_MODEL = "xiaomi/mimo-v2-pro"
_NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
_AUTH_JSON_PATH = get_hermes_home() / "auth.json"

# Codex fallback: uses the Responses API (the only endpoint the Codex
# OAuth token can access) with a fast model for auxiliary tasks.
# ChatGPT-backed Codex accounts currently reject gpt-5.3-codex for these
# auxiliary flows, while gpt-5.2-codex remains broadly available and supports
# vision via Responses.
_CODEX_AUX_MODEL = "gpt-5.2-codex"
_CODEX_AUX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_SILICONFLOW_DEFAULT_EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"
_SILICONFLOW_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
_ALAPI_DOUTU_URL = "https://v3.alapi.cn/api/doutu"


def _env_trim(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


async def get_embedding(
    text: str,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    session: Optional[aiohttp.ClientSession] = None,
) -> Optional[List[float]]:
    """Fetch a text embedding from SiliconFlow (OpenAI-compatible embeddings API)."""
    prompt = str(text or "").strip()
    if not prompt:
        return None

    resolved_base = (base_url or _env_trim("SILICONFLOW_API_URL", _SILICONFLOW_DEFAULT_EMBEDDING_URL)).strip().rstrip("/")
    resolved_key = (api_key or _env_trim("SILICONFLOW_API_KEY", "")).strip()
    resolved_model = (model or _env_trim("SILICONFLOW_EMBEDDING_MODEL", _SILICONFLOW_DEFAULT_EMBEDDING_MODEL)).strip()
    if not resolved_base or not resolved_key or not resolved_model:
        logger.debug(
            "Embedding config missing (base=%s, key=%s, model=%s)",
            bool(resolved_base), bool(resolved_key), bool(resolved_model),
        )
        return None

    payload = {"model": resolved_model, "input": [prompt]}
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
        "User-Agent": "her_agent-embedding-client/1.0",
    }
    timeout_cfg = aiohttp.ClientTimeout(total=timeout)

    try:
        if session is None:
            async with aiohttp.ClientSession(trust_env=False) as owned_session:
                return await _post_embedding_request(owned_session, resolved_base, headers, payload, timeout_cfg)
        return await _post_embedding_request(session, resolved_base, headers, payload, timeout_cfg)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError) as exc:
        logger.warning("Embedding request failed: %s", exc)
        return None


async def _post_embedding_request(
    session: aiohttp.ClientSession,
    base_url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_cfg: aiohttp.ClientTimeout,
) -> Optional[List[float]]:
    url = f"{base_url.rstrip('/')}/embeddings" if not base_url.rstrip('/').endswith("/embeddings") else base_url.rstrip('/')
    async with session.post(url, headers=headers, json=payload, timeout=timeout_cfg) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"embedding api error {resp.status}: {text[:300]}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"embedding api returned non-JSON response: {text[:300]}") from exc

    vector = _extract_embedding_vector(data)
    if vector is None:
        raise RuntimeError("embedding api response did not contain a vector")
    return vector


def _extract_embedding_vector(payload: Any) -> Optional[List[float]]:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("embedding"), list):
        return [float(v) for v in payload["embedding"]]
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                vector = item.get("embedding")
                if isinstance(vector, list):
                    return [float(v) for v in vector]
    return None


async def search_alapi_meme(
    keyword: str,
    *,
    token: Optional[str] = None,
    page: int = 1,
    timeout: float = 30.0,
    session: Optional[aiohttp.ClientSession] = None,
) -> Optional[str]:
    """Search ALAPI doutu and return the first meme image URL, if any."""
    query = str(keyword or "").strip()
    if not query:
        return None

    resolved_token = (token or _env_trim("ALAPI_TOKEN", "")).strip()
    if not resolved_token:
        logger.debug("ALAPI token missing; meme fallback disabled")
        return None

    payload = {"token": resolved_token, "keyword": query, "page": int(page or 1)}
    headers = {"Content-Type": "application/json", "User-Agent": "her_agent-alapi-client/1.0"}
    timeout_cfg = aiohttp.ClientTimeout(total=timeout)

    try:
        if session is None:
            async with aiohttp.ClientSession(trust_env=False) as owned_session:
                return await _post_alapi_request(owned_session, headers, payload, timeout_cfg)
        return await _post_alapi_request(session, headers, payload, timeout_cfg)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError) as exc:
        logger.warning("ALAPI meme search failed: %s", exc)
        return None


async def _post_alapi_request(
    session: aiohttp.ClientSession,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_cfg: aiohttp.ClientTimeout,
) -> Optional[str]:
    async with session.post(_ALAPI_DOUTU_URL, headers=headers, json=payload, timeout=timeout_cfg) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"alapi error {resp.status}: {text[:300]}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"alapi returned non-JSON response: {text[:300]}") from exc

    if not isinstance(data, dict) or int(data.get("code", 0) or 0) != 200:
        return None
    raw_data = data.get("data")
    if isinstance(raw_data, list) and raw_data:
        first = raw_data[0]
        if isinstance(first, dict):
            url = first.get("url") or first.get("pic") or first.get("image")
            if isinstance(url, str) and url.strip():
                return url.strip()
    if isinstance(raw_data, dict):
        url = raw_data.get("url") or raw_data.get("pic") or raw_data.get("image")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


# ── Provider resolution helpers ─────────────────────────────────────────────


def _codex_cloudflare_headers(access_token: str) -> Dict[str, str]:
    """Headers required to avoid Cloudflare 403s on chatgpt.com/backend-api/codex.

    The Cloudflare layer in front of the Codex endpoint whitelists a small set of
    first-party originators (``codex_cli_rs``, ``codex_vscode``, ``codex_sdk_ts``,
    anything starting with ``Codex``). Requests from non-residential IPs (VPS,
    server-hosted agents) that don't advertise an allowed originator are served
    a 403 with ``cf-mitigated: challenge`` regardless of auth correctness.
    """
    headers = {
        "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
        "originator": "codex_cli_rs",
    }
    if not isinstance(access_token, str) or not access_token.strip():
        return headers
    try:
        import base64
        parts = access_token.split(".")
        if len(parts) < 2:
            return headers
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        acct_id = claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id")
        if isinstance(acct_id, str) and acct_id:
            headers["ChatGPT-Account-ID"] = acct_id
    except Exception:
        pass
    return headers


def _to_openai_base_url(base_url: str) -> str:
    """Normalize an Anthropic-style base URL to OpenAI-compatible format."""
    url = str(base_url or "").strip().rstrip("/")
    if url.endswith("/anthropic"):
        rewritten = url[: -len("/anthropic")] + "/v1"
        logger.debug("Auxiliary client: rewrote base URL %s → %s", url, rewritten)
        return rewritten
    return url


def _select_pool_entry(provider: str) -> Tuple[bool, Optional[Any]]:
    """Return (pool_exists_for_provider, selected_entry)."""
    try:
        pool = load_pool(provider)
    except Exception as exc:
        logger.debug("Auxiliary client: could not load pool for %s: %s", provider, exc)
        return False, None
    if not pool or not pool.has_credentials():
        return False, None
    try:
        return True, pool.select()
    except Exception as exc:
        logger.debug("Auxiliary client: could not select pool entry for %s: %s", provider, exc)
        return True, None


def _pool_runtime_api_key(entry: Any) -> str:
    if entry is None:
        return ""
    key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    return str(key or "").strip()


def _pool_runtime_base_url(entry: Any, fallback: str = "") -> str:
    if entry is None:
        return str(fallback or "").strip().rstrip("/")
    url = (
        getattr(entry, "runtime_base_url", None)
        or getattr(entry, "inference_base_url", None)
        or getattr(entry, "base_url", None)
        or fallback
    )
    return str(url or "").strip().rstrip("/")


# ── Codex Responses → chat.completions adapter ─────────────────────────────


def _convert_content_for_responses(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""

    converted: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype == "text":
            converted.append({"type": "input_text", "text": part.get("text", "")})
        elif ptype == "image_url":
            image_data = part.get("image_url", {})
            url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data)
            entry: Dict[str, Any] = {"type": "input_image", "image_url": url}
            detail = image_data.get("detail") if isinstance(image_data, dict) else None
            if detail:
                entry["detail"] = detail
            converted.append(entry)
        elif ptype in ("input_text", "input_image"):
            converted.append(part)
        else:
            text = part.get("text", "")
            if text:
                converted.append({"type": "input_text", "text": text})

    return converted or ""


class _CodexCompletionsAdapter:
    def __init__(self, real_client: OpenAI, model: str):
        self._client = real_client
        self._model = model

    def create(self, **kwargs) -> Any:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self._model)
        instructions = "You are a helpful assistant."
        input_msgs: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "system":
                instructions = content if isinstance(content, str) else str(content)
            else:
                input_msgs.append({"role": role, "content": _convert_content_for_responses(content)})

        resp_kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_msgs or [{"role": "user", "content": ""}],
            "store": False,
        }

        tools = kwargs.get("tools")
        if tools:
            converted = []
            for t in tools:
                fn = t.get("function", {}) if isinstance(t, dict) else {}
                name = fn.get("name")
                if not name:
                    continue
                converted.append({
                    "type": "function",
                    "name": name,
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            if converted:
                resp_kwargs["tools"] = converted

        text_parts: List[str] = []
        tool_calls_raw: List[Any] = []
        usage = None

        try:
            collected_output_items: List[Any] = []
            collected_text_deltas: List[str] = []
            has_function_calls = False
            with self._client.responses.stream(**resp_kwargs) as stream:
                for _event in stream:
                    _etype = getattr(_event, "type", "")
                    if _etype == "response.output_item.done":
                        _done = getattr(_event, "item", None)
                        if _done is not None:
                            collected_output_items.append(_done)
                    elif "output_text.delta" in _etype:
                        _delta = getattr(_event, "delta", "")
                        if _delta:
                            collected_text_deltas.append(_delta)
                    elif "function_call" in _etype:
                        has_function_calls = True
                final = stream.get_final_response()

            _output = getattr(final, "output", None)
            if isinstance(_output, list) and not _output:
                if collected_output_items:
                    final.output = list(collected_output_items)
                    logger.debug(
                        "Codex auxiliary: backfilled %d output items from stream events",
                        len(collected_output_items),
                    )
                elif collected_text_deltas and not has_function_calls:
                    assembled = "".join(collected_text_deltas)
                    final.output = [SimpleNamespace(
                        type="message", role="assistant", status="completed",
                        content=[SimpleNamespace(type="output_text", text=assembled)],
                    )]
                    logger.debug(
                        "Codex auxiliary: synthesized from %d deltas (%d chars)",
                        len(collected_text_deltas), len(assembled),
                    )

            def _item_get(obj: Any, key: str, default: Any = None) -> Any:
                val = getattr(obj, key, None)
                if val is None and isinstance(obj, dict):
                    val = obj.get(key, default)
                return val if val is not None else default

            for item in getattr(final, "output", []):
                item_type = _item_get(item, "type")
                if item_type == "message":
                    for part in (_item_get(item, "content") or []):
                        ptype = _item_get(part, "type")
                        if ptype in ("output_text", "text"):
                            text_parts.append(_item_get(part, "text", ""))
                elif item_type == "function_call":
                    tool_calls_raw.append(SimpleNamespace(
                        id=_item_get(item, "call_id", ""),
                        type="function",
                        function=SimpleNamespace(
                            name=_item_get(item, "name", ""),
                            arguments=_item_get(item, "arguments", "{}"),
                        ),
                    ))

            resp_usage = getattr(final, "usage", None)
            if resp_usage:
                usage = SimpleNamespace(
                    prompt_tokens=getattr(resp_usage, "input_tokens", 0),
                    completion_tokens=getattr(resp_usage, "output_tokens", 0),
                    total_tokens=getattr(resp_usage, "total_tokens", 0),
                )
        except Exception as exc:
            logger.debug("Codex auxiliary Responses API call failed: %s", exc)
            raise

        content = "".join(text_parts).strip() or None
        message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls_raw or None)
        choice = SimpleNamespace(index=0, message=message, finish_reason="stop" if not tool_calls_raw else "tool_calls")
        return SimpleNamespace(choices=[choice], model=model, usage=usage)


class _CodexChatShim:
    def __init__(self, adapter: _CodexCompletionsAdapter):
        self.completions = adapter


class CodexAuxiliaryClient:
    def __init__(self, real_client: OpenAI, model: str):
        self._real_client = real_client
        adapter = _CodexCompletionsAdapter(real_client, model)
        self.chat = _CodexChatShim(adapter)
        self.api_key = real_client.api_key
        self.base_url = real_client.base_url

    def close(self):
        self._real_client.close()


class _AsyncCodexCompletionsAdapter:
    def __init__(self, sync_adapter: _CodexCompletionsAdapter):
        self._sync = sync_adapter

    async def create(self, **kwargs) -> Any:
        import asyncio
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncCodexChatShim:
    def __init__(self, adapter: _AsyncCodexCompletionsAdapter):
        self.completions = adapter


class AsyncCodexAuxiliaryClient:
    def __init__(self, sync_wrapper: "CodexAuxiliaryClient"):
        sync_adapter = sync_wrapper.chat.completions
        async_adapter = _AsyncCodexCompletionsAdapter(sync_adapter)
        self.chat = _AsyncCodexChatShim(async_adapter)
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url


class _AnthropicCompletionsAdapter:
    def __init__(self, real_client: Any, model: str, is_oauth: bool = False):
        self._client = real_client
        self._model = model
        self._is_oauth = is_oauth

    def create(self, **kwargs) -> Any:
        from agent.anthropic_adapter import build_anthropic_kwargs, normalize_anthropic_response

        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self._model)
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 2000
        temperature = kwargs.get("temperature")

        normalized_tool_choice = None
        if isinstance(tool_choice, str):
            normalized_tool_choice = tool_choice
        elif isinstance(tool_choice, dict):
            choice_type = str(tool_choice.get("type", "")).lower()
            if choice_type == "function":
                normalized_tool_choice = tool_choice.get("function", {}).get("name")
            elif choice_type in {"auto", "required", "none"}:
                normalized_tool_choice = choice_type

        anthropic_kwargs = build_anthropic_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            reasoning_config=None,
            tool_choice=normalized_tool_choice,
            is_oauth=self._is_oauth,
        )
        if temperature is not None:
            from agent.anthropic_adapter import _forbids_sampling_params
            if not _forbids_sampling_params(model):
                anthropic_kwargs["temperature"] = temperature

        response = self._client.messages.create(**anthropic_kwargs)
        assistant_message, finish_reason = normalize_anthropic_response(response)

        usage = None
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "input_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "output_tokens", 0) or 0
            total_tokens = getattr(response.usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
            usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)

        choice = SimpleNamespace(index=0, message=assistant_message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice], model=model, usage=usage)


class _AnthropicChatShim:
    def __init__(self, adapter: _AnthropicCompletionsAdapter):
        self.completions = adapter


class AnthropicAuxiliaryClient:
    def __init__(self, real_client: Any, model: str, api_key: str, base_url: str, is_oauth: bool = False):
        self._real_client = real_client
        adapter = _AnthropicCompletionsAdapter(real_client, model, is_oauth=is_oauth)
        self.chat = _AnthropicChatShim(adapter)
        self.api_key = api_key
        self.base_url = base_url

    def close(self):
        close_fn = getattr(self._real_client, "close", None)
        if callable(close_fn):
            close_fn()


class _AsyncAnthropicCompletionsAdapter:
    def __init__(self, sync_adapter: _AnthropicCompletionsAdapter):
        self._sync = sync_adapter

    async def create(self, **kwargs) -> Any:
        import asyncio
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncAnthropicChatShim:
    def __init__(self, adapter: _AsyncAnthropicCompletionsAdapter):
        self.completions = adapter


class AsyncAnthropicAuxiliaryClient:
    def __init__(self, sync_wrapper: "AnthropicAuxiliaryClient"):
        sync_adapter = sync_wrapper.chat.completions
        async_adapter = _AsyncAnthropicCompletionsAdapter(sync_adapter)
        self.chat = _AsyncAnthropicChatShim(async_adapter)
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url


def _read_nous_auth() -> Optional[dict]:
    """Read and validate ~/.her_agent/auth.json for an active Nous provider."""
    pool_present, entry = _select_pool_entry("nous")
    if pool_present:
        if entry is None:
            return None
        return {
            "access_token": getattr(entry, "access_token", ""),
            "refresh_token": getattr(entry, "refresh_token", None),
            "agent_key": getattr(entry, "agent_key", None),
            "inference_base_url": _pool_runtime_base_url(entry, _NOUS_DEFAULT_BASE_URL),
            "portal_base_url": getattr(entry, "portal_base_url", None),
            "client_id": getattr(entry, "client_id", None),
            "scope": getattr(entry, "scope", None),
            "token_type": getattr(entry, "token_type", "Bearer"),
            "source": "pool",
        }

    try:
        if not _AUTH_JSON_PATH.is_file():
            return None
        data = json.loads(_AUTH_JSON_PATH.read_text())
        if data.get("active_provider") != "nous":
            return None
        provider = data.get("providers", {}).get("nous", {})
        if not provider.get("agent_key") and not provider.get("access_token"):
            return None
        return provider
    except Exception as exc:
        logger.debug("Could not read Nous auth: %s", exc)
        return None


def _nous_api_key(provider: dict) -> str:
    return provider.get("agent_key") or provider.get("access_token", "")


def _nous_base_url() -> str:
    return os.getenv("NOUS_INFERENCE_BASE_URL", _NOUS_DEFAULT_BASE_URL)


def _read_codex_access_token() -> Optional[str]:
    pool_present, entry = _select_pool_entry("openai-codex")
    if pool_present:
        token = _pool_runtime_api_key(entry)
        if token:
            return token

    try:
        from her_cli.auth import _read_codex_tokens
        data = _read_codex_tokens()
        tokens = data.get("tokens", {})
        access_token = tokens.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            return None

        try:
            import base64
            payload = access_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            exp = claims.get("exp", 0)
            if exp and time.time() > exp:
                logger.debug("Codex access token expired (exp=%s), skipping", exp)
                return None
        except Exception:
            pass

        return access_token.strip()
    except Exception as exc:
        logger.debug("Could not read Codex auth for auxiliary client: %s", exc)
        return None


def _resolve_api_key_provider() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Try each API-key provider in PROVIDER_REGISTRY order."""
    try:
        from her_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials
    except ImportError:
        logger.debug("Could not import PROVIDER_REGISTRY for API-key fallback")
        return None, None

    for provider_id, pconfig in PROVIDER_REGISTRY.items():
        if pconfig.auth_type != "api_key":
            continue
        if provider_id == "anthropic":
            try:
                from her_cli.auth import is_provider_explicitly_configured
                if not is_provider_explicitly_configured("anthropic"):
                    continue
            except ImportError:
                pass
            return _try_anthropic()

        pool_present, entry = _select_pool_entry(provider_id)
        if pool_present:
            api_key = _pool_runtime_api_key(entry)
            if not api_key:
                continue

            base_url = _to_openai_base_url(
                _pool_runtime_base_url(entry, pconfig.inference_base_url) or pconfig.inference_base_url
            )
            model = _API_KEY_PROVIDER_AUX_MODELS.get(provider_id)
            if model is None:
                continue
            logger.debug("Auxiliary text client: %s (%s) via pool", pconfig.name, model)
            if provider_id == "gemini":
                from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

                if is_native_gemini_base_url(base_url):
                    return GeminiNativeClient(api_key=api_key, base_url=base_url), model
            extra = {}
            if "api.kimi.com" in base_url.lower():
                extra["default_headers"] = {"User-Agent": "KimiCLI/1.30.0"}
            elif "api.githubcopilot.com" in base_url.lower():
                from her_cli.models import copilot_default_headers

                extra["default_headers"] = copilot_default_headers()
            return OpenAI(api_key=api_key, base_url=base_url, **extra), model

        creds = resolve_api_key_provider_credentials(provider_id)
        api_key = str(creds.get("api_key", "")).strip()
        if not api_key:
            continue

        base_url = _to_openai_base_url(
            str(creds.get("base_url", "")).strip().rstrip("/") or pconfig.inference_base_url
        )
        model = _API_KEY_PROVIDER_AUX_MODELS.get(provider_id)
        if model is None:
            continue
        logger.debug("Auxiliary text client: %s (%s)", pconfig.name, model)
        if provider_id == "gemini":
            from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

            if is_native_gemini_base_url(base_url):
                return GeminiNativeClient(api_key=api_key, base_url=base_url), model
        extra = {}
        if "api.kimi.com" in base_url.lower():
            extra["default_headers"] = {"User-Agent": "KimiCLI/1.30.0"}
        elif "api.githubcopilot.com" in base_url.lower():
            from her_cli.models import copilot_default_headers

            extra["default_headers"] = copilot_default_headers()
        return OpenAI(api_key=api_key, base_url=base_url, **extra), model

    return None, None


def _try_openrouter() -> Tuple[Optional[OpenAI], Optional[str]]:
    pool_present, entry = _select_pool_entry("openrouter")
    if pool_present:
        or_key = _pool_runtime_api_key(entry)
        if not or_key:
            return None, None
        base_url = _pool_runtime_base_url(entry, OPENROUTER_BASE_URL) or OPENROUTER_BASE_URL
        logger.debug("Auxiliary client: OpenRouter via pool")
        return OpenAI(api_key=or_key, base_url=base_url,
                       default_headers=_OR_HEADERS), _OPENROUTER_MODEL

    or_key = os.getenv("OPENROUTER_API_KEY")
    if not or_key:
        return None, None
    logger.debug("Auxiliary client: OpenRouter")
    return OpenAI(api_key=or_key, base_url=OPENROUTER_BASE_URL,
                   default_headers=_OR_HEADERS), _OPENROUTER_MODEL


def _try_nous(vision: bool = False) -> Tuple[Optional[OpenAI], Optional[str]]:
    # Check cross-session rate limit guard before attempting Nous —
    # if another session already recorded a 429, skip Nous entirely
    # to avoid piling more requests onto the tapped RPH bucket.
    try:
        from agent.nous_rate_guard import nous_rate_limit_remaining
        _remaining = nous_rate_limit_remaining()
        if _remaining is not None and _remaining > 0:
            logger.debug(
                "Auxiliary: skipping Nous Portal (rate-limited, resets in %.0fs)",
                _remaining,
            )
            return None, None
    except Exception:
        pass

    nous = _read_nous_auth()
    if not nous:
        return None, None
    global auxiliary_is_nous
    auxiliary_is_nous = True
    logger.debug("Auxiliary client: Nous Portal")
    if nous.get("source") == "pool":
        model = "gemini-3-flash"
    else:
        model = _NOUS_MODEL
    # Free-tier users can't use paid auxiliary models — use the free
    # models instead: mimo-v2-omni for vision, mimo-v2-pro for text tasks.
    try:
        from her_cli.models import check_nous_free_tier
        if check_nous_free_tier():
            model = _NOUS_FREE_TIER_VISION_MODEL if vision else _NOUS_FREE_TIER_AUX_MODEL
            logger.debug("Free-tier Nous account — using %s for auxiliary/%s",
                         model, "vision" if vision else "text")
    except Exception:
        pass
    return (
        OpenAI(
            api_key=_nous_api_key(nous),
            base_url=str(nous.get("inference_base_url") or _nous_base_url()).rstrip("/"),
        ),
        model,
    )


def _read_main_model() -> str:
    """Read the user's configured main model from cli-config.yaml.

    cli-config.yaml model.default is the single source of truth for the active
    model. Environment variables are no longer consulted.
    """
    try:
        from her_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, str) and model_cfg.strip():
            return model_cfg.strip()
        if isinstance(model_cfg, dict):
            default = model_cfg.get("default", "")
            if isinstance(default, str) and default.strip():
                return default.strip()
    except Exception:
        pass
    return ""


def _read_main_provider() -> str:
    """Read the user's configured main provider from cli-config.yaml.

    Returns the lowercase provider id (e.g. "alibaba", "openrouter") or ""
    if not configured.
    """
    try:
        from her_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, dict):
            provider = model_cfg.get("provider", "")
            if isinstance(provider, str) and provider.strip():
                return provider.strip().lower()
    except Exception:
        pass
    return ""


def _resolve_custom_runtime() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve the active custom/main endpoint the same way the main CLI does.

    This covers both env-driven OPENAI_BASE_URL setups and config-saved custom
    endpoints where the base URL lives in cli-config.yaml instead of the live
    environment.
    """
    try:
        from her_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested="custom")
    except Exception as exc:
        logger.debug("Auxiliary client: custom runtime resolution failed: %s", exc)
        runtime = None

    if not isinstance(runtime, dict):
        openai_base = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_base:
            return None, None, None
        runtime = {
            "base_url": openai_base,
            "api_key": openai_key,
        }

    custom_base = runtime.get("base_url")
    custom_key = runtime.get("api_key")
    custom_mode = runtime.get("api_mode")
    if not isinstance(custom_base, str) or not custom_base.strip():
        return None, None, None

    custom_base = custom_base.strip().rstrip("/")
    if "openrouter.ai" in custom_base.lower():
        # requested='custom' falls back to OpenRouter when no custom endpoint is
        # configured. Treat that as "no custom endpoint" for auxiliary routing.
        return None, None, None

    # Local servers (Ollama, llama.cpp, vLLM, LM Studio) don't require auth.
    # Use a placeholder key — the OpenAI SDK requires a non-empty string but
    # local servers ignore the Authorization header.  Same fix as cli.py
    # _ensure_runtime_credentials() (PR #2556).
    if not isinstance(custom_key, str) or not custom_key.strip():
        custom_key = "no-key-required"

    if not isinstance(custom_mode, str) or not custom_mode.strip():
        custom_mode = None

    return custom_base, custom_key.strip(), custom_mode


def _current_custom_base_url() -> str:
    custom_base, _, _ = _resolve_custom_runtime()
    return custom_base or ""


def _validate_proxy_env_urls() -> None:
    """Fail fast with a clear error when proxy env vars have malformed URLs.

    Common cause: shell config (e.g. .zshrc) with a typo like
    ``export HTTP_PROXY=http://127.0.0.1:6153export NEXT_VAR=...``
    which concatenates 'export' into the port number.  Without this
    check the OpenAI/httpx client raises a cryptic ``Invalid port``
    error that doesn't name the offending env var.
    """
    from urllib.parse import urlparse

    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        value = str(os.environ.get(key) or "").strip()
        if not value:
            continue
        try:
            parsed = urlparse(value)
            if parsed.scheme:
                _ = parsed.port          # raises ValueError for e.g. '6153export'
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed proxy environment variable {key}={value!r}. "
                "Fix or unset your proxy settings and try again."
            ) from exc


def _validate_base_url(base_url: str) -> None:
    """Reject obviously broken custom endpoint URLs before they reach httpx."""
    from urllib.parse import urlparse

    candidate = str(base_url or "").strip()
    if not candidate or candidate.startswith("acp://"):
        return
    try:
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            _ = parsed.port              # raises ValueError for malformed ports
    except ValueError as exc:
        raise RuntimeError(
            f"Malformed custom endpoint URL: {candidate!r}. "
            "Run `her_agent setup` or `her_agent model` and enter a valid http(s) base URL."
        ) from exc


def _try_custom_endpoint() -> Tuple[Optional[Any], Optional[str]]:
    runtime = _resolve_custom_runtime()
    if len(runtime) == 2:
        custom_base, custom_key = runtime
        custom_mode = None
    else:
        custom_base, custom_key, custom_mode = runtime
    if not custom_base or not custom_key:
        return None, None
    if custom_base.lower().startswith(_CODEX_AUX_BASE_URL.lower()):
        return None, None
    model = _read_main_model() or "gpt-4o-mini"
    logger.debug("Auxiliary client: custom endpoint (%s, api_mode=%s)", model, custom_mode or "chat_completions")
    if custom_mode == "codex_responses":
        real_client = OpenAI(api_key=custom_key, base_url=custom_base)
        return CodexAuxiliaryClient(real_client, model), model
    if custom_mode == "anthropic_messages":
        # Third-party Anthropic-compatible gateway (MiniMax, Zhipu GLM,
        # LiteLLM proxies, etc.).  Must NEVER be treated as OAuth —
        # Anthropic OAuth claims only apply to api.anthropic.com.
        try:
            from agent.anthropic_adapter import build_anthropic_client
            real_client = build_anthropic_client(custom_key, custom_base)
        except ImportError:
            logger.warning(
                "Custom endpoint declares api_mode=anthropic_messages but the "
                "anthropic SDK is not installed — falling back to OpenAI-wire."
            )
            return OpenAI(api_key=custom_key, base_url=custom_base), model
        return (
            AnthropicAuxiliaryClient(real_client, model, custom_key, custom_base, is_oauth=False),
            model,
        )
    return OpenAI(api_key=custom_key, base_url=custom_base), model


def _try_codex() -> Tuple[Optional[Any], Optional[str]]:
    pool_present, entry = _select_pool_entry("openai-codex")
    if pool_present:
        codex_token = _pool_runtime_api_key(entry)
        if codex_token:
            base_url = _pool_runtime_base_url(entry, _CODEX_AUX_BASE_URL) or _CODEX_AUX_BASE_URL
        else:
            codex_token = _read_codex_access_token()
            if not codex_token:
                return None, None
            base_url = _CODEX_AUX_BASE_URL
    else:
        codex_token = _read_codex_access_token()
        if not codex_token:
            return None, None
        base_url = _CODEX_AUX_BASE_URL
    logger.debug("Auxiliary client: Codex OAuth (%s via Responses API)", _CODEX_AUX_MODEL)
    real_client = OpenAI(
        api_key=codex_token,
        base_url=base_url,
        default_headers=_codex_cloudflare_headers(codex_token),
    )
    return CodexAuxiliaryClient(real_client, _CODEX_AUX_MODEL), _CODEX_AUX_MODEL


def _try_anthropic() -> Tuple[Optional[Any], Optional[str]]:
    try:
        from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
    except ImportError:
        return None, None

    pool_present, entry = _select_pool_entry("anthropic")
    if pool_present:
        if entry is None:
            return None, None
        token = _pool_runtime_api_key(entry)
    else:
        entry = None
        token = resolve_anthropic_token()
    if not token:
        return None, None

    # Allow base URL override from cli-config.yaml model.base_url, but only
    # when the configured provider is anthropic — otherwise a non-Anthropic
    # base_url (e.g. Codex endpoint) would leak into Anthropic requests.
    base_url = _pool_runtime_base_url(entry, _ANTHROPIC_DEFAULT_BASE_URL) if pool_present else _ANTHROPIC_DEFAULT_BASE_URL
    try:
        from her_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model")
        if isinstance(model_cfg, dict):
            cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
            if cfg_provider == "anthropic":
                cfg_base_url = (model_cfg.get("base_url") or "").strip().rstrip("/")
                if cfg_base_url:
                    base_url = cfg_base_url
    except Exception:
        pass

    from agent.anthropic_adapter import _is_oauth_token
    is_oauth = _is_oauth_token(token)
    model = _API_KEY_PROVIDER_AUX_MODELS.get("anthropic", "claude-haiku-4-5-20251001")
    logger.debug("Auxiliary client: Anthropic native (%s) at %s (oauth=%s)", model, base_url, is_oauth)
    try:
        real_client = build_anthropic_client(token, base_url)
    except ImportError:
        # The anthropic_adapter module imports fine but the SDK itself is
        # missing — build_anthropic_client raises ImportError at call time
        # when _anthropic_sdk is None.  Treat as unavailable.
        return None, None
    return AnthropicAuxiliaryClient(real_client, model, token, base_url, is_oauth=is_oauth), model


_AUTO_PROVIDER_LABELS = {
    "_try_openrouter": "openrouter",
    "_try_nous": "nous",
    "_try_custom_endpoint": "local/custom",
    "_try_codex": "openai-codex",
    "_resolve_api_key_provider": "api-key",
}

_MAIN_RUNTIME_FIELDS = ("provider", "model", "base_url", "api_key", "api_mode")


def _normalize_main_runtime(main_runtime: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Return a sanitized copy of a live main-runtime override."""
    if not isinstance(main_runtime, dict):
        return {}
    normalized: Dict[str, str] = {}
    for field in _MAIN_RUNTIME_FIELDS:
        value = main_runtime.get(field)
        if isinstance(value, str) and value.strip():
            normalized[field] = value.strip()
    provider = normalized.get("provider")
    if provider:
        normalized["provider"] = provider.lower()
    return normalized


def _get_provider_chain() -> List[tuple]:
    """Return the ordered provider detection chain.

    Built at call time (not module level) so that test patches
    on the ``_try_*`` functions are picked up correctly.
    """
    return [
        ("openrouter", _try_openrouter),
        ("nous", _try_nous),
        ("local/custom", _try_custom_endpoint),
        ("openai-codex", _try_codex),
        ("api-key", _resolve_api_key_provider),
    ]


def _is_payment_error(exc: Exception) -> bool:
    """Detect payment/credit/quota exhaustion errors.

    Returns True for HTTP 402 (Payment Required) and for 429/other errors
    whose message indicates billing exhaustion rather than rate limiting.
    """
    status = getattr(exc, "status_code", None)
    if status == 402:
        return True
    err_lower = str(exc).lower()
    # OpenRouter and other providers include "credits" or "afford" in 402 bodies,
    # but sometimes wrap them in 429 or other codes.
    if status in (402, 429, None):
        if any(kw in err_lower for kw in ("credits", "insufficient funds",
                                           "can only afford", "billing",
                                           "payment required")):
            return True
    return False


def _is_connection_error(exc: Exception) -> bool:
    """Detect connection/network errors that warrant provider fallback.

    Returns True for errors indicating the provider endpoint is unreachable
    (DNS failure, connection refused, TLS errors, timeouts).  These are
    distinct from API errors (4xx/5xx) which indicate the provider IS
    reachable but returned an error.
    """
    from openai import APIConnectionError, APITimeoutError

    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    # urllib3 / httpx / httpcore connection errors
    err_type = type(exc).__name__
    if any(kw in err_type for kw in ("Connection", "Timeout", "DNS", "SSL")):
        return True
    err_lower = str(exc).lower()
    if any(kw in err_lower for kw in (
        "connection refused", "name or service not known",
        "no route to host", "network is unreachable",
        "timed out", "connection reset",
    )):
        return True
    return False


def _try_payment_fallback(
    failed_provider: str,
    task: str = None,
    reason: str = "payment error",
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try alternative providers after a payment/credit or connection error.

    Iterates the standard auto-detection chain, skipping the provider that
    failed.

    Returns:
        (client, model, provider_label) or (None, None, "") if no fallback.
    """
    # Normalise the failed provider label for matching.
    skip = failed_provider.lower().strip()
    # Also skip Step-1 main-provider path if it maps to the same backend.
    # (e.g. main_provider="openrouter" → skip "openrouter" in chain)
    main_provider = _read_main_provider()
    skip_labels = {skip}
    if main_provider and main_provider.lower() in skip:
        skip_labels.add(main_provider.lower())
    # Map common resolved_provider values back to chain labels.
    _alias_to_label = {"openrouter": "openrouter", "nous": "nous",
                       "openai-codex": "openai-codex", "codex": "openai-codex",
                       "custom": "local/custom", "local/custom": "local/custom"}
    skip_chain_labels = {_alias_to_label.get(s, s) for s in skip_labels}

    tried = []
    for label, try_fn in _get_provider_chain():
        if label in skip_chain_labels:
            continue
        client, model = try_fn()
        if client is not None:
            logger.info(
                "Auxiliary %s: %s on %s — falling back to %s (%s)",
                task or "call", reason, failed_provider, label, model or "default",
            )
            return client, model, label
        tried.append(label)

    logger.warning(
        "Auxiliary %s: %s on %s and no fallback available (tried: %s)",
        task or "call", reason, failed_provider, ", ".join(tried),
    )
    return None, None, ""


def _resolve_auto(main_runtime: Optional[Dict[str, Any]] = None) -> Tuple[Optional[OpenAI], Optional[str]]:
    """Full auto-detection chain.

    Priority:
      1. User's main provider + main model, regardless of provider type.
         This means auxiliary tasks (compression, vision, web extraction,
         session search, etc.) use the same model the user configured for
         chat.  Users on OpenRouter/Nous get their chosen chat model; users
         on DeepSeek/ZAI/Alibaba get theirs; etc.  Running aux tasks on the
         user's picked model keeps behavior predictable — no surprise
         switches to a cheap fallback model for side tasks.
      2. OpenRouter → Nous → custom → Codex → API-key providers (fallback
         chain, only used when the main provider has no working client).
    """
    global auxiliary_is_nous, _stale_base_url_warned
    auxiliary_is_nous = False  # Reset — _try_nous() will set True if it wins
    runtime = _normalize_main_runtime(main_runtime)
    runtime_provider = runtime.get("provider", "")
    runtime_model = runtime.get("model", "")
    runtime_base_url = runtime.get("base_url", "")
    runtime_api_key = runtime.get("api_key", "")
    runtime_api_mode = runtime.get("api_mode", "")

    # ── Warn once if OPENAI_BASE_URL is set but cli-config.yaml uses a named
    #    provider (not 'custom').  This catches the common "env poisoning"
    #    scenario where a user switches providers via `her_agent model` but the
    #    old OPENAI_BASE_URL lingers in ~/.her_agent/.env. ──
    if not _stale_base_url_warned:
        _env_base = os.getenv("OPENAI_BASE_URL", "").strip()
        _cfg_provider = runtime_provider or _read_main_provider()
        if (_env_base and _cfg_provider
                and _cfg_provider != "custom"
                and not _cfg_provider.startswith("custom:")):
            logger.warning(
                "OPENAI_BASE_URL is set (%s) but model.provider is '%s'. "
                "Auxiliary clients may route to the wrong endpoint. "
                "Run: her_agent model to reconfigure, or remove "
                "OPENAI_BASE_URL from ~/.her_agent/.env",
                _env_base, _cfg_provider,
            )
            _stale_base_url_warned = True

    # ── Step 1: main provider + main model → use them directly ──
    #
    # This is the primary aux backend for every user.  "auto" means
    # "use my main chat model for side tasks as well" — including users
    # on aggregators (OpenRouter, Nous) who previously got routed to a
    # cheap provider-side default.  Explicit per-task overrides set via
    # cli-config.yaml (auxiliary.<task>.provider) still win over this.
    main_provider = runtime_provider or _read_main_provider()
    main_model = runtime_model or _read_main_model()
    if (main_provider and main_model
            and main_provider not in ("auto", "")):
        resolved_provider = main_provider
        explicit_base_url = None
        explicit_api_key = None
        if runtime_base_url and (main_provider == "custom" or main_provider.startswith("custom:")):
            resolved_provider = "custom"
            explicit_base_url = runtime_base_url
            explicit_api_key = runtime_api_key or None
        client, resolved = resolve_provider_client(
            resolved_provider,
            main_model,
            explicit_base_url=explicit_base_url,
            explicit_api_key=explicit_api_key,
            api_mode=runtime_api_mode or None,
        )
        if client is not None:
            logger.info("Auxiliary auto-detect: using main provider %s (%s)",
                        main_provider, resolved or main_model)
            return client, resolved or main_model

    # ── Step 2: aggregator / fallback chain ──────────────────────────────
    tried = []
    for label, try_fn in _get_provider_chain():
        client, model = try_fn()
        if client is not None:
            if tried:
                logger.info("Auxiliary auto-detect: using %s (%s) — skipped: %s",
                            label, model or "default", ", ".join(tried))
            else:
                logger.info("Auxiliary auto-detect: using %s (%s)", label, model or "default")
            return client, model
        tried.append(label)
    logger.warning("Auxiliary auto-detect: no provider available (tried: %s). "
                   "Compression, summarization, and memory flush will not work. "
                   "Set OPENROUTER_API_KEY or configure a local model in cli-config.yaml.",
                   ", ".join(tried))
    return None, None


# ── Centralized Provider Router ─────────────────────────────────────────────
#
# resolve_provider_client() is the single entry point for creating a properly
# configured client given a (provider, model) pair.  It handles auth lookup,
# base URL resolution, provider-specific headers, and API format differences
# (Chat Completions vs Responses API for Codex).
#
# All auxiliary consumer code should go through this or the public helpers
# below — never look up auth env vars ad-hoc.


def _to_async_client(sync_client, model: str):
    """Convert a sync client to its async counterpart, preserving Codex routing."""
    from openai import AsyncOpenAI

    if isinstance(sync_client, CodexAuxiliaryClient):
        return AsyncCodexAuxiliaryClient(sync_client), model
    if isinstance(sync_client, AnthropicAuxiliaryClient):
        return AsyncAnthropicAuxiliaryClient(sync_client), model
    try:
        from agent.gemini_native_adapter import GeminiNativeClient, AsyncGeminiNativeClient

        if isinstance(sync_client, GeminiNativeClient):
            return AsyncGeminiNativeClient(sync_client), model
    except ImportError:
        pass
    try:
        from agent.copilot_acp_client import CopilotACPClient
        if isinstance(sync_client, CopilotACPClient):
            return sync_client, model
    except ImportError:
        pass

    async_kwargs = {
        "api_key": sync_client.api_key,
        "base_url": str(sync_client.base_url),
    }
    base_lower = str(sync_client.base_url).lower()
    if "openrouter" in base_lower:
        async_kwargs["default_headers"] = dict(_OR_HEADERS)
    elif "api.githubcopilot.com" in base_lower:
        from her_cli.models import copilot_default_headers

        async_kwargs["default_headers"] = copilot_default_headers()
    elif "api.kimi.com" in base_lower:
        async_kwargs["default_headers"] = {"User-Agent": "KimiCLI/1.30.0"}
    return AsyncOpenAI(**async_kwargs), model


def _normalize_resolved_model(model_name: Optional[str], provider: str) -> Optional[str]:
    """Normalize a resolved model for the provider that will receive it."""
    if not model_name:
        return model_name
    try:
        from her_cli.model_normalize import normalize_model_for_provider

        return normalize_model_for_provider(model_name, provider)
    except Exception:
        return model_name


def resolve_provider_client(
    provider: str,
    model: str = None,
    async_mode: bool = False,
    raw_codex: bool = False,
    explicit_base_url: str = None,
    explicit_api_key: str = None,
    api_mode: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Central router: given a provider name and optional model, return a
    configured client with the correct auth, base URL, and API format.

    The returned client always exposes ``.chat.completions.create()`` — for
    Codex/Responses API providers, an adapter handles the translation
    transparently.

    Args:
        provider: Provider identifier.  One of:
            "openrouter", "nous", "openai-codex" (or "codex"),
            "zai", "kimi-coding", "minimax", "minimax-cn",
            "custom" (OPENAI_BASE_URL + OPENAI_API_KEY),
            "auto" (full auto-detection chain).
        model: Model slug override.  If None, uses the provider's default
               auxiliary model.
        async_mode: If True, return an async-compatible client.
        raw_codex: If True, return a raw OpenAI client for Codex providers
            instead of wrapping in CodexAuxiliaryClient.  Use this when
            the caller needs direct access to responses.stream() (e.g.,
            the main agent loop).
        explicit_base_url: Optional direct OpenAI-compatible endpoint.
        explicit_api_key: Optional API key paired with explicit_base_url.
        api_mode: API mode override.  One of "chat_completions",
            "codex_responses", or None (auto-detect).  When set to
            "codex_responses", the client is wrapped in
            CodexAuxiliaryClient to route through the Responses API.

    Returns:
        (client, resolved_model) or (None, None) if auth is unavailable.
    """
    _validate_proxy_env_urls()
    # Normalise aliases
    provider = _normalize_aux_provider(provider)

    def _needs_codex_wrap(client_obj, base_url_str: str, model_str: str) -> bool:
        """Decide if a plain OpenAI client should be wrapped for Responses API.

        Returns True when api_mode is explicitly "codex_responses", or when
        auto-detection (api.openai.com + codex-family model) suggests it.
        Already-wrapped clients (CodexAuxiliaryClient) are skipped.
        """
        if isinstance(client_obj, CodexAuxiliaryClient):
            return False
        if raw_codex:
            return False
        if api_mode == "codex_responses":
            return True
        # Auto-detect: api.openai.com + codex model name pattern
        if api_mode and api_mode != "codex_responses":
            return False  # explicit non-codex mode
        normalized_base = (base_url_str or "").strip().lower()
        if "api.openai.com" in normalized_base and "openrouter" not in normalized_base:
            model_lower = (model_str or "").lower()
            if "codex" in model_lower:
                return True
        return False

    def _wrap_if_needed(client_obj, final_model_str: str, base_url_str: str = ""):
        """Wrap a plain OpenAI client in CodexAuxiliaryClient if Responses API is needed."""
        if _needs_codex_wrap(client_obj, base_url_str, final_model_str):
            logger.debug(
                "resolve_provider_client: wrapping client in CodexAuxiliaryClient "
                "(api_mode=%s, model=%s, base_url=%s)",
                api_mode or "auto-detected", final_model_str,
                base_url_str[:60] if base_url_str else "")
            return CodexAuxiliaryClient(client_obj, final_model_str)
        return client_obj

    # ── Auto: try all providers in priority order ────────────────────
    if provider == "auto":
        client, resolved = _resolve_auto(main_runtime=main_runtime)
        if client is None:
            return None, None
        # When auto-detection lands on a non-OpenRouter provider (e.g. a
        # local server), an OpenRouter-formatted model override like
        # "google/gemini-3-flash-preview" won't work.  Drop it and use
        # the provider's own default model instead.
        if model and "/" in model and resolved and "/" not in resolved:
            logger.debug(
                "Dropping OpenRouter-format model %r for non-OpenRouter "
                "auxiliary provider (using %r instead)", model, resolved)
            model = None
        final_model = model or resolved
        return (_to_async_client(client, final_model) if async_mode
                else (client, final_model))

    # ── OpenRouter ───────────────────────────────────────────────────
    if provider == "openrouter":
        client, default = _try_openrouter()
        if client is None:
            logger.warning("resolve_provider_client: openrouter requested "
                           "but OPENROUTER_API_KEY not set")
            return None, None
        final_model = _normalize_resolved_model(model or default, provider)
        return (_to_async_client(client, final_model) if async_mode
                else (client, final_model))

    # ── Nous Portal (OAuth) ──────────────────────────────────────────
    if provider == "nous":
        client, default = _try_nous()
        if client is None:
            logger.warning("resolve_provider_client: nous requested "
                           "but Nous Portal not configured (run: her_agent auth)")
            return None, None
        final_model = _normalize_resolved_model(model or default, provider)
        return (_to_async_client(client, final_model) if async_mode
                else (client, final_model))

    # ── OpenAI Codex (OAuth → Responses API) ─────────────────────────
    if provider == "openai-codex":
        if raw_codex:
            # Return the raw OpenAI client for callers that need direct
            # access to responses.stream() (e.g., the main agent loop).
            codex_token = _read_codex_access_token()
            if not codex_token:
                logger.warning("resolve_provider_client: openai-codex requested "
                               "but no Codex OAuth token found (run: her_agent model)")
                return None, None
            final_model = _normalize_resolved_model(model or _CODEX_AUX_MODEL, provider)
            raw_client = OpenAI(
                api_key=codex_token,
                base_url=_CODEX_AUX_BASE_URL,
                default_headers=_codex_cloudflare_headers(codex_token),
            )
            return (raw_client, final_model)
        # Standard path: wrap in CodexAuxiliaryClient adapter
        client, default = _try_codex()
        if client is None:
            logger.warning("resolve_provider_client: openai-codex requested "
                           "but no Codex OAuth token found (run: her_agent model)")
            return None, None
        final_model = _normalize_resolved_model(model or default, provider)
        return (_to_async_client(client, final_model) if async_mode
                else (client, final_model))

    # ── Custom endpoint (OPENAI_BASE_URL + OPENAI_API_KEY) ───────────
    if provider == "custom":
        if explicit_base_url:
            custom_base = explicit_base_url.strip()
            custom_key = (
                (explicit_api_key or "").strip()
                or os.getenv("OPENAI_API_KEY", "").strip()
                or "no-key-required"  # local servers don't need auth
            )
            if not custom_base:
                logger.warning(
                    "resolve_provider_client: explicit custom endpoint requested "
                    "but base_url is empty"
                )
                return None, None
            final_model = _normalize_resolved_model(
                model or _read_main_model() or "gpt-4o-mini",
                provider,
            )
            extra = {}
            if "api.kimi.com" in custom_base.lower():
                extra["default_headers"] = {"User-Agent": "KimiCLI/1.30.0"}
            elif "api.githubcopilot.com" in custom_base.lower():
                from her_cli.models import copilot_default_headers
                extra["default_headers"] = copilot_default_headers()
            client = OpenAI(api_key=custom_key, base_url=custom_base, **extra)
            client = _wrap_if_needed(client, final_model, custom_base)
            return (_to_async_client(client, final_model) if async_mode
                    else (client, final_model))
        # Try custom first, then codex, then API-key providers
        for try_fn in (_try_custom_endpoint, _try_codex,
                       _resolve_api_key_provider):
            client, default = try_fn()
            if client is not None:
                final_model = _normalize_resolved_model(model or default, provider)
                _cbase = str(getattr(client, "base_url", "") or "")
                client = _wrap_if_needed(client, final_model, _cbase)
                return (_to_async_client(client, final_model) if async_mode
                        else (client, final_model))
        logger.warning("resolve_provider_client: custom/main requested "
                       "but no endpoint credentials found")
        return None, None

    # ── Named custom providers (cli-config.yaml custom_providers list) ───
    try:
        from her_cli.runtime_provider import _get_named_custom_provider
        custom_entry = _get_named_custom_provider(provider)
        if custom_entry:
            custom_base = custom_entry.get("base_url", "").strip()
            custom_key = custom_entry.get("api_key", "").strip()
            custom_key_env = custom_entry.get("key_env", "").strip()
            if not custom_key and custom_key_env:
                custom_key = os.getenv(custom_key_env, "").strip()
            custom_key = custom_key or "no-key-required"
            if custom_base:
                final_model = _normalize_resolved_model(
                    model or custom_entry.get("model") or _read_main_model() or "gpt-4o-mini",
                    provider,
                )
                client = OpenAI(api_key=custom_key, base_url=custom_base)
                client = _wrap_if_needed(client, final_model, custom_base)
                logger.debug(
                    "resolve_provider_client: named custom provider %r (%s)",
                    provider, final_model)
                return (_to_async_client(client, final_model) if async_mode
                        else (client, final_model))
            logger.warning(
                "resolve_provider_client: named custom provider %r has no base_url",
                provider)
            return None, None
    except ImportError:
        pass

    # ── API-key providers from PROVIDER_REGISTRY ─────────────────────
    try:
        from her_cli.auth import (
            PROVIDER_REGISTRY,
            resolve_api_key_provider_credentials,
            resolve_external_process_provider_credentials,
        )
    except ImportError:
        logger.debug("her_cli.auth not available for provider %s", provider)
        return None, None

    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig is None:
        logger.warning("resolve_provider_client: unknown provider %r", provider)
        return None, None

    if pconfig.auth_type == "api_key":
        if provider == "anthropic":
            client, default_model = _try_anthropic()
            if client is None:
                logger.warning("resolve_provider_client: anthropic requested but no Anthropic credentials found")
                return None, None
            final_model = _normalize_resolved_model(model or default_model, provider)
            return (_to_async_client(client, final_model) if async_mode else (client, final_model))

        creds = resolve_api_key_provider_credentials(provider)
        api_key = str(creds.get("api_key", "")).strip()
        if not api_key:
            tried_sources = list(pconfig.api_key_env_vars)
            if provider == "copilot":
                tried_sources.append("gh auth token")
            logger.debug("resolve_provider_client: provider %s has no API "
                         "key configured (tried: %s)",
                         provider, ", ".join(tried_sources))
            return None, None

        base_url = _to_openai_base_url(
            str(creds.get("base_url", "")).strip().rstrip("/") or pconfig.inference_base_url
        )

        default_model = _API_KEY_PROVIDER_AUX_MODELS.get(provider, "")
        final_model = _normalize_resolved_model(model or default_model, provider)

        if provider == "gemini":
            from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

            if is_native_gemini_base_url(base_url):
                client = GeminiNativeClient(api_key=api_key, base_url=base_url)
                logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
                return (_to_async_client(client, final_model) if async_mode
                        else (client, final_model))

        # Provider-specific headers
        headers = {}
        if "api.kimi.com" in base_url.lower():
            headers["User-Agent"] = "KimiCLI/1.30.0"
        elif "api.githubcopilot.com" in base_url.lower():
            from her_cli.models import copilot_default_headers

            headers.update(copilot_default_headers())
        client = OpenAI(api_key=api_key, base_url=base_url,
                        **({"default_headers": headers} if headers else {}))

        # Copilot GPT-5+ models (except gpt-5-mini) require the Responses
        # API — they are not accessible via /chat/completions.  Wrap the
        # plain client in CodexAuxiliaryClient so call_llm() transparently
        # routes through responses.stream().
        if provider == "copilot" and final_model and not raw_codex:
            try:
                from her_cli.models import _should_use_copilot_responses_api
                if _should_use_copilot_responses_api(final_model):
                    logger.debug(
                        "resolve_provider_client: copilot model %s needs "
                        "Responses API — wrapping with CodexAuxiliaryClient",
                        final_model)
                    client = CodexAuxiliaryClient(client, final_model)
            except ImportError:
                pass

        # Honor api_mode for any API-key provider (e.g. direct OpenAI with
        # codex-family models).  The copilot-specific wrapping above handles
        # copilot; this covers the general case (#6800).
        client = _wrap_if_needed(client, final_model, base_url)

        logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
        return (_to_async_client(client, final_model) if async_mode
                else (client, final_model))

    if pconfig.auth_type == "external_process":
        creds = resolve_external_process_provider_credentials(provider)
        final_model = _normalize_resolved_model(model or _read_main_model(), provider)
        if provider == "copilot-acp":
            api_key = str(creds.get("api_key", "")).strip()
            base_url = str(creds.get("base_url", "")).strip()
            command = str(creds.get("command", "")).strip() or None
            args = list(creds.get("args") or [])
            if not final_model:
                logger.warning(
                    "resolve_provider_client: copilot-acp requested but no model "
                    "was provided or configured"
                )
                return None, None
            if not api_key or not base_url:
                logger.warning(
                    "resolve_provider_client: copilot-acp requested but external "
                    "process credentials are incomplete"
                )
                return None, None
            from agent.copilot_acp_client import CopilotACPClient

            client = CopilotACPClient(
                api_key=api_key,
                base_url=base_url,
                command=command,
                args=args,
            )
            logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
            return (_to_async_client(client, final_model) if async_mode
                    else (client, final_model))
        logger.warning("resolve_provider_client: external-process provider %s not "
                       "directly supported", provider)
        return None, None

    elif pconfig.auth_type in ("oauth_device_code", "oauth_external"):
        # OAuth providers — route through their specific try functions
        if provider == "nous":
            return resolve_provider_client("nous", model, async_mode)
        if provider == "openai-codex":
            return resolve_provider_client("openai-codex", model, async_mode)
        # Other OAuth providers not directly supported
        logger.warning("resolve_provider_client: OAuth provider %s not "
                       "directly supported, try 'auto'", provider)
        return None, None

    logger.warning("resolve_provider_client: unhandled auth_type %s for %s",
                   pconfig.auth_type, provider)
    return None, None


# ── Public API ──────────────────────────────────────────────────────────────

def get_text_auxiliary_client(
    task: str = "",
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, default_model_slug) for text-only auxiliary tasks.

    Args:
        task: Optional task name ("compression", "web_extract") to check
              for a task-specific provider override.

    Callers may override the returned model via cli-config.yaml
    (e.g. auxiliary.compression.model, auxiliary.web_extract.model).
    """
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(task or None)
    return resolve_provider_client(
        provider,
        model=model,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
    )


def get_async_text_auxiliary_client(task: str = "", *, main_runtime: Optional[Dict[str, Any]] = None):
    """Return (async_client, model_slug) for async consumers.

    For standard providers returns (AsyncOpenAI, model). For Codex returns
    (AsyncCodexAuxiliaryClient, model) which wraps the Responses API.
    Returns (None, None) when no provider is available.
    """
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(task or None)
    return resolve_provider_client(
        provider,
        model=model,
        async_mode=True,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
    )


_VISION_AUTO_PROVIDER_ORDER = (
    "openrouter",
    "nous",
)


def _normalize_vision_provider(provider: Optional[str]) -> str:
    return _normalize_aux_provider(provider)


def _resolve_strict_vision_backend(provider: str) -> Tuple[Optional[Any], Optional[str]]:
    provider = _normalize_vision_provider(provider)
    if provider == "openrouter":
        return _try_openrouter()
    if provider == "nous":
        return _try_nous(vision=True)
    if provider == "openai-codex":
        return _try_codex()
    if provider == "anthropic":
        return _try_anthropic()
    if provider == "custom":
        return _try_custom_endpoint()
    return None, None


def _strict_vision_backend_available(provider: str) -> bool:
    return _resolve_strict_vision_backend(provider)[0] is not None


def get_available_vision_backends() -> List[str]:
    """Return the currently available vision backends in auto-selection order.

    Order: active provider → OpenRouter → Nous → stop.  This is the single
    source of truth for setup, tool gating, and runtime auto-routing of
    vision tasks.
    """
    available: List[str] = []
    # 1. Active provider — if the user configured a provider, try it first.
    main_provider = _read_main_provider()
    if main_provider and main_provider not in ("auto", ""):
        if main_provider in _VISION_AUTO_PROVIDER_ORDER:
            if _strict_vision_backend_available(main_provider):
                available.append(main_provider)
        else:
            client, _ = resolve_provider_client(main_provider, _read_main_model())
            if client is not None:
                available.append(main_provider)
    # 2. OpenRouter, 3. Nous — skip if already covered by main provider.
    for p in _VISION_AUTO_PROVIDER_ORDER:
        if p not in available and _strict_vision_backend_available(p):
            available.append(p)
    return available


def resolve_vision_provider_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    async_mode: bool = False,
) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """Resolve the client actually used for vision tasks.

    Direct endpoint overrides take precedence over provider selection. Explicit
    provider overrides still use the generic provider router for non-standard
    backends, so users can intentionally force experimental providers. Auto mode
    stays conservative and only tries vision backends known to work today.
    """
    requested, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        "vision", provider, model, base_url, api_key
    )
    requested = _normalize_vision_provider(requested)

    def _finalize(resolved_provider: str, sync_client: Any, default_model: Optional[str]):
        if sync_client is None:
            return resolved_provider, None, None
        final_model = resolved_model or default_model
        if async_mode:
            async_client, async_model = _to_async_client(sync_client, final_model)
            return resolved_provider, async_client, async_model
        return resolved_provider, sync_client, final_model

    if resolved_base_url:
        client, final_model = resolve_provider_client(
            "custom",
            model=resolved_model,
            async_mode=async_mode,
            explicit_base_url=resolved_base_url,
            explicit_api_key=resolved_api_key,
            api_mode=resolved_api_mode,
        )
        if client is None:
            return "custom", None, None
        return "custom", client, final_model

    if requested == "auto":
        # Vision auto-detection order:
        #   1. User's main provider + main model (including aggregators).
        #      _PROVIDER_VISION_MODELS provides per-provider vision model
        #      overrides when the provider has a dedicated multimodal model
        #      that differs from the chat model (e.g. xiaomi → mimo-v2-omni,
        #      zai → glm-5v-turbo).
        #   2. OpenRouter  (vision-capable aggregator fallback)
        #   3. Nous Portal (vision-capable aggregator fallback)
        #   4. Stop
        main_provider = _read_main_provider()
        main_model = _read_main_model()
        if main_provider and main_provider not in ("auto", ""):
            vision_model = _PROVIDER_VISION_MODELS.get(main_provider, main_model)
            rpc_client, rpc_model = resolve_provider_client(
                main_provider, vision_model,
                api_mode=resolved_api_mode)
            if rpc_client is not None:
                logger.info(
                    "Vision auto-detect: using main provider %s (%s)",
                    main_provider, rpc_model or vision_model,
                )
                return _finalize(
                    main_provider, rpc_client, rpc_model or vision_model)

        # Fall back through aggregators (uses their dedicated vision model,
        # not the user's main model) when main provider has no client.
        for candidate in _VISION_AUTO_PROVIDER_ORDER:
            if candidate == main_provider:
                continue  # already tried above
            sync_client, default_model = _resolve_strict_vision_backend(candidate)
            if sync_client is not None:
                return _finalize(candidate, sync_client, default_model)

        logger.debug("Auxiliary vision client: none available")
        return None, None, None

    if requested in _VISION_AUTO_PROVIDER_ORDER:
        sync_client, default_model = _resolve_strict_vision_backend(requested)
        return _finalize(requested, sync_client, default_model)

    client, final_model = _get_cached_client(requested, resolved_model, async_mode,
                                             api_mode=resolved_api_mode)
    if client is None:
        return requested, None, None
    return requested, client, final_model


def get_auxiliary_extra_body() -> dict:
    """Return extra_body kwargs for auxiliary API calls.
    
    Includes Nous Portal product tags when the auxiliary client is backed
    by Nous Portal. Returns empty dict otherwise.
    """
    return dict(NOUS_EXTRA_BODY) if auxiliary_is_nous else {}


def auxiliary_max_tokens_param(value: int) -> dict:
    """Return the correct max tokens kwarg for the auxiliary client's provider.
    
    OpenRouter and local models use 'max_tokens'. Direct OpenAI with newer
    models (gpt-4o, o-series, gpt-5+) requires 'max_completion_tokens'.
    The Codex adapter translates max_tokens internally, so we use max_tokens
    for it as well.
    """
    custom_base = _current_custom_base_url()
    or_key = os.getenv("OPENROUTER_API_KEY")
    # Only use max_completion_tokens for direct OpenAI custom endpoints
    if (not or_key
            and _read_nous_auth() is None
            and "api.openai.com" in custom_base.lower()):
        return {"max_completion_tokens": value}
    return {"max_tokens": value}


def _coerce_response_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _coerce_response_text(item)
            if text:
                parts.append(text)
        joined = "\n".join(parts).strip()
        return joined or None
    if isinstance(value, dict):
        for key in ("content", "text", "reasoning_content", "reasoning"):
            text = _coerce_response_text(value.get(key))
            if text:
                return text
        return None

    for attr in ("content", "text", "reasoning_content", "reasoning"):
        text = _coerce_response_text(getattr(value, attr, None))
        if text:
            return text
    return None


def extract_content_or_reasoning(response: Any) -> Optional[str]:
    """Extract assistant content or reasoning text from an OpenAI-style response."""
    direct = _coerce_response_text(getattr(response, "output_text", None))
    if direct:
        return direct

    for attr in ("content", "text", "reasoning_content", "reasoning"):
        text = _coerce_response_text(getattr(response, attr, None))
        if text:
            return text

    if isinstance(response, dict):
        for key in ("output_text", "content", "text", "reasoning_content", "reasoning"):
            text = _coerce_response_text(response.get(key))
            if text:
                return text
        choices = response.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                text = _coerce_response_text(choice)
                if text:
                    return text
                if isinstance(choice, dict):
                    message = choice.get("message")
                    text = _coerce_response_text(message)
                    if text:
                        return text
        output = response.get("output")
        if isinstance(output, list):
            for item in output:
                text = _coerce_response_text(item)
                if text:
                    return text
        return None

    choices = getattr(response, "choices", None)
    if isinstance(choices, list):
        for choice in choices:
            message = getattr(choice, "message", None)
            text = _coerce_response_text(message)
            if text:
                return text
            if isinstance(choice, dict):
                text = _coerce_response_text(choice.get("message"))
                if text:
                    return text
            text = _coerce_response_text(choice)
            if text:
                return text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            text = _coerce_response_text(item)
            if text:
                return text
    return None


async def async_call_llm(
    task: str = "",
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Resolve an auxiliary client and call it asynchronously."""
    resolved_provider, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        task or None,
        provider,
        model,
        base_url,
        api_key,
    )
    client, default_model = resolve_provider_client(
        resolved_provider,
        model=resolved_model,
        async_mode=True,
        explicit_base_url=resolved_base_url,
        explicit_api_key=resolved_api_key,
        api_mode=resolved_api_mode,
        main_runtime=main_runtime,
    )
    if client is None:
        raise RuntimeError("No auxiliary model available")

    call_kwargs = dict(kwargs)
    effective_model = call_kwargs.pop("model", None) or resolved_model or default_model
    if not effective_model:
        raise RuntimeError("No auxiliary model available")
    call_kwargs["model"] = effective_model

    if "max_tokens" in call_kwargs and "max_completion_tokens" not in call_kwargs:
        raw_max_tokens = call_kwargs.pop("max_tokens")
        if raw_max_tokens is not None:
            call_kwargs.update(auxiliary_max_tokens_param(int(raw_max_tokens)))

    merged_extra = get_auxiliary_extra_body()
    if extra_body:
        merged_extra.update(extra_body)
    existing_extra = call_kwargs.get("extra_body")
    if isinstance(existing_extra, dict):
        merged = dict(merged_extra)
        merged.update(existing_extra)
        merged_extra = merged
    if merged_extra:
        call_kwargs["extra_body"] = merged_extra

    return await client.chat.completions.create(**call_kwargs)


def call_llm(*args: Any, **kwargs: Any) -> Any:
    """Synchronous wrapper for auxiliary LLM calls."""
    coro = async_call_llm(*args, **kwargs)
    try:
        return asyncio.run(coro)
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()


# ── Centralized LLM Call API ────────────────────────────────────────────────
#
# call_llm() and async_call_llm() own the full request lifecycle:
#   1. Resolve provider + model from task config (or explicit args)
#   2. Get or create a cached client for that provider
#   3. Format request args for the provider + model (max_tokens handling, etc.)
#   4. Make the API call
#   5. Return the response
#
# Every auxiliary LLM consumer should use these instead of manually
# constructing clients and calling .chat.completions.create().

# Client cache: (provider, async_mode, base_url, api_key, api_mode, runtime_key) -> (client, default_model, loop)
# NOTE: loop identity is NOT part of the key.  On async cache hits we check
# whether the cached loop is the *current* loop; if not, the stale entry is
# replaced in-place.  This bounds cache growth to one entry per unique
# provider config rather than one per (config × event-loop), which previously
# caused unbounded fd accumulation in long-running gateway processes (#10200).
_client_cache: Dict[tuple, tuple] = {}
_client_cache_lock = threading.Lock()
_CLIENT_CACHE_MAX_SIZE = 64  # safety belt — evict oldest when exceeded


def neuter_async_httpx_del() -> None:
    """Monkey-patch ``AsyncHttpxClientWrapper.__del__`` to be a no-op.

    The OpenAI SDK's ``AsyncHttpxClientWrapper.__del__`` schedules
    ``self.aclose()`` via ``asyncio.get_running_loop().create_task()``.
    When an ``AsyncOpenAI`` client is garbage-collected while
    prompt_toolkit's event loop is running (the common CLI idle state),
    the ``aclose()`` task runs on prompt_toolkit's loop but the
    underlying TCP transport is bound to a *different* loop (the worker
    thread's loop that the client was originally created on).  If that
    loop is closed or its thread is dead, the transport's
    ``self._loop.call_soon()`` raises ``RuntimeError("Event loop is
    closed")``, which prompt_toolkit surfaces as "Unhandled exception
    in event loop ... Press ENTER to continue...".

    Neutering ``__del__`` is safe because:
    - Cached clients are explicitly cleaned via ``_force_close_async_httpx``
      on stale-loop detection and ``shutdown_cached_clients`` on exit.
    - Uncached clients' TCP connections are cleaned up by the OS when the
      process exits.
    - The OpenAI SDK itself marks this as a TODO (``# TODO(someday):
      support non asyncio runtimes here``).

    Call this once at CLI startup, before any ``AsyncOpenAI`` clients are
    created.
    """
    try:
        from openai._base_client import AsyncHttpxClientWrapper
        AsyncHttpxClientWrapper.__del__ = lambda self: None  # type: ignore[assignment]
    except (ImportError, AttributeError):
        pass  # Graceful degradation if the SDK changes its internals


def _force_close_async_httpx(client: Any) -> None:
    """Mark the httpx AsyncClient inside an AsyncOpenAI client as closed.

    This prevents ``AsyncHttpxClientWrapper.__del__`` from scheduling
    ``aclose()`` on a (potentially closed) event loop, which causes
    ``RuntimeError: Event loop is closed`` → prompt_toolkit's
    "Press ENTER to continue..." handler.

    We intentionally do NOT run the full async close path — the
    connections will be dropped by the OS when the process exits.
    """
    try:
        from httpx._client import ClientState
        inner = getattr(client, "_client", None)
        if inner is not None and not getattr(inner, "is_closed", True):
            inner._state = ClientState.CLOSED
    except Exception:
        pass


def shutdown_cached_clients() -> None:
    """Close all cached clients (sync and async) to prevent event-loop errors.

    Call this during CLI shutdown, *before* the event loop is closed, to
    avoid ``AsyncHttpxClientWrapper.__del__`` raising on a dead loop.
    """
    import inspect

    with _client_cache_lock:
        for key, entry in list(_client_cache.items()):
            client = entry[0]
            if client is None:
                continue
            # Mark any async httpx transport as closed first (prevents __del__
            # from scheduling aclose() on a dead event loop).
            _force_close_async_httpx(client)
            # Sync clients: close the httpx connection pool cleanly.
            # Async clients: skip — we already neutered __del__ above.
            try:
                close_fn = getattr(client, "close", None)
                if close_fn and not inspect.iscoroutinefunction(close_fn):
                    close_fn()
            except Exception:
                pass
        _client_cache.clear()


def cleanup_stale_async_clients() -> None:
    """Force-close cached async clients whose event loop is closed.

    Call this after each agent turn to proactively clean up stale clients
    before GC can trigger ``AsyncHttpxClientWrapper.__del__`` on them.
    This is defense-in-depth — the primary fix is ``neuter_async_httpx_del``
    which disables ``__del__`` entirely.
    """
    with _client_cache_lock:
        stale_keys = []
        for key, entry in _client_cache.items():
            client, _default, cached_loop = entry
            if cached_loop is not None and cached_loop.is_closed():
                _force_close_async_httpx(client)
                stale_keys.append(key)
        for key in stale_keys:
            del _client_cache[key]


def _is_openrouter_client(client: Any) -> bool:
    for obj in (client, getattr(client, "_client", None), getattr(client, "client", None)):
        if obj and "openrouter" in str(getattr(obj, "base_url", "") or "").lower():
            return True
    return False


def _compat_model(client: Any, model: Optional[str], cached_default: Optional[str]) -> Optional[str]:
    """Drop OpenRouter-format model slugs (with '/') for non-OpenRouter clients.

    Mirrors the guard in resolve_provider_client() which is skipped on cache hits.
    """
    if model and "/" in model and not _is_openrouter_client(client):
        return cached_default
    return model or cached_default


def _get_cached_client(
    provider: str,
    model: str = None,
    async_mode: bool = False,
    base_url: str = None,
    api_key: str = None,
    api_mode: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Get or create a cached client for the given provider.

    Async clients (AsyncOpenAI) use httpx.AsyncClient internally, which
    binds to the event loop that was current when the client was created.
    Using such a client on a *different* loop causes deadlocks or
    RuntimeError.  To prevent cross-loop issues, the cache validates on
    every async hit that the cached loop is the *current* and *open* loop.
    If the loop changed (e.g. a new gateway worker-thread loop), the stale
    entry is replaced in-place rather than creating an additional entry.

    This keeps cache size bounded to one entry per unique provider config,
    preventing the fd-exhaustion that previously occurred in long-running
    gateways where recycled worker threads created unbounded entries (#10200).
    """
    # Resolve the current event loop for async clients so we can validate
    # cached entries.  Loop identity is NOT in the cache key — instead we
    # check at hit time whether the cached loop is still current and open.
    # This prevents unbounded cache growth from recycled worker-thread loops
    # while still guaranteeing we never reuse a client on the wrong loop
    # (which causes deadlocks, see #2681).
    current_loop = None
    if async_mode:
        try:
            import asyncio as _aio
            current_loop = _aio.get_event_loop()
        except RuntimeError:
            pass
    runtime = _normalize_main_runtime(main_runtime)
    runtime_key = tuple(runtime.get(field, "") for field in _MAIN_RUNTIME_FIELDS) if provider == "auto" else ()
    cache_key = (provider, async_mode, base_url or "", api_key or "", api_mode or "", runtime_key)
    with _client_cache_lock:
        if cache_key in _client_cache:
            cached_client, cached_default, cached_loop = _client_cache[cache_key]
            if async_mode:
                # Validate: the cached client must be bound to the CURRENT,
                # OPEN loop.  If the loop changed or was closed, the httpx
                # transport inside is dead — force-close and replace.
                loop_ok = (
                    cached_loop is not None
                    and cached_loop is current_loop
                    and not cached_loop.is_closed()
                )
                if loop_ok:
                    effective = _compat_model(cached_client, model, cached_default)
                    return cached_client, effective
                # Stale — evict and fall through to create a new client.
                _force_close_async_httpx(cached_client)
                del _client_cache[cache_key]
            else:
                effective = _compat_model(cached_client, model, cached_default)
                return cached_client, effective
    # Build outside the lock
    client, default_model = resolve_provider_client(
        provider,
        model,
        async_mode,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
        main_runtime=runtime,
    )
    if client is not None:
        # For async clients, remember which loop they were created on so we
        # can detect stale entries later.
        bound_loop = current_loop
        with _client_cache_lock:
            if cache_key not in _client_cache:
                # Safety belt: if the cache has grown beyond the max, evict
                # the oldest entries (FIFO — dict preserves insertion order).
                while len(_client_cache) >= _CLIENT_CACHE_MAX_SIZE:
                    evict_key, evict_entry = next(iter(_client_cache.items()))
                    _force_close_async_httpx(evict_entry[0])
                    del _client_cache[evict_key]
                _client_cache[cache_key] = (client, default_model, bound_loop)
            else:
                client, default_model, _ = _client_cache[cache_key]
    return client, model or default_model


def _resolve_task_provider_model(
    task: str = None,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Determine provider + model for a call.

    Priority:
      1. Explicit provider/model/base_url/api_key args (always win)
      2. Config file (auxiliary.{task}.provider/model/base_url)
      3. "auto" (full auto-detection chain)

    Returns (provider, model, base_url, api_key, api_mode) where model may
    be None (use provider default). When base_url is set, provider is forced
    to "custom" and the task uses that direct endpoint. api_mode is one of
    "chat_completions", "codex_responses", or None (auto-detect).
    """
    cfg_provider = None
    cfg_model = None
    cfg_base_url = None
    cfg_api_key = None
    cfg_api_mode = None

    if task:
        task_config = _get_auxiliary_task_config(task)
        cfg_provider = str(task_config.get("provider", "")).strip() or None
        cfg_model = str(task_config.get("model", "")).strip() or None
        cfg_base_url = str(task_config.get("base_url", "")).strip() or None
        cfg_api_key = str(task_config.get("api_key", "")).strip() or None
        cfg_api_mode = str(task_config.get("api_mode", "")).strip() or None

    resolved_model = model or cfg_model
    resolved_api_mode = cfg_api_mode

    if base_url:
        return "custom", resolved_model, base_url, api_key, resolved_api_mode
    if provider:
        return provider, resolved_model, base_url, api_key, resolved_api_mode

    if task:
        # Config.yaml is the primary source for per-task overrides.
        if cfg_base_url:
            return "custom", resolved_model, cfg_base_url, cfg_api_key, resolved_api_mode
        if cfg_provider and cfg_provider != "auto":
            return cfg_provider, resolved_model, None, None, resolved_api_mode

        return "auto", resolved_model, None, None, resolved_api_mode

    return "auto", resolved_model, None, None, resolved_api_mode


_DEFAULT_AUX_TIMEOUT = 30.0


def _get_auxiliary_task_config(task: str) -> Dict[str, Any]:
    """Return the config dict for auxiliary.<task>, or {} when unavailable."""
    if not task:
        return {}
    try:
        from her_cli.config import load_config
        config = load_config()
    except ImportError:
        return {}
    aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
    task_config = aux.get(task, {}) if isinstance(aux, dict) else {}
    return task_config if isinstance(task_config, dict) else {}


def _get_task_timeout(task: str, default: float = _DEFAULT_AUX_TIMEOUT) -> float:
    """Read timeout from auxiliary.{task}.timeout in config, falling back to *default*."""
    if not task:
        return default
    task_config = _get_auxiliary_task_config(task)
    raw = task_config.get("timeout")
    if raw is not None:
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass
    return default


def _get_task_extra_body(task: str) -> Dict[str, Any]:
    """Read auxiliary.<task>.extra_body and return a shallow copy when valid."""
    task_config = _get_auxiliary_task_config(task)
    raw = task_config.get("extra_body")
    if isinstance(raw, dict):
        return dict(raw)
    return {}

