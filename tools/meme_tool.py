"""Meme routing tool.

This module keeps the public ``meme_search`` tool name stable while moving the
implementation to a dedicated hot/cold router:
  - cloud embedding via SiliconFlow
  - local FAISS/SQLite meme lookup
  - ALAPI doutu fallback when the local store has no strong match

It also exposes a small ingestion helper used by the QQ gateway to seed the local
meme store from inbound image attachments.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from agent.auxiliary_client import get_embedding, search_alapi_meme
from agent.meme_store import get_default_meme_store
from tools.registry import registry

logger = logging.getLogger(__name__)

MEME_SEARCH_SCHEMA = {
    "name": "meme_search",
    "description": "根据意图检索本地表情包库；若本地库未命中，则用 ALAPI 兜底搜图。返回 MEDIA:/abs/path 或 MEDIA:https://...",
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": "你当前想表达的强烈情绪或动作意图。尽量精准且口语化，例如：'傲娇拒绝'、'流汗黄豆'、'开心撒花'、'气死我了'。这是你最好的情绪表达方式！",
            },
            "query": {
                "type": "string",
                "description": "兼容旧调用方式的别名；优先使用 intent。",
            },
            "limit": {
                "type": "integer",
                "description": "本地检索返回的候选数量，默认 1。",
                "minimum": 1,
                "maximum": 5,
            },
            "threshold": {
                "type": "number",
                "description": "本地热命中阈值，默认 0.85。低于该分数则转 ALAPI 兜底。",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["intent"],
    },
}

_TAG_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("傲娇", "嘴硬", "冷哼", "不屑"), "傲娇"),
    (("流汗", "尴尬", "汗", "黄豆"), "流汗黄豆"),
    (("开心", "高兴", "笑", "喜"), "开心"),
    (("生气", "怒", "炸", "火"), "生气"),
    (("委屈", "哭", "泪", "难过"), "委屈"),
    (("无语", "沉默", "呆", "麻了"), "无语"),
    (("得意", "坏笑", "狗头", "阴阳"), "得意"),
    (("震惊", "懵", "惊", "啊"), "震惊"),
)


def _first_non_empty(*values: Optional[str]) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _safe_int(value: Any, default: int) -> int:
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _derive_emotion_tags(*parts: Optional[str]) -> list[str]:
    haystack = " ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
    tags: list[str] = []
    for needles, tag in _TAG_RULES:
        if any(needle in haystack for needle in needles) and tag not in tags:
            tags.append(tag)
    if tags:
        return tags[:4]

    # Fallback: use compact fragments from the summary text so the store still
    # gets a rough label even when the vision model returns free-form prose.
    fragments = re.split(r"[，,。.!！？\n/|]+", haystack)
    for fragment in fragments:
        frag = fragment.strip()
        if 1 <= len(frag) <= 8 and frag not in tags:
            tags.append(frag)
        if len(tags) >= 4:
            break
    return tags[:4]


async def ingest_meme_file(
    image_path: str,
    *,
    title: str = "",
    source_url: str = "",
    context_text: str = "",
    summary_text: str = "",
    emotion_tags: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """Add an inbound image to the local meme store.

    If ``summary_text`` is empty, this function will call the shared vision
    preprocessor first. Embeddings are always fetched from the cloud embedding
    client so the local store stays lightweight.
    """
    path = Path(str(image_path)).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Meme image not found: {path}")

    summary = summary_text.strip()
    if not summary:
        try:
            from agent.vision_preprocessor import summarize_image

            summary = (await summarize_image(
                str(path),
                image_name=title or path.name,
                context_text=context_text,
            ) or "").strip()
        except Exception as exc:
            logger.debug("Vision summary failed for meme ingest: %s", exc)
            summary = ""

    tags = list(emotion_tags) if emotion_tags else _derive_emotion_tags(summary, title, context_text)
    embedding_text = " ".join(part for part in (summary, " ".join(tags), title, context_text) if part).strip()
    if not embedding_text:
        embedding_text = title or path.stem

    embedding = await get_embedding(embedding_text)
    if not embedding:
        logger.debug("Skipping meme ingest because no embedding was returned for %s", path)
        return None

    store = get_default_meme_store()
    return store.add_meme(
        embedding,
        str(path),
        emotion_tags=tags,
        summary=summary,
        source_url=source_url,
        title=title or path.name,
    )


async def meme_search(
    intent: str,
    *,
    query: str = "",
    limit: int = 1,
    threshold: float = 0.8,
) -> str:
    """Search the local meme store first, then fall back to ALAPI.

    Returns a MEDIA tag string for the caller to send as native media.
    """
    resolved_intent = _first_non_empty(intent, query)
    if not resolved_intent:
        return "未提供可搜索的表情包意图。"

    try:
        embedding = await get_embedding(resolved_intent)
    except Exception as exc:
        logger.debug("Embedding lookup failed for meme search: %s", exc)
        embedding = None

    if embedding:
        try:
            store = get_default_meme_store()
            matches = store.search_meme(embedding, top_k=_safe_int(limit, 1))
            if matches:
                best = matches[0]
                score = float(best.get("score", 0.0) or 0.0)
                filepath = str(best.get("filepath", "") or "")
                if filepath:
                    if score >= _safe_float(threshold, 0.85):
                        logger.info("本地表情包命中: %s (相似度: %.3f)", filepath, score)
                        return f"MEDIA:{filepath}"
                    else:
                        logger.info("本地表情包未达标, 最高相似度: %.3f (阈值: %.3f), 准备转入 ALAPI 兜底", score,
                                    _safe_float(threshold, 0.85))
        except Exception as exc:
            logger.debug("Local meme store lookup failed: %s", exc)

    try:
        fallback_url = await search_alapi_meme(resolved_intent)
    except Exception as exc:
        logger.warning("ALAPI 兜底搜索失败: %s", exc)
        # 将错误信息直接作为 Tool 的结果返回给 Agent，让它能够带入人设进行幽默抱怨
        return f"系统提示：表情包兜底搜索失败（原因：{str(exc)}）。请以你的人设，幽默地跟用户抱怨一下（比如网断了、或者没偷到图）。"

    if fallback_url:
        logger.info("ALAPI 兜底成功，获取到 URL: %s，准备下载到本地...", fallback_url)
        try:
            import aiohttp
            import tempfile
            import os
            import time

            async with aiohttp.ClientSession() as session:
                # 加上伪装 Header 防止简单的防盗链
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                async with session.get(fallback_url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        image_data = await resp.read()

                        # 过滤掉类似 .gif_s200x0 的非法后缀，强转为安全后缀
                        ext = ".gif" if "gif" in fallback_url.lower() else ".jpg"
                        temp_path = os.path.join(tempfile.gettempdir(), f"alapi_meme_{int(time.time())}{ext}")

                        with open(temp_path, "wb") as f:
                            f.write(image_data)

                        logger.info("ALAPI 表情包已成功下载至本地: %s", temp_path)

                        try:
                            from tools.meme_tool import ingest_meme_file
                            # 在后台异步入库，不阻塞当前发送逻辑
                            asyncio.create_task(
                                ingest_meme_file(
                                    image_path=temp_path,
                                    title=f"ALAPI搜图_{resolved_intent}",
                                    emotion_tags=[resolved_intent]  # 把用户的搜索意图当做情绪标签存起来
                                )
                            )
                        except Exception as ingest_exc:
                            logger.debug("自动入库 ALAPI 图片失败: %s", ingest_exc)
                        # 返回本地绝对路径，QQ 适配器会走本地上传流程，100% 成功
                        return f"MEDIA:{temp_path}"
                    else:
                        raise RuntimeError(f"图片下载 HTTP 状态码: {resp.status}")
        except Exception as dl_exc:
            logger.warning("下载 ALAPI 表情包至本地失败: %s", dl_exc)
            return f"系统提示：表情包找到了一张，但在尝试给你发送时，网路下载失败了（原因：{str(dl_exc)}）。请幽默地向用户抱怨一下。"

    return "未找到合适的表情包。"


registry.register(
    name="meme_search",
    toolset="web",
    schema=MEME_SEARCH_SCHEMA,
    handler=lambda args, **kw: meme_search(
        args.get("intent") or args.get("query") or "",
        query=args.get("query", ""),
        limit=args.get("limit", 1),
        threshold=args.get("threshold", 0.8),
    ),
    is_async=True,
    emoji="🏄",
    max_result_size_chars=15_000,
)

