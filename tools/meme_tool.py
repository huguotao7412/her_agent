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
import os
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
    import os
    import asyncio

    resolved_intent = _first_non_empty(intent, query)
    if not resolved_intent:
        return "未提供可搜索的表情包意图。"

    # ==========================================
    # 1. 本地热启动检索 (保留你的完美逻辑)
    # ==========================================
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

                if filepath and os.path.exists(filepath):
                    if score >= _safe_float(threshold, 0.85):
                        logger.info("本地表情包命中: %s", filepath)
                        return f"MEDIA:{filepath}"
                elif filepath:
                    logger.warning("脏数据过滤: %s", filepath)
        except Exception as exc:
            logger.debug("Local meme store lookup failed: %s", exc)

    # ==========================================
    # 2. 网络冷启动兜底 (前台下载 + 物理校验)
    # ==========================================
    try:
        fallback_url = await search_alapi_meme(resolved_intent)
    except Exception as exc:
        return f"系统提示：表情包兜底搜索失败（原因：{str(exc)}）。请幽默地跟用户抱怨一下。"

    if fallback_url:
        logger.info("ALAPI 获取 URL 成功，开始启动高匿下载: %s", fallback_url)
        try:
            import aiohttp
            import time

            async with aiohttp.ClientSession() as session:
                # 【核心防线 1】：浏览器级终极伪装，突破防盗链
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                    "Referer": "https://www.alapi.cn/"
                }
                async with session.get(fallback_url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        image_data = await resp.read()

                        # 【核心防线 2】：Magic Bytes 精确校验与后缀分配！
                        # 绝不能“张冠李戴”，严格分配后缀，否则 QQ 会将格式不符的图降级为“未知文件”
                        if image_data.startswith(b'GIF8'):
                            ext = ".gif"
                        elif image_data.startswith(b'\x89PNG'):
                            ext = ".png"
                        elif image_data.startswith(b'\xff\xd8'):
                            ext = ".jpg"
                        else:
                            # 过滤掉防盗链页面，以及 QQ 官方根本不支持渲染的 WebP (RIFF) 格式
                            logger.error("下载到了假图片，或是 QQ 不兼容的格式，阻断发送！")
                            return "（你想发个表情包，但是偷来的图格式不对发不出来，幽默地抱怨一下吧）"

                        if len(image_data) < 1024:
                            logger.error("图片体积过小，可能是残次品防盗链。")
                            return "（偷来的表情包裂开了，幽默地抱怨一句吧）"

                        # 存入纯英文安全路径
                        cache_dir = os.path.join(os.getcwd(), "cache", "images")
                        os.makedirs(cache_dir, exist_ok=True)
                        safe_local_path = os.path.join(cache_dir, f"alapi_{int(time.time())}{ext}")

                        # 写入真实的图片二进制文件
                        with open(safe_local_path, "wb") as f:
                            f.write(image_data)

                        logger.info("图片下载并物理校验成功: %s", safe_local_path)

                        # 后台异步入库，养肥你的私人图库
                        try:
                            from tools.meme_tool import ingest_meme_file
                            asyncio.create_task(
                                ingest_meme_file(
                                    image_path=safe_local_path,
                                    title=f"ALAPI_{resolved_intent}",
                                    emotion_tags=[resolved_intent]
                                )
                            )
                        except Exception as ingest_exc:
                            logger.debug("入库失败: %s", ingest_exc)

                        # 将纯天然无污染的本地绝对路径交还给网关
                        return f"MEDIA:{safe_local_path}"
                    else:
                        raise RuntimeError(f"HTTP {resp.status}")
        except Exception as e:
            logger.warning("下载任务崩溃: %s", e)
            return "（表情包下载超时，抱怨一下网络太差）"

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

