"""Message splitting helpers for humanized delivery.

This module is intentionally pure: it turns a block of text into ordered
chunks without doing any I/O.  The gateway layer can then decide how to send
each chunk (text pacing, typing indicators, media delivery, etc.).
"""

from __future__ import annotations

import re
from typing import Dict, List

_MEDIA_TAG_RE = re.compile(r"MEDIA:\s*(\S+)")
_SENTENCE_SPLIT_RE = re.compile(r"([。！？!?])")
_NEWLINE_SPLIT_RE = re.compile(r"(\n+)")
_MAX_TEXT_CHUNK = 40


def _normalize_media_target(target: str) -> str:
    return target.strip().strip('"\'`').rstrip("，。！？!?.,;；")


def _emit_text_chunks(text: str) -> List[str]:
    chunks: List[str] = []
    pending = ""
    for part in _SENTENCE_SPLIT_RE.split(text):
        if not part:
            continue
        pending += part
        if part in ("。", "！", "？", "!", "?"):
            cleaned = pending.strip()
            if cleaned:
                chunks.append(cleaned)
            pending = ""
        while len(pending) > _MAX_TEXT_CHUNK:
            head = pending[:_MAX_TEXT_CHUNK].strip()
            if head:
                chunks.append(head)
            pending = pending[_MAX_TEXT_CHUNK:].lstrip()
    cleaned = pending.strip()
    if cleaned:
        chunks.append(cleaned)
    return chunks


def _split_text_segment(segment: str) -> List[str]:
    text = str(segment or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    for line in _NEWLINE_SPLIT_RE.split(text):
        if not line or not line.strip() or line.isspace():
            continue
        for item in _emit_text_chunks(line):
            chunks.append(item)

    return chunks


def split_into_chunks(text: str) -> List[Dict[str, str]]:
    """Split text into ordered text/media chunks.

    Media tags use the form ``MEDIA:/path/to/file`` or ``MEDIA:https://...``.
    They are extracted as standalone chunks and preserved in order relative to
    the surrounding text.
    """
    raw = str(text or "")
    if not raw.strip():
        return []

    chunks: List[Dict[str, str]] = []
    cursor = 0

    for match in _MEDIA_TAG_RE.finditer(raw):
        before = raw[cursor:match.start()]
        for text_chunk in _split_text_segment(before):
            chunks.append({"type": "text", "content": text_chunk})

        media_target = _normalize_media_target(match.group(1))
        if media_target:
            chunks.append({"type": "media", "content": media_target})

        cursor = match.end()

    tail = raw[cursor:]
    for text_chunk in _split_text_segment(tail):
        chunks.append({"type": "text", "content": text_chunk})

    return chunks
