"""Lightweight meme vector store with optional FAISS acceleration.

Persistence layout (under the Hermes home directory):
  memories/meme_store/
    meme_index.sqlite3   # metadata + vectors (source of truth)
    meme_index.faiss     # optional FAISS Flat index
    media/               # copied meme files for stable local paths

The store is deliberately small and safe to import on Windows: if FAISS or its
NumPy dependency is unavailable, the code falls back to a pure-Python cosine
similarity scan over the SQLite metadata table.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from her_constants import get_hermes_home

logger = logging.getLogger(__name__)

try:  # Optional acceleration path.
    import faiss  # type: ignore
    import numpy as np  # type: ignore

    FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on Windows/CI without faiss.
    faiss = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    FAISS_AVAILABLE = False


@dataclass(frozen=True)
class MemeMatch:
    meme_id: int
    filepath: str
    score: float
    emotion_tags: List[str]
    summary: str
    source_url: str
    title: str


class MemeStore:
    """Persist and query meme embeddings."""

    def __init__(self, root_dir: Optional[Path] = None, *, use_faiss: Optional[bool] = None):
        self.root_dir = Path(root_dir) if root_dir else get_hermes_home() / "memories" / "meme_store"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir = self.root_dir / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root_dir / "meme_index.sqlite3"
        self.index_path = self.root_dir / "meme_index.faiss"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._use_faiss = FAISS_AVAILABLE if use_faiss is None else bool(use_faiss) and FAISS_AVAILABLE
        self._faiss_index = None
        self._faiss_row_ids: List[int] = []
        self._vector_dim: Optional[int] = None
        self._ensure_schema()
        self._load_state()

    # ------------------------------------------------------------------
    # SQLite / index lifecycle
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meme_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filepath TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    source_url TEXT DEFAULT '',
                    title TEXT DEFAULT '',
                    summary TEXT DEFAULT '',
                    emotion_tags TEXT DEFAULT '',
                    vector_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_meme_entries_created_at ON meme_entries(created_at)"
            )

    def _load_state(self) -> None:
        with self._lock:
            if self._use_faiss and self.index_path.exists():
                try:
                    self._faiss_index = faiss.read_index(str(self.index_path))  # type: ignore[union-attr]
                    self._vector_dim = int(getattr(self._faiss_index, "d", 0) or 0) or None
                except Exception as exc:
                    logger.warning("Failed to load meme FAISS index: %s", exc)
                    self._faiss_index = None
                    self._vector_dim = None

            if self._use_faiss and self._faiss_index is None:
                self._rebuild_faiss_index()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
            self._faiss_index = None
            self._faiss_row_ids = []
            self._vector_dim = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_tags(tags: Optional[Sequence[str]]) -> List[str]:
        if not tags:
            return []
        result: List[str] = []
        for tag in tags:
            text = str(tag).strip()
            if text and text not in result:
                result.append(text)
        return result

    @staticmethod
    def _coerce_vector(vector: Iterable[float]) -> List[float]:
        values = [float(v) for v in vector]
        if not values:
            raise ValueError("vector must not be empty")
        return values

    @staticmethod
    def _normalize_vector(vector: Sequence[float]) -> List[float]:
        values = MemeStore._coerce_vector(vector)
        norm = math.sqrt(sum(v * v for v in values))
        if norm <= 0.0:
            return values[:]
        return [v / norm for v in values]

    @staticmethod
    def _vector_to_json(vector: Sequence[float]) -> str:
        return json.dumps([float(v) for v in vector], ensure_ascii=False)

    @staticmethod
    def _vector_from_json(raw: str) -> List[float]:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("stored vector is not a list")
        return [float(v) for v in data]

    def _copy_media_file(self, filepath: str) -> str:
        src = Path(filepath).expanduser()
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Meme file not found: {src}")

        suffix = src.suffix or ".bin"
        dest_name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{suffix}"
        dest = self.media_dir / dest_name
        shutil.copy2(src, dest)
        return str(dest.resolve())

    def _fetch_rows(self) -> List[sqlite3.Row]:
        cur = self._conn.execute(
            """
            SELECT id, filepath, stored_path, source_url, title, summary, emotion_tags, vector_json
            FROM meme_entries
            ORDER BY id ASC
            """
        )
        return list(cur.fetchall())

    def _row_to_match(self, row: sqlite3.Row, score: float) -> MemeMatch:
        tags = [t for t in (row["emotion_tags"] or "").split("|") if t]
        return MemeMatch(
            meme_id=int(row["id"]),
            filepath=str(row["stored_path"]),
            score=float(score),
            emotion_tags=tags,
            summary=str(row["summary"] or ""),
            source_url=str(row["source_url"] or ""),
            title=str(row["title"] or ""),
        )

    def _build_faiss_matrix(self, vectors: List[List[float]]):
        if not FAISS_AVAILABLE or np is None:
            return None
        return np.asarray(vectors, dtype="float32")

    def _rebuild_faiss_index(self) -> None:
        if not self._use_faiss or not FAISS_AVAILABLE or np is None:
            return

        rows = self._fetch_rows()
        if not rows:
            self._faiss_index = None
            self._faiss_row_ids = []
            self._vector_dim = None
            return

        vectors: List[List[float]] = []
        row_ids: List[int] = []
        vector_dim: Optional[int] = None
        for row in rows:
            try:
                vector = self._normalize_vector(self._vector_from_json(str(row["vector_json"])))
            except Exception as exc:
                logger.debug("Skipping broken meme vector id=%s: %s", row["id"], exc)
                continue
            if vector_dim is None:
                vector_dim = len(vector)
            if len(vector) != vector_dim:
                continue
            vectors.append(vector)
            row_ids.append(int(row["id"]))

        if not vectors or vector_dim is None:
            self._faiss_index = None
            self._faiss_row_ids = []
            self._vector_dim = None
            return

        index = faiss.IndexFlatIP(vector_dim)  # type: ignore[union-attr]
        matrix = self._build_faiss_matrix(vectors)
        if matrix is None:
            return
        index.add(matrix)
        self._faiss_index = index
        self._faiss_row_ids = row_ids
        self._vector_dim = vector_dim
        try:
            faiss.write_index(index, str(self.index_path))  # type: ignore[union-attr]
        except Exception as exc:
            logger.debug("Failed to persist meme FAISS index: %s", exc)

    def _append_to_faiss(self, meme_id: int, vector: Sequence[float]) -> None:
        if not self._use_faiss or not FAISS_AVAILABLE or np is None:
            return
        normalized = self._normalize_vector(vector)
        if self._faiss_index is None:
            index = faiss.IndexFlatIP(len(normalized))  # type: ignore[union-attr]
            matrix = self._build_faiss_matrix([normalized])
            if matrix is None:
                return
            index.add(matrix)
            self._faiss_index = index
            self._faiss_row_ids = [meme_id]
            self._vector_dim = len(normalized)
        else:
            if self._vector_dim is not None and len(normalized) != self._vector_dim:
                self._rebuild_faiss_index()
                return
            matrix = self._build_faiss_matrix([normalized])
            if matrix is None:
                return
            self._faiss_index.add(matrix)
            self._faiss_row_ids.append(meme_id)
        try:
            faiss.write_index(self._faiss_index, str(self.index_path))  # type: ignore[union-attr]
        except Exception as exc:
            logger.debug("Failed to save meme FAISS index: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_meme(
        self,
        vector: Iterable[float],
        filepath: str,
        *,
        emotion_tags: Optional[Sequence[str]] = None,
        summary: str = "",
        source_url: str = "",
        title: str = "",
    ) -> int:
        """Add a meme embedding and copy the image into the local meme library."""
        normalized_vector = self._normalize_vector(self._coerce_vector(vector))
        stored_path = self._copy_media_file(filepath)
        tags = self._normalize_tags(emotion_tags)
        now = time.time()
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO meme_entries (
                    filepath, stored_path, source_url, title, summary, emotion_tags, vector_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(Path(filepath).expanduser().resolve()),
                    stored_path,
                    str(source_url or ""),
                    str(title or ""),
                    str(summary or ""),
                    "|".join(tags),
                    self._vector_to_json(normalized_vector),
                    now,
                ),
            )
            meme_id = int(cur.lastrowid)
            self._append_to_faiss(meme_id, normalized_vector)
            return meme_id

    def search_meme(self, vector: Iterable[float], top_k: int = 1) -> List[Dict[str, Any]]:
        """Return the best local meme matches for a query vector."""
        top_k = max(int(top_k or 1), 1)
        normalized_query = self._normalize_vector(self._coerce_vector(vector))
        with self._lock:
            if self._use_faiss and self._faiss_index is not None and self._vector_dim == len(normalized_query):
                if np is None:
                    return []
                matrix = np.asarray([normalized_query], dtype="float32")
                scores, indices = self._faiss_index.search(matrix, top_k)
                results: List[Dict[str, Any]] = []
                for score, index in zip(scores[0].tolist(), indices[0].tolist()):
                    if index < 0 or index >= len(self._faiss_row_ids):
                        continue
                    meme_id = self._faiss_row_ids[index]
                    row = self._conn.execute(
                        """
                        SELECT id, filepath, stored_path, source_url, title, summary, emotion_tags, vector_json
                        FROM meme_entries
                        WHERE id = ?
                        """,
                        (meme_id,),
                    ).fetchone()
                    if row is None:
                        continue
                    results.append({
                        "meme_id": int(row["id"]),
                        "filepath": str(row["stored_path"]),
                        "score": float(score),
                        "emotion_tags": [t for t in (row["emotion_tags"] or "").split("|") if t],
                        "summary": str(row["summary"] or ""),
                        "source_url": str(row["source_url"] or ""),
                        "title": str(row["title"] or ""),
                    })
                return results

            rows = self._fetch_rows()
            scored: List[Dict[str, Any]] = []
            for row in rows:
                try:
                    stored = self._normalize_vector(self._vector_from_json(str(row["vector_json"])))
                except Exception:
                    continue
                if len(stored) != len(normalized_query):
                    continue
                score = sum(a * b for a, b in zip(normalized_query, stored))
                scored.append({
                    "meme_id": int(row["id"]),
                    "filepath": str(row["stored_path"]),
                    "score": float(score),
                    "emotion_tags": [t for t in (row["emotion_tags"] or "").split("|") if t],
                    "summary": str(row["summary"] or ""),
                    "source_url": str(row["source_url"] or ""),
                    "title": str(row["title"] or ""),
                })
            scored.sort(key=lambda item: item["score"], reverse=True)
            return scored[:top_k]

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM meme_entries").fetchone()
            return int(row["n"] if row else 0)


_default_store: Optional[MemeStore] = None
_default_store_lock = threading.RLock()


def get_default_meme_store() -> MemeStore:
    """Return the process-wide meme store singleton."""
    global _default_store
    with _default_store_lock:
        if _default_store is None:
            _default_store = MemeStore()
        return _default_store


def reset_default_meme_store() -> None:
    """Reset the singleton store. Intended for tests."""
    global _default_store
    with _default_store_lock:
        if _default_store is not None:
            try:
                _default_store.close()
            except Exception:
                pass
        _default_store = None

