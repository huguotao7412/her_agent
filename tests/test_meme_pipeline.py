from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from agent import auxiliary_client as aux
from agent.meme_store import MemeStore
from gateway.platforms.qqbot.adapter import QQAdapter
from tools import meme_tool


class MemeStoreTests(unittest.TestCase):
    def test_add_and_search_without_faiss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemeStore(root_dir=Path(tmpdir), use_faiss=False)
            try:
                source = Path(tmpdir) / "source.jpg"
                source.write_bytes(b"fake image bytes")

                meme_id = store.add_meme(
                    [1.0, 0.0, 0.0],
                    str(source),
                    emotion_tags=["傲娇"],
                    summary="示例梗图",
                    source_url="https://example.com/original.jpg",
                    title="example",
                )

                self.assertEqual(meme_id, 1)
                self.assertEqual(store.count(), 1)

                matches = store.search_meme([1.0, 0.0, 0.0], top_k=1)
                self.assertEqual(len(matches), 1)
                best = matches[0]
                self.assertEqual(best["meme_id"], 1)
                self.assertEqual(best["title"], "example")
                self.assertIn("傲娇", best["emotion_tags"])
                self.assertGreater(best["score"], 0.99)
                self.assertTrue(Path(best["filepath"]).exists())
                self.assertNotEqual(Path(best["filepath"]).resolve(), source.resolve())
            finally:
                store.close()


class MemeToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_meme_search_prefers_local_store_and_falls_back_to_alapi(self) -> None:
        class FakeStore:
            def __init__(self, matches):
                self.matches = matches
                self.searched = []

            def search_meme(self, vector, top_k=1):
                self.searched.append((list(vector), top_k))
                return self.matches

        with patch.object(meme_tool, "get_embedding", new=AsyncMock(return_value=[0.1, 0.9])), \
             patch.object(meme_tool, "search_alapi_meme", new=AsyncMock(return_value="https://alapi.example/meme.jpg")) as alapi_mock:
            hot_store = FakeStore([
                {
                    "meme_id": 7,
                    "filepath": r"F:\meme\hot.jpg",
                    "score": 0.91,
                    "emotion_tags": ["傲娇"],
                    "summary": "hot",
                    "source_url": "",
                    "title": "hot meme",
                }
            ])
            with patch.object(meme_tool, "get_default_meme_store", return_value=hot_store):
                result = await meme_tool.meme_search("傲娇拒绝", threshold=0.8)
                self.assertEqual(result, "MEDIA:F:\\meme\\hot.jpg")
                alapi_mock.assert_not_awaited()

            cold_store = FakeStore([
                {
                    "meme_id": 8,
                    "filepath": r"F:\meme\cold.jpg",
                    "score": 0.2,
                    "emotion_tags": ["无语"],
                    "summary": "cold",
                    "source_url": "",
                    "title": "cold meme",
                }
            ])
            with patch.object(meme_tool, "get_default_meme_store", return_value=cold_store):
                result = await meme_tool.meme_search("流汗黄豆", threshold=0.8)
                self.assertEqual(result, "MEDIA:https://alapi.example/meme.jpg")
                self.assertGreaterEqual(alapi_mock.await_count, 1)


class AuxiliaryClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_call_llm_and_content_extraction(self) -> None:
        class FakeCompletions:
            def __init__(self):
                self.calls = []

            async def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="  ok  "))]
                )

        fake_completions = FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))

        with patch.object(aux, "_resolve_task_provider_model", return_value=("custom", "demo-model", None, None, None)), \
             patch.object(aux, "resolve_provider_client", return_value=(fake_client, "demo-model")), \
             patch.object(aux, "get_auxiliary_extra_body", return_value={"tags": ["product=her_agent-agent"]}):
            response = await aux.async_call_llm(
                task="web_extract",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=128,
                extra_body={"foo": "bar"},
            )

        self.assertEqual(len(fake_completions.calls), 1)
        call_kwargs = fake_completions.calls[0]
        self.assertEqual(call_kwargs["model"], "demo-model")
        self.assertEqual(call_kwargs["messages"][0]["content"], "hello")
        self.assertEqual(call_kwargs["extra_body"]["foo"], "bar")
        self.assertIn("tags", call_kwargs["extra_body"])
        self.assertEqual(aux.extract_content_or_reasoning(response), "ok")
        self.assertEqual(aux.extract_content_or_reasoning({"choices": [{"message": {"reasoning_content": " reasoning "}}]}), "reasoning")


class QQAdapterMemeIngestTests(unittest.TestCase):
    def test_queue_meme_ingest_schedules_background_task(self) -> None:
        adapter = QQAdapter.__new__(QQAdapter)
        adapter._app_id = "test-app"

        fake_task = Mock()
        fake_task.add_done_callback = Mock()

        async_ingest = AsyncMock(return_value=123)

        captured = {}
        def fake_create_task(coro):
            captured["coro"] = coro
            coro.close()
            return fake_task

        with patch("tools.meme_tool.ingest_meme_file", async_ingest), \
             patch("asyncio.create_task", side_effect=fake_create_task) as create_task_mock:
            adapter._queue_meme_ingest(
                r"F:\cache\meme.jpg",
                title="meme.jpg",
                source_url="https://example.com/meme.jpg",
                summary_text="流汗黄豆",
                context_text="群聊上下文",
            )

        create_task_mock.assert_called_once()
        self.assertIn("coro", captured)
        self.assertTrue(asyncio.iscoroutine(captured["coro"]))
        self.assertEqual(async_ingest.call_count, 1)
        ingest_args = async_ingest.call_args.args
        ingest_kwargs = async_ingest.call_args.kwargs
        self.assertEqual(ingest_args[0], r"F:\cache\meme.jpg")
        self.assertEqual(ingest_kwargs["title"], "meme.jpg")
        self.assertEqual(ingest_kwargs["source_url"], "https://example.com/meme.jpg")
        self.assertEqual(ingest_kwargs["summary_text"], "流汗黄豆")
        self.assertEqual(ingest_kwargs["context_text"], "群聊上下文")
        fake_task.add_done_callback.assert_called_once()


if __name__ == "__main__":
    unittest.main()

