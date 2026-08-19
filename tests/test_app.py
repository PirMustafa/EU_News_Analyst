"""Unit tests for the RAG logic inside the Streamlit app."""

import asyncio
from unittest.mock import MagicMock

import pytest


class TestDetectQueryType:
    @pytest.mark.parametrize("query", [
        "What's the news?",
        "todays news please",
        "Give me the headlines",
        "daily briefing",
        "Any news on Brexit",
        "OVERVIEW",
        "  what is the news  ",
    ])
    def test_overview_patterns_win_over_everything(self, app_module, query):
        assert app_module.detect_query_type(query, ["prior message"]) == "overview"

    def test_specific_query_with_history_is_detailed(self, app_module):
        assert app_module.detect_query_type("Tell me about tariffs", ["prior"]) == "detailed"

    def test_short_vague_query_without_history_is_overview(self, app_module):
        assert app_module.detect_query_type("tariffs", []) == "overview"

    @pytest.mark.parametrize("query", [
        "why tariffs",
        "how tariffs",
        "explain tariffs",
        "details tariffs",
        "more about tariffs",
        "tell me about tariffs",
    ])
    def test_short_query_with_analytical_keyword_is_detailed(self, app_module, query):
        assert app_module.detect_query_type(query, []) == "detailed"

    def test_long_query_without_history_is_detailed(self, app_module):
        query = "what did the commission decide regarding steel import tariffs this week"
        assert app_module.detect_query_type(query, []) == "detailed"


class TestGetEmbedding:
    def test_returns_a_list(self, app_module):
        assert app_module.get_embedding("query text") == [0.0, 0.0, 0.0, 0.0]

    def test_returns_none_on_failure(self, app_module, monkeypatch):
        monkeypatch.setattr(app_module.embed_model, "encode",
                            MagicMock(side_effect=RuntimeError("no model")))
        assert app_module.get_embedding("query text") is None


class TestGroqGenerate:
    def _post(self, content="generated text"):
        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": content}}]}
        return MagicMock(return_value=response)

    def test_returns_the_first_choice_content(self, app_module, monkeypatch):
        import requests

        post = self._post()
        monkeypatch.setattr(requests, "post", post)

        assert app_module.groq_generate("prompt") == "generated text"

    def test_sends_the_prompt_and_model_to_the_api(self, app_module, monkeypatch):
        import requests

        post = self._post()
        monkeypatch.setattr(requests, "post", post)

        app_module.groq_generate("my prompt", model="custom-model")

        payload = post.call_args.kwargs["json"]
        assert payload["model"] == "custom-model"
        assert payload["messages"] == [{"role": "user", "content": "my prompt"}]
        assert post.call_args.kwargs["headers"]["Authorization"].startswith("Bearer ")

    def test_http_errors_propagate(self, app_module, monkeypatch):
        import requests

        response = MagicMock()
        response.raise_for_status.side_effect = RuntimeError("429 rate limited")
        monkeypatch.setattr(requests, "post", MagicMock(return_value=response))

        with pytest.raises(RuntimeError):
            app_module.groq_generate("prompt")


@pytest.fixture
def todays_articles(app_module, article_factory):
    """Articles dated today, i.e. the ones the app considers current."""
    return [
        article_factory(title=f"Article {i} about tariffs and steel", date=app_module.DATE_STR,
                        content=f"Body {i} discussing tariffs and steel imports. " * 5)
        for i in range(12)
    ]


class TestAnalyzeQuery:
    def test_returns_placeholder_when_no_news_for_today(self, app_module, article_factory):
        stale = [article_factory(date="Monday, 01 January 1990")]

        result = app_module.analyze_query("headlines", stale, index=None, items=[])

        assert "No news data available" in result["analysis"]
        assert result["sources"] == []
        assert result["query_type"] == "overview"

    def test_overview_mode_returns_generated_briefing_and_ten_sources(
            self, app_module, monkeypatch, todays_articles):
        monkeypatch.setattr(app_module, "groq_generate", MagicMock(return_value="## Briefing"))

        result = app_module.analyze_query("what's the news", todays_articles, index=None, items=[],
                                          query_type="overview")

        assert result["analysis"] == "## Briefing"
        assert result["thoughts"] == ""
        assert result["query_type"] == "overview"
        assert len(result["sources"]) == 10
        assert len(result["sources"][0]["content"]) <= 300

    def test_overview_prompt_includes_headlines_and_article_count(
            self, app_module, monkeypatch, todays_articles):
        generate = MagicMock(return_value="briefing")
        monkeypatch.setattr(app_module, "groq_generate", generate)

        app_module.analyze_query("headlines", todays_articles, index=None, items=[])

        prompt = generate.call_args.args[0]
        assert "Article 0 about tariffs and steel" in prompt
        assert f"TOTAL ARTICLES: {len(todays_articles)}" in prompt

    def test_overview_llm_failure_is_reported_with_sources(
            self, app_module, monkeypatch, todays_articles):
        monkeypatch.setattr(app_module, "groq_generate", MagicMock(side_effect=RuntimeError("boom")))

        result = app_module.analyze_query("headlines", todays_articles, index=None, items=[])

        assert "Error generating overview" in result["analysis"]
        assert result["sources"]

    def test_detailed_mode_returns_analysis_and_thoughts(
            self, app_module, monkeypatch, todays_articles):
        monkeypatch.setattr(app_module, "groq_generate",
                            MagicMock(side_effect=["analysis text", "thoughts text"]))

        result = app_module.analyze_query("why were steel tariffs raised", todays_articles,
                                          index=None, items=[], query_type="detailed")

        assert result["analysis"] == "analysis text"
        assert result["thoughts"] == "thoughts text"
        assert result["query_type"] == "detailed"
        assert len(result["sources"]) == 5

    def test_detailed_mode_uses_keyword_matching_articles_in_the_prompt(
            self, app_module, monkeypatch, article_factory):
        news = [
            article_factory(title="Steel tariffs decision", date=app_module.DATE_STR,
                            content="steel content"),
            article_factory(title="Fisheries quota update", date=app_module.DATE_STR,
                            content="fish content"),
        ]
        generate = MagicMock(side_effect=["analysis", "thoughts"])
        monkeypatch.setattr(app_module, "groq_generate", generate)

        app_module.analyze_query("steel tariffs", news, index=None, items=[],
                                 query_type="detailed")

        prompt = generate.call_args_list[0].args[0]
        assert "Steel tariffs decision" in prompt
        assert "Fisheries quota update" not in prompt

    def test_detailed_mode_falls_back_to_faiss_neighbours(
            self, app_module, monkeypatch, article_factory):
        news = [
            article_factory(title="Fisheries quota update", date=app_module.DATE_STR,
                            content="fish content"),
            article_factory(title="Digital services rules", date=app_module.DATE_STR,
                            content="digital content"),
        ]
        items = [{"text": "chunk", "meta": {"title": "Digital services rules"}}]

        class _Index:
            def search(self, query, k):
                import numpy as np
                return np.zeros((1, 1)), np.array([[0]])

        generate = MagicMock(side_effect=["analysis", "thoughts"])
        monkeypatch.setattr(app_module, "groq_generate", generate)

        app_module.analyze_query("unrelated xyzzy", news, index=_Index(), items=items,
                                 query_type="detailed")

        prompt = generate.call_args_list[0].args[0]
        assert "Digital services rules" in prompt
        assert "Fisheries quota update" not in prompt

    def test_detailed_mode_falls_back_to_top_articles(
            self, app_module, monkeypatch, todays_articles):
        generate = MagicMock(side_effect=["analysis", "thoughts"])
        monkeypatch.setattr(app_module, "groq_generate", generate)

        app_module.analyze_query("completely unrelated xyzzy topic here", todays_articles,
                                 index=None, items=[], query_type="detailed")

        prompt = generate.call_args_list[0].args[0]
        assert prompt.count("ARTICLE:") == 5

    def test_detailed_llm_failure_is_reported(self, app_module, monkeypatch, todays_articles):
        monkeypatch.setattr(app_module, "groq_generate", MagicMock(side_effect=RuntimeError("boom")))

        result = app_module.analyze_query("why steel tariffs", todays_articles, index=None,
                                         items=[], query_type="detailed")

        assert "Analysis generation failed" in result["analysis"]
        assert result["thoughts"] == ""

    def test_auto_mode_delegates_to_query_type_detection(
            self, app_module, monkeypatch, todays_articles):
        monkeypatch.setattr(app_module, "groq_generate", MagicMock(return_value="text"))
        detect = MagicMock(return_value="overview")
        monkeypatch.setattr(app_module, "detect_query_type", detect)

        result = app_module.analyze_query("anything", todays_articles, index=None, items=[],
                                         query_type="auto", conversation_history=["prior"])

        detect.assert_called_once_with("anything", ["prior"])
        assert result["query_type"] == "overview"


class TestSpeechToText:
    def test_returns_none_when_recognition_is_unavailable(self, app_module):
        assert app_module.speech_to_text(b"not audio") is None

    @staticmethod
    def _stub_recognizer(monkeypatch, recognize):
        import sys
        import types

        module = types.ModuleType("speech_recognition")
        module.paths = []

        class _AudioFile:
            def __init__(self, path):
                module.paths.append(path)

            def __enter__(self):
                return "source"

            def __exit__(self, *exc):
                return False

        class _Recognizer:
            def record(self, source):
                return f"audio-from-{source}"

            def recognize_google(self, audio_data):
                return recognize(audio_data)

        module.AudioFile = _AudioFile
        module.Recognizer = _Recognizer
        monkeypatch.setitem(sys.modules, "speech_recognition", module)
        return module

    def test_transcribes_audio_and_removes_the_temp_file(self, app_module, monkeypatch):
        import os

        module = self._stub_recognizer(monkeypatch, lambda audio: "what is the news")

        assert app_module.speech_to_text(b"RIFFfake") == "what is the news"
        assert not os.path.exists(module.paths[0])

    def test_recognition_errors_return_none(self, app_module, monkeypatch):
        def _boom(audio):
            raise RuntimeError("unintelligible")

        self._stub_recognizer(monkeypatch, _boom)

        assert app_module.speech_to_text(b"RIFFfake") is None


class TestTextToSpeech:
    def test_returns_none_when_edge_tts_is_unavailable(self, app_module, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "edge_tts", None)

        assert asyncio.run(app_module.text_to_speech("hello")) is None

    def test_strips_markdown_urls_and_sources_before_synthesis(self, app_module, monkeypatch):
        import sys
        import types

        captured = {}

        class _Communicate:
            def __init__(self, text, voice):
                captured["text"] = text
                captured["voice"] = voice

            async def stream(self):
                yield {"type": "audio", "data": b"abc"}
                yield {"type": "WordBoundary"}

        module = types.ModuleType("edge_tts")
        module.Communicate = _Communicate
        monkeypatch.setitem(sys.modules, "edge_tts", module)

        text = "**Bold** point 1. See http://example.com [ref]\n**Sources:** hidden"
        buffer = asyncio.run(app_module.text_to_speech(text, voice="en-US-JennyNeural"))

        assert buffer.read() == b"abc"
        assert captured["voice"] == "en-US-JennyNeural"
        assert "hidden" not in captured["text"]
        assert "http://example.com" not in captured["text"]
        assert "**" not in captured["text"]
        assert "[ref]" not in captured["text"]


class TestLoadData:
    def test_reports_error_status_when_index_is_missing(self, app_module, monkeypatch):
        monkeypatch.setattr(app_module.faiss, "read_index",
                            MagicMock(side_effect=FileNotFoundError("news_index.faiss")))

        index, items, news_data, stats, status = app_module.load_data()

        assert index is None
        assert (items, news_data, stats) == ([], [], {})
        assert status.startswith("Error:")

    def test_loads_items_news_and_stats(self, app_module, monkeypatch, tmp_path, article_factory):
        import json
        import pickle

        items_with_embeddings = [
            {"text": "chunk 1", "metadata": {"title": "A"}, "embedding": [0.0]},
            {"text": "chunk 2", "metadata": {"title": "A"}, "embedding": [0.0]},
            {"text": "chunk 3", "metadata": {"title": "B"}, "embedding": [0.0]},
        ]
        (tmp_path / "items_with_embeddings.pkl").write_bytes(pickle.dumps(items_with_embeddings))
        (tmp_path / "eu_news_data.json").write_text(json.dumps([article_factory()]), encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        index, items, news_data, stats, status = app_module.load_data()

        assert status == "Online"
        assert index is not None
        assert items[0] == {"text": "chunk 1", "meta": {"title": "A"}}
        assert len(news_data) == 1
        assert stats == {"articles": 2, "chunks": 3}
