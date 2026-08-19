"""Shared fixtures and import stubs for the test suite.

The production modules import heavy, optional dependencies (sentence-transformers,
faiss, streamlit, edge-tts, ...) at import time and, in the case of ``app.py``, run
Streamlit UI code as a side effect of the import. The stubs below make those modules
importable in a plain CPython process so their pure logic can be unit tested.
"""

import importlib
import json
import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class FakeEncoder:
    """Stand-in for a SentenceTransformer model returning deterministic vectors."""

    def __init__(self, *args, **kwargs):
        self.calls = []

    def encode(self, text, **kwargs):
        self.calls.append(text)
        if isinstance(text, (list, tuple)):
            return np.zeros((len(text), 4), dtype=np.float32)
        return np.zeros(4, dtype=np.float32)


class FakeIndex:
    """Minimal FAISS index stub: records added vectors, returns fixed neighbours."""

    def __init__(self, dim=4):
        self.dim = dim
        self.ntotal = 0
        self.search_results = None

    def add(self, vectors):
        self.ntotal += len(vectors)

    def search(self, query, k):
        if self.search_results is not None:
            return self.search_results
        indices = np.array([list(range(k))], dtype=np.int64)
        distances = np.zeros((1, k), dtype=np.float32)
        return distances, indices


def _fake_faiss():
    module = types.ModuleType("faiss")
    module.IndexFlatL2 = lambda dim: FakeIndex(dim)
    module.write_index = MagicMock()
    module.read_index = MagicMock(return_value=FakeIndex())
    return module


def _fake_sentence_transformers():
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeEncoder
    return module


def _fake_tqdm():
    module = types.ModuleType("tqdm")

    class _Tqdm:
        def __init__(self, *args, **kwargs):
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, amount):
            self.n += amount

    module.tqdm = _Tqdm
    return module


class FakeSessionState(dict):
    """Streamlit's session_state supports both attribute and item access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def _fake_streamlit(secrets=None):
    module = types.ModuleType("streamlit")
    errors = types.ModuleType("streamlit.errors")

    class StreamlitSecretNotFoundError(Exception):
        pass

    errors.StreamlitSecretNotFoundError = StreamlitSecretNotFoundError
    module.errors = errors
    sys.modules["streamlit.errors"] = errors
    module.cache_resource = lambda func: func
    module.cache_data = lambda func: func
    module.session_state = FakeSessionState()
    module.secrets = secrets if secrets is not None else {}

    # MagicMock supports the context manager protocol, both when used directly
    # (``with st.sidebar:``) and when called first (``with st.spinner("..."):``).
    for name in ("sidebar", "container", "expander", "chat_message", "spinner", "status", "form"):
        setattr(module, name, MagicMock())

    module.columns = lambda spec, **kwargs: [MagicMock() for _ in (spec if isinstance(spec, (list, tuple)) else range(spec))]
    module.chat_input = MagicMock(return_value=None)
    module.audio_input = MagicMock(return_value=None)
    module.file_uploader = MagicMock(return_value=None)
    module.button = MagicMock(return_value=False)
    module.toggle = MagicMock(return_value=False)
    module.selectbox = MagicMock(return_value="British Female (Sonia)")
    module.stop = MagicMock(side_effect=RuntimeError("st.stop() called"))
    module.rerun = MagicMock()
    for name in ("set_page_config", "markdown", "write", "caption", "error", "success",
                 "warning", "info", "audio", "title", "header", "subheader", "divider"):
        setattr(module, name, MagicMock())

    errors_module = types.ModuleType("streamlit.errors")

    class StreamlitSecretNotFoundError(Exception):
        pass

    errors_module.StreamlitSecretNotFoundError = StreamlitSecretNotFoundError
    module.errors = errors_module
    return module


@pytest.fixture
def stub_modules(monkeypatch):
    """Install the heavy-dependency stubs into ``sys.modules``."""
    monkeypatch.setitem(sys.modules, "faiss", _fake_faiss())
    monkeypatch.setitem(sys.modules, "sentence_transformers", _fake_sentence_transformers())
    monkeypatch.setitem(sys.modules, "tqdm", _fake_tqdm())
    return sys.modules


@pytest.fixture
def build_index_module(stub_modules, monkeypatch):
    """Freshly imported ``build_index`` with stubbed embedding/index backends."""
    monkeypatch.delitem(sys.modules, "build_index", raising=False)
    module = importlib.import_module("build_index")
    return module


@pytest.fixture
def scraper():
    """The scraper module (no heavy dependencies)."""
    return importlib.import_module("scrape_eu_news")


@pytest.fixture
def app_module(stub_modules, monkeypatch):
    """Freshly imported ``app`` with Streamlit and model backends stubbed out."""
    fake_streamlit = _fake_streamlit({"GROQ_API_KEY": "test-key"})
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "streamlit.errors", fake_streamlit.errors)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delitem(sys.modules, "app", raising=False)
    module = importlib.import_module("app")
    return module


@pytest.fixture
def article_factory():
    """Build article dicts shaped like the entries in ``eu_news_data.json``."""

    def _make(title="Commission adopts trade package", content="Trade content. " * 20, **overrides):
        article = {
            "title": title,
            "date": "Monday, 29 June 2026",
            "source": "European Commission",
            "link": "https://commission.europa.eu/news/example_en",
            "content": content,
        }
        article.update(overrides)
        return article

    return _make


@pytest.fixture
def news_json(tmp_path, article_factory):
    """Write a small ``eu_news_data.json`` style file and return its path."""
    path = tmp_path / "eu_news_data.json"
    articles = [
        article_factory(title="First article", content="A" * 2500),
        article_factory(title="Second article", content="Short body"),
    ]
    path.write_text(json.dumps(articles), encoding="utf-8")
    return path
