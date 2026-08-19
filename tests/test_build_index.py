"""Unit tests for the chunking / index building pipeline."""

import json

import pytest


class TestSimpleTextSplitter:
    def test_text_shorter_than_chunk_size_is_returned_whole(self, build_index_module):
        assert build_index_module.simple_text_splitter("short text") == ["short text"]

    def test_text_exactly_chunk_size_is_returned_whole(self, build_index_module):
        text = "a" * 10
        assert build_index_module.simple_text_splitter(text, chunk_size=10, overlap=2) == [text]

    def test_empty_text_returns_single_empty_chunk(self, build_index_module):
        assert build_index_module.simple_text_splitter("") == [""]

    def test_chunks_respect_chunk_size(self, build_index_module):
        chunks = build_index_module.simple_text_splitter("a" * 100, chunk_size=10, overlap=2)
        assert all(len(chunk) <= 10 for chunk in chunks)

    def test_consecutive_chunks_share_the_overlap(self, build_index_module):
        text = "".join(str(i % 10) for i in range(60))
        chunks = build_index_module.simple_text_splitter(text, chunk_size=20, overlap=5)
        for first, second in zip(chunks, chunks[1:]):
            assert first[-5:] == second[:5]

    def test_chunks_cover_the_whole_text(self, build_index_module):
        text = "".join(str(i % 10) for i in range(95))
        chunks = build_index_module.simple_text_splitter(text, chunk_size=20, overlap=5)
        rebuilt = chunks[0] + "".join(chunk[5:] for chunk in chunks[1:])
        assert rebuilt == text

    def test_zero_overlap_produces_disjoint_chunks(self, build_index_module):
        text = "".join(str(i % 10) for i in range(50))
        chunks = build_index_module.simple_text_splitter(text, chunk_size=10, overlap=0)
        assert "".join(chunks) == text

    def test_default_chunk_size_and_overlap_are_used(self, build_index_module):
        text = "x" * (build_index_module.CHUNK_SIZE + 100)
        chunks = build_index_module.simple_text_splitter(text)
        assert len(chunks[0]) == build_index_module.CHUNK_SIZE
        assert len(chunks) == 2


class TestGenerateEmbedding:
    def test_returns_a_plain_list_of_floats(self, build_index_module):
        embedding = build_index_module.generate_embedding("some text")
        assert isinstance(embedding, list)
        assert len(embedding) == 4

    def test_returns_none_when_the_model_fails(self, build_index_module, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(build_index_module.embed_model, "encode", _boom)
        assert build_index_module.generate_embedding("some text") is None


class TestMain:
    @pytest.fixture
    def run_main(self, build_index_module, monkeypatch, news_json, tmp_path):
        def _run():
            monkeypatch.setattr(build_index_module, "DATA_FILE", str(news_json))
            monkeypatch.setattr(build_index_module, "ITEMS_FILE", str(tmp_path / "items.json"))
            monkeypatch.setattr(build_index_module, "INDEX_FILE", str(tmp_path / "news.faiss"))
            monkeypatch.setattr(build_index_module, "EMBEDDING_DIM", 4)
            build_index_module.main()
            return tmp_path

        return _run

    def test_writes_index_and_json_items(self, build_index_module, run_main):
        out = run_main()

        build_index_module.faiss.write_index.assert_called_once()
        assert (out / "items.json").exists()

    def test_long_articles_are_chunked_into_multiple_items(self, run_main, tmp_path):
        run_main()

        items = json.loads((tmp_path / "items.json").read_text(encoding="utf-8"))
        chunks_per_title = {}
        for item in items:
            chunks_per_title.setdefault(item["metadata"]["title"], 0)
            chunks_per_title[item["metadata"]["title"]] += 1
        assert chunks_per_title["First article"] > 1
        assert chunks_per_title["Second article"] == 1

    def test_items_carry_article_metadata_without_embeddings(self, run_main, tmp_path):
        run_main()

        item = json.loads((tmp_path / "items.json").read_text(encoding="utf-8"))[0]
        assert item["type"] == "text"
        assert item["metadata"]["source"] == "European Commission"
        assert item["metadata"]["url"].startswith("https://")
        assert "embedding" not in item
        assert item["text"].startswith("Date: ")

    def test_items_json_excludes_embeddings(self, run_main, tmp_path):
        run_main()

        items = json.loads((tmp_path / "items.json").read_text(encoding="utf-8"))
        assert all("embedding" not in item for item in items)
        assert all("text" in item for item in items)

    def test_every_chunk_is_added_to_the_index(self, build_index_module, run_main, tmp_path):
        run_main()

        items = json.loads((tmp_path / "items.json").read_text(encoding="utf-8"))
        index = build_index_module.faiss.write_index.call_args.args[0]
        assert index.ntotal == len(items)
