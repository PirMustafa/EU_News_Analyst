"""Unit tests for the EU Commission scraper."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest


def _response(html="", status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.content = html.encode("utf-8")
    return response


class TestGetPageUrl:
    def test_first_page_has_no_query_parameter(self, scraper):
        assert scraper.get_page_url(1) == scraper.NEWS_PAGE

    @pytest.mark.parametrize("page_num,expected_param", [(2, 1), (3, 2), (50, 49)])
    def test_later_pages_are_zero_indexed(self, scraper, page_num, expected_param):
        assert scraper.get_page_url(page_num) == f"{scraper.NEWS_PAGE}?page={expected_param}"


class TestParseDate:
    @pytest.mark.parametrize("date_str,expected", [
        ("02 February 2026", datetime(2026, 2, 2)),
        ("February 02, 2026", datetime(2026, 2, 2)),
        ("2026-02-02", datetime(2026, 2, 2)),
        ("02/02/2026", datetime(2026, 2, 2)),
        ("02.02.2026", datetime(2026, 2, 2)),
    ])
    def test_supported_formats(self, scraper, date_str, expected):
        assert scraper.parse_date(date_str) == expected

    def test_surrounding_whitespace_is_ignored(self, scraper):
        assert scraper.parse_date("\n  15 March 2026 \t") == datetime(2026, 3, 15)

    @pytest.mark.parametrize("date_str", [None, "", "   ", "not a date", "2026", "31/02/2026"])
    def test_unparseable_input_returns_none(self, scraper, date_str):
        assert scraper.parse_date(date_str) is None


class TestScrapeArticleContent:
    def test_extracts_title_date_and_content(self, scraper, monkeypatch):
        html = """
        <html><body>
          <h1> Commission adopts new trade package </h1>
          <time datetime="2026-06-15T10:30:00+00:00">15 June 2026</time>
          <article><p>First paragraph.</p><p>Second   paragraph.</p></article>
        </body></html>
        """
        monkeypatch.setattr(scraper.requests, "get", MagicMock(return_value=_response(html)))

        result = scraper.scrape_article_content("https://commission.europa.eu/news/a_en")

        assert result["title"] == "Commission adopts new trade package"
        assert result["date"] == datetime(2026, 6, 15)
        assert result["content"] == "First paragraph. Second paragraph."

    def test_falls_back_to_plain_date_text(self, scraper, monkeypatch):
        html = "<h1>T</h1><div class='date'>15 June 2026</div><article><p>Body text.</p></article>"
        monkeypatch.setattr(scraper.requests, "get", MagicMock(return_value=_response(html)))

        assert scraper.scrape_article_content("https://x/news/a")["date"] == datetime(2026, 6, 15)

    def test_unparseable_date_yields_none_date_but_keeps_content(self, scraper, monkeypatch):
        html = "<h1>T</h1><div class='date'>sometime soon</div><article><p>Body text.</p></article>"
        monkeypatch.setattr(scraper.requests, "get", MagicMock(return_value=_response(html)))

        result = scraper.scrape_article_content("https://x/news/a")
        assert result["date"] is None
        assert result["content"] == "Body text."

    def test_missing_title_returns_none_title(self, scraper, monkeypatch):
        monkeypatch.setattr(scraper.requests, "get",
                            MagicMock(return_value=_response("<article><p>Body.</p></article>")))

        assert scraper.scrape_article_content("https://x/news/a")["title"] is None

    def test_prefers_first_selector_with_substantial_text(self, scraper, monkeypatch):
        html = """
        <div class="ecl-paragraph"><p>short</p></div>
        <div class="field--name-body"><p>{}</p></div>
        """.format("long body " * 20)
        monkeypatch.setattr(scraper.requests, "get", MagicMock(return_value=_response(html)))

        content = scraper.scrape_article_content("https://x/news/a")["content"]
        assert content.startswith("long body")

    def test_content_is_truncated_to_5000_characters(self, scraper, monkeypatch):
        html = "<h1>T</h1><article><p>{}</p></article>".format("x" * 6000)
        monkeypatch.setattr(scraper.requests, "get", MagicMock(return_value=_response(html)))

        assert len(scraper.scrape_article_content("https://x/news/a")["content"]) == 5000

    def test_non_200_response_returns_none(self, scraper, monkeypatch):
        monkeypatch.setattr(scraper.requests, "get",
                            MagicMock(return_value=_response("<h1>T</h1>", status_code=404)))

        assert scraper.scrape_article_content("https://x/news/a") is None

    def test_request_exception_returns_none(self, scraper, monkeypatch):
        monkeypatch.setattr(scraper.requests, "get", MagicMock(side_effect=OSError("boom")))

        assert scraper.scrape_article_content("https://x/news/a") is None


class TestScrapeNewsPage:
    def test_collects_relative_and_absolute_article_links(self, scraper, monkeypatch):
        html = """
        <a href="/news/first-article-with-long-title_en">First article with a long title</a>
        <a href="https://commission.europa.eu/news/second-article_en">Second article with long title</a>
        <a href="https://commission.europa.eu.evil.com/news/attacker">Attacker-controlled host</a>
        """
        monkeypatch.setattr(scraper.requests, "get", MagicMock(return_value=_response(html)))

        links, success = scraper.scrape_news_page(1)

        assert success is True
        assert [link["url"] for link in links] == [
            scraper.BASE_URL + "/news/first-article-with-long-title_en",
            "https://commission.europa.eu/news/second-article_en",
        ]

    def test_rejects_lookalike_commission_host(self, scraper, monkeypatch):
        html = '<a href="https://commission.europa.eu.evil.com/news/article">A long attacker article title</a>'
        monkeypatch.setattr(scraper.requests, "get", MagicMock(return_value=_response(html)))

        links, success = scraper.scrape_news_page(1)

        assert success is True
        assert links == []

    def test_skips_navigation_short_and_non_news_links(self, scraper, monkeypatch):
        html = """
        <a href="/news-and-media/news_en">News listing navigation link</a>
        <a href="/news/">News category index page link</a>
        <a href="/about/commission_en">Not a news article at all</a>
        <a href="/news/short_en">too short</a>
        <a href="news/relative-without-slash_en">Relative link without leading slash</a>
        <a href="/news/kept-article_en">A genuine article title long enough</a>
        """
        monkeypatch.setattr(scraper.requests, "get", MagicMock(return_value=_response(html)))

        links, success = scraper.scrape_news_page(2)

        assert success is True
        assert [link["url"] for link in links] == [scraper.BASE_URL + "/news/kept-article_en"]

    def test_duplicate_urls_are_removed_keeping_first(self, scraper, monkeypatch):
        html = """
        <a href="/news/dup_en">First occurrence of the duplicate</a>
        <a href="/news/dup_en">Second occurrence of the duplicate</a>
        """
        monkeypatch.setattr(scraper.requests, "get", MagicMock(return_value=_response(html)))

        links, _ = scraper.scrape_news_page(1)

        assert len(links) == 1
        assert links[0]["preview_title"] == "First occurrence of the duplicate"

    def test_preview_title_is_truncated_to_100_characters(self, scraper, monkeypatch):
        html = '<a href="/news/long_en">{}</a>'.format("t" * 150)
        monkeypatch.setattr(scraper.requests, "get", MagicMock(return_value=_response(html)))

        links, _ = scraper.scrape_news_page(1)

        assert len(links[0]["preview_title"]) == 100

    def test_requests_the_url_for_the_given_page(self, scraper, monkeypatch):
        get = MagicMock(return_value=_response(""))
        monkeypatch.setattr(scraper.requests, "get", get)

        scraper.scrape_news_page(3)

        assert get.call_args.args[0] == f"{scraper.NEWS_PAGE}?page=2"

    def test_non_200_response_reports_failure(self, scraper, monkeypatch):
        monkeypatch.setattr(scraper.requests, "get",
                            MagicMock(return_value=_response("", status_code=500)))

        assert scraper.scrape_news_page(1) == ([], False)

    def test_request_exception_reports_failure(self, scraper, monkeypatch):
        monkeypatch.setattr(scraper.requests, "get", MagicMock(side_effect=OSError("boom")))

        assert scraper.scrape_news_page(1) == ([], False)


class TestMain:
    @pytest.fixture(autouse=True)
    def _no_sleep(self, scraper, monkeypatch):
        monkeypatch.setattr(scraper.time, "sleep", lambda *_: None)

    def _run_main(self, scraper, monkeypatch, tmp_path, pages, articles):
        output = tmp_path / "out.json"
        monkeypatch.setattr(scraper, "OUTPUT_FILE", str(output))
        monkeypatch.setattr(scraper, "scrape_news_page", MagicMock(side_effect=pages))
        monkeypatch.setattr(scraper, "scrape_article_content",
                            MagicMock(side_effect=lambda url: articles.get(url)))
        return scraper.main(), output

    def test_saves_scraped_articles_to_output_file(self, scraper, monkeypatch, tmp_path):
        import json

        pages = [([{"url": "https://x/news/a", "preview_title": "A"}], True), ([], False)]
        articles = {"https://x/news/a": {
            "title": "Article A", "date": datetime(2026, 6, 20), "content": "body " * 100}}

        result, output = self._run_main(scraper, monkeypatch, tmp_path, pages, articles)

        assert [a["title"] for a in result] == ["Article A"]
        assert result[0]["date"] == "Saturday, 20 June 2026"
        assert result[0]["source"] == "European Commission"
        assert json.loads(output.read_text(encoding="utf-8")) == result

    def test_stops_when_article_predates_cutoff(self, scraper, monkeypatch, tmp_path):
        pages = [([{"url": "https://x/news/old", "preview_title": "old"},
                   {"url": "https://x/news/new", "preview_title": "new"}], True)]
        articles = {
            "https://x/news/old": {"title": "Old", "date": datetime(2020, 1, 1), "content": "b" * 300},
            "https://x/news/new": {"title": "New", "date": datetime(2026, 6, 20), "content": "b" * 300},
        }

        result, _ = self._run_main(scraper, monkeypatch, tmp_path, pages, articles)

        assert result == []

    def test_skips_short_content_and_failed_scrapes(self, scraper, monkeypatch, tmp_path):
        pages = [([{"url": "https://x/news/short", "preview_title": "short"},
                   {"url": "https://x/news/failed", "preview_title": "failed"}], True), ([], False)]
        articles = {
            "https://x/news/short": {"title": "Short", "date": datetime(2026, 6, 20), "content": "tiny"},
            "https://x/news/failed": None,
        }

        result, _ = self._run_main(scraper, monkeypatch, tmp_path, pages, articles)

        assert result == []

    def test_articles_without_a_date_fall_back_to_today(self, scraper, monkeypatch, tmp_path):
        pages = [([{"url": "https://x/news/a", "preview_title": "A"}], True), ([], False)]
        articles = {"https://x/news/a": {"title": "A", "date": None, "content": "b" * 300}}

        result, _ = self._run_main(scraper, monkeypatch, tmp_path, pages, articles)

        assert result[0]["date"] == datetime.now().strftime("%A, %d %B %Y")

    def test_duplicate_titles_across_pages_are_deduplicated(self, scraper, monkeypatch, tmp_path):
        pages = [
            ([{"url": "https://x/news/a", "preview_title": "A"}], True),
            ([{"url": "https://x/news/b", "preview_title": "B"}], True),
            ([], False),
        ]
        articles = {
            "https://x/news/a": {"title": "Same", "date": datetime(2026, 6, 20), "content": "b" * 300},
            "https://x/news/b": {"title": "Same", "date": datetime(2026, 6, 21), "content": "c" * 300},
        }

        result, _ = self._run_main(scraper, monkeypatch, tmp_path, pages, articles)

        assert len(result) == 1
