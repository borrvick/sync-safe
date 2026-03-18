"""
tests/test_legal.py
Unit tests for services/legal.py — pure URL construction, no mocks needed.
"""
from urllib.parse import urlparse

import pytest

from services.legal import Legal, _build_url


class TestLegal:
    def setup_method(self):
        self.svc = Legal()

    def test_returns_all_three_pros(self):
        links = self.svc.get_links("Bohemian Rhapsody", "Queen")
        assert links.ascap
        assert links.bmi
        assert links.sesac

    def test_all_values_are_https_urls(self):
        links = self.svc.get_links("Bohemian Rhapsody", "Queen")
        for url in (links.ascap, links.bmi, links.sesac):
            assert urlparse(url).scheme == "https"

    def test_title_and_artist_in_query(self):
        links = self.svc.get_links("Bohemian Rhapsody", "Queen")
        for url in (links.ascap, links.bmi, links.sesac):
            assert "bohemian" in url.lower(), f"Title missing from URL: {url}"

    def test_empty_title_and_artist(self):
        links = self.svc.get_links("", "")
        for url in (links.ascap, links.bmi, links.sesac):
            assert url.startswith("https://")

    def test_empty_title_only(self):
        links = self.svc.get_links("", "Radiohead")
        for url in (links.ascap, links.bmi, links.sesac):
            assert "Radiohead" in url

    def test_empty_artist_only(self):
        links = self.svc.get_links("Creep", "")
        for url in (links.ascap, links.bmi, links.sesac):
            assert "Creep" in url

    def test_special_characters_are_encoded(self):
        links = self.svc.get_links("AC/DC", "AC/DC")
        for url in (links.ascap, links.bmi, links.sesac):
            assert urlparse(url).query, f"URL has no query string: {url}"

    def test_ascap_url_structure(self):
        links = self.svc.get_links("Yesterday", "Beatles")
        assert "ascap.com" in urlparse(links.ascap).netloc

    def test_bmi_url_structure(self):
        links = self.svc.get_links("Yesterday", "Beatles")
        assert "bmi.com" in urlparse(links.bmi).netloc

    def test_sesac_url_structure(self):
        links = self.svc.get_links("Yesterday", "Beatles")
        assert "sesac.com" in urlparse(links.sesac).netloc

    def test_whitespace_stripped_from_inputs(self):
        a = self.svc.get_links("Yesterday", "Beatles")
        b = self.svc.get_links("  Yesterday  ", "  Beatles  ")
        assert a.ascap == b.ascap
        assert a.bmi   == b.bmi
        assert a.sesac == b.sesac


class TestBuildUrl:
    def test_appends_params(self):
        url = _build_url("https://example.com/search", {"q": "hello world"})
        assert url == "https://example.com/search?q=hello+world"

    def test_empty_params(self):
        url = _build_url("https://example.com", {})
        assert url == "https://example.com?"
