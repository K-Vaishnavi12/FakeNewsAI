"""Tests for the source-credibility scoring module."""

import pytest

from server.credibility import (
    CREDIBLE_THRESHOLD,
    _contains_word,
    _domain_from_url,
    get_source_credibility,
)


def test_domain_from_url_strips_www():
    assert _domain_from_url("https://www.reuters.com/article/1") == "reuters.com"
    assert _domain_from_url("https://bbc.co.uk/news") == "bbc.co.uk"
    assert _domain_from_url("not a url") == ""


def test_known_high_credibility_source_by_name():
    res = get_source_credibility("Reuters")
    assert res["score"] == 0.95
    assert res["tier"] == "high"
    assert res["is_fact_checker"] is False


def test_high_credibility_matches_case_and_whitespace():
    res = get_source_credibility("  THE NEW YORK TIMES ")
    assert res["tier"] == "high"


def test_fact_checker_scored_highest():
    res = get_source_credibility("Snopes")
    assert res["score"] == 1.0
    assert res["tier"] == "factchecker"
    assert res["is_fact_checker"] is True


def test_unknown_source_defaults_to_neutral():
    res = get_source_credibility("Some Random Blog", "https://randomblog.example/x")
    assert res["tier"] == "unknown"
    assert res["score"] == 0.5


def test_domain_matching_catches_spoofed_name():
    """A fake site using a trusted brand in its display name is caught by URL."""
    res = get_source_credibility("Reuters Breaking News",
                                 "https://reuters-breaking-news.example/story")
    # "reuters-breaking-news.example" is not an exact domain match, so the
    # spoofed-but-not-exact name must NOT score as high credibility.
    assert res["tier"] == "unknown"


def test_domain_match_alone_scores_source():
    res = get_source_credibility(None, "https://www.bbc.co.uk/article/1")
    assert res["tier"] == "high"
    assert res["matched_on"] == "bbc.co.uk"


def test_substring_match_prefers_longest_token():
    # "the new york times" must win over the bare "times" token.
    assert _contains_word("the new york times", "times") is True
    assert _contains_word("times of india", "new york times") is False


def test_low_credibility_source_flagged():
    res = get_source_credibility("Infowars", "https://www.infowars.com/x")
    assert res["tier"] == "low"
    assert res["score"] < CREDIBLE_THRESHOLD


def test_dict_source_shape_supported():
    res = get_source_credibility({"name": "NPR"}, "https://www.npr.org/a")
    assert res["tier"] == "high"