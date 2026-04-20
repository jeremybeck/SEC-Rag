"""
company_matcher.py — spaCy PhraseMatcher-based company and industry extraction.

Given a natural-language query, detects:
  - Company names and common aliases  →  canonical ticker symbols
  - Industry keywords                 →  all tickers in that industry
  - Uppercase ticker symbols          →  canonical ticker symbols (regex fallback)

Public API:
    match_query(query) -> {"tickers": [...], "industries": [...]}

The spaCy model and matchers are loaded lazily on first call and cached
module-level, so subsequent calls are fast.

Requires:
    pip install spacy>=3.7
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TypedDict

# ---------------------------------------------------------------------------
# Company data
# ---------------------------------------------------------------------------

COMPANIES: list[dict] = [
    # Technology
    {"ticker": "AAPL",  "name": "Apple",             "aliases": [],                         "industry": "Technology"},
    {"ticker": "MSFT",  "name": "Microsoft",          "aliases": [],                         "industry": "Technology"},
    {"ticker": "NVDA",  "name": "NVIDIA",             "aliases": ["Nvidia"],                 "industry": "Technology"},
    {"ticker": "AMZN",  "name": "Amazon",             "aliases": [],                         "industry": "Technology"},
    {"ticker": "GOOG",  "name": "Alphabet",           "aliases": ["Google"],                 "industry": "Technology"},
    {"ticker": "META",  "name": "Meta",               "aliases": ["Facebook"],               "industry": "Technology"},
    {"ticker": "AMD",   "name": "AMD",                "aliases": [],                         "industry": "Technology"},
    {"ticker": "INTC",  "name": "Intel",              "aliases": [],                         "industry": "Technology"},
    {"ticker": "CRM",   "name": "Salesforce",         "aliases": [],                         "industry": "Technology"},
    {"ticker": "ADBE",  "name": "Adobe",              "aliases": [],                         "industry": "Technology"},
    {"ticker": "ORCL",  "name": "Oracle",             "aliases": [],                         "industry": "Technology"},
    {"ticker": "CSCO",  "name": "Cisco",              "aliases": [],                         "industry": "Technology"},
    {"ticker": "IBM",   "name": "IBM",                "aliases": [],                         "industry": "Technology"},
    # Automotive
    {"ticker": "TSLA",  "name": "Tesla",              "aliases": [],                         "industry": "Automotive"},
    # Finance
    {"ticker": "JPM",   "name": "JPMorgan Chase",     "aliases": ["JPMorgan", "JP Morgan"],  "industry": "Finance"},
    {"ticker": "BAC",   "name": "Bank of America",    "aliases": [],                         "industry": "Finance"},
    {"ticker": "GS",    "name": "Goldman Sachs",      "aliases": [],                         "industry": "Finance"},
    {"ticker": "MS",    "name": "Morgan Stanley",     "aliases": [],                         "industry": "Finance"},
    {"ticker": "AXP",   "name": "American Express",   "aliases": ["Amex"],                   "industry": "Finance"},
    {"ticker": "BLK",   "name": "BlackRock",          "aliases": [],                         "industry": "Finance"},
    {"ticker": "BRK",   "name": "Berkshire Hathaway", "aliases": ["Berkshire"],              "industry": "Finance"},
    {"ticker": "V",     "name": "Visa",               "aliases": [],                         "industry": "Finance"},
    {"ticker": "MA",    "name": "Mastercard",         "aliases": [],                         "industry": "Finance"},
    # Healthcare
    {"ticker": "JNJ",   "name": "Johnson & Johnson",  "aliases": ["J&J"],                    "industry": "Healthcare"},
    {"ticker": "PFE",   "name": "Pfizer",             "aliases": [],                         "industry": "Healthcare"},
    {"ticker": "MRK",   "name": "Merck",              "aliases": [],                         "industry": "Healthcare"},
    {"ticker": "LLY",   "name": "Eli Lilly",          "aliases": ["Lilly"],                  "industry": "Healthcare"},
    {"ticker": "ABBV",  "name": "AbbVie",             "aliases": [],                         "industry": "Healthcare"},
    {"ticker": "UNH",   "name": "UnitedHealth",       "aliases": ["United Health"],          "industry": "Healthcare"},
    {"ticker": "TMO",   "name": "Thermo Fisher",      "aliases": [],                         "industry": "Healthcare"},
    # Energy
    {"ticker": "XOM",   "name": "ExxonMobil",         "aliases": ["Exxon"],                  "industry": "Energy"},
    {"ticker": "CVX",   "name": "Chevron",            "aliases": [],                         "industry": "Energy"},
    # Consumer Staples
    {"ticker": "KO",    "name": "Coca-Cola",          "aliases": ["Coke"],                   "industry": "Consumer Staples"},
    {"ticker": "PEP",   "name": "PepsiCo",            "aliases": ["Pepsi"],                  "industry": "Consumer Staples"},
    {"ticker": "WMT",   "name": "Walmart",            "aliases": [],                         "industry": "Consumer Staples"},
    {"ticker": "COST",  "name": "Costco",             "aliases": [],                         "industry": "Consumer Staples"},
    {"ticker": "PG",    "name": "Procter & Gamble",   "aliases": ["P&G"],                    "industry": "Consumer Staples"},
    # Consumer Discretionary
    {"ticker": "TGT",   "name": "Target",             "aliases": [],                         "industry": "Consumer Discretionary"},
    {"ticker": "NKE",   "name": "Nike",               "aliases": [],                         "industry": "Consumer Discretionary"},
    {"ticker": "MCD",   "name": "McDonald's",         "aliases": ["McDonalds"],              "industry": "Consumer Discretionary"},
    {"ticker": "SBUX",  "name": "Starbucks",          "aliases": [],                         "industry": "Consumer Discretionary"},
    {"ticker": "HD",    "name": "Home Depot",         "aliases": [],                         "industry": "Consumer Discretionary"},
    # Media
    {"ticker": "DIS",   "name": "Disney",             "aliases": [],                         "industry": "Media"},
    {"ticker": "NFLX",  "name": "Netflix",            "aliases": [],                         "industry": "Media"},
    {"ticker": "CMCSA", "name": "Comcast",            "aliases": [],                         "industry": "Media"},
    # Telecom
    {"ticker": "T",     "name": "AT&T",               "aliases": [],                         "industry": "Telecom"},
    {"ticker": "VZ",    "name": "Verizon",            "aliases": [],                         "industry": "Telecom"},
    # Industrials
    {"ticker": "BA",    "name": "Boeing",             "aliases": [],                         "industry": "Industrials"},
    {"ticker": "CAT",   "name": "Caterpillar",        "aliases": [],                         "industry": "Industrials"},
    {"ticker": "LMT",   "name": "Lockheed Martin",    "aliases": ["Lockheed"],               "industry": "Industrials"},
    {"ticker": "RTX",   "name": "RTX",                "aliases": ["Raytheon"],               "industry": "Industrials"},
    {"ticker": "DE",    "name": "Deere & Company",    "aliases": ["John Deere", "Deere"],    "industry": "Industrials"},
    {"ticker": "UPS",   "name": "UPS",                "aliases": [],                         "industry": "Industrials"},
    {"ticker": "GE",    "name": "GE",                 "aliases": ["General Electric"],       "industry": "Industrials"},
]

# Industry keyword patterns — terms people use in queries that imply an industry
INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "Technology":             ["tech", "technology", "software", "semiconductor", "chip",
                               "cloud computing", "artificial intelligence", "AI"],
    "Healthcare":             ["pharma", "pharmaceutical", "healthcare", "biotech",
                               "biotechnology", "drug maker", "drug company", "medical"],
    "Finance":                ["bank", "banking", "financial", "finance", "investment bank",
                               "fintech", "insurance", "asset management"],
    "Energy":                 ["energy", "oil", "gas", "petroleum", "fossil fuel", "oil and gas"],
    "Consumer Staples":       ["consumer staples", "grocery", "beverage", "household goods"],
    "Consumer Discretionary": ["consumer discretionary", "apparel", "restaurant", "retailer"],
    "Automotive":             ["auto", "automotive", "electric vehicle", "EV", "automaker"],
    "Media":                  ["media", "streaming", "entertainment", "broadcasting"],
    "Telecom":                ["telecom", "telecommunications", "wireless", "broadband"],
    "Industrials":            ["industrial", "aerospace", "defense", "manufacturing",
                               "logistics", "conglomerate"],
}

# ---------------------------------------------------------------------------
# Module-level lazy state
# ---------------------------------------------------------------------------

_nlp = None
_name_matcher = None
_industry_matcher = None
_ticker_to_meta: dict[str, dict] = {}
_industry_to_tickers: dict[str, list[str]] = defaultdict(list)
_known_tickers: set[str] = set()


def _ensure_loaded() -> None:
    """Initialize spaCy model and matchers on first call."""
    global _nlp, _name_matcher, _industry_matcher, _ticker_to_meta, _industry_to_tickers, _known_tickers

    if _nlp is not None:
        return

    import spacy
    from spacy.matcher import PhraseMatcher

    _nlp = spacy.load("en_core_web_sm")
    _name_matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
    _industry_matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")

    for co in COMPANIES:
        ticker = co["ticker"]
        _ticker_to_meta[ticker] = co
        _industry_to_tickers[co["industry"]].append(ticker)
        _known_tickers.add(ticker)

        # Register name + all aliases under this ticker's key
        patterns = [_nlp.make_doc(co["name"])] + [_nlp.make_doc(a) for a in co["aliases"]]
        _name_matcher.add(ticker, patterns)

    for industry, keywords in INDUSTRY_KEYWORDS.items():
        _industry_matcher.add(industry, [_nlp.make_doc(kw) for kw in keywords])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MatchResult(TypedDict):
    tickers: list[str]
    industries: list[str]


def match_query(query: str) -> MatchResult:
    """
    Extract tickers and industries from a natural-language query.

    Detection sources (merged, deduplicated):
      1. Uppercase ticker tokens in the query (e.g. "AAPL", "JPM")
      2. Company names / aliases via PhraseMatcher (e.g. "Apple", "JP Morgan")
      3. Industry keywords via PhraseMatcher (e.g. "pharma" → all Healthcare tickers)

    Returns:
        {
          "tickers":    sorted list of canonical ticker strings,
          "industries": sorted list of industry labels matched by keyword
                        (tickers from these industries are already in "tickers")
        }
    """
    _ensure_loaded()

    found_tickers: set[str] = set()
    found_industries: set[str] = set()

    doc = _nlp(query)

    # 1. Direct uppercase ticker symbols (e.g. "AAPL", "TSLA")
    for token in doc:
        if token.is_upper and token.text in _known_tickers:
            found_tickers.add(token.text)

    # 2. Company name / alias matches
    for match_id, _start, _end in _name_matcher(doc):
        ticker = _nlp.vocab.strings[match_id]
        found_tickers.add(ticker)

    # 3. Industry keyword matches → expand to all tickers in that industry
    for match_id, _start, _end in _industry_matcher(doc):
        industry = _nlp.vocab.strings[match_id]
        found_industries.add(industry)
        found_tickers.update(_industry_to_tickers[industry])

    return {
        "tickers":    sorted(found_tickers),
        "industries": sorted(found_industries),
    }


def tickers_for_industry(industry: str) -> list[str]:
    """Return all tickers belonging to an industry label."""
    _ensure_loaded()
    return sorted(_industry_to_tickers.get(industry, []))


def ticker_metadata(ticker: str) -> dict | None:
    """Return the full company dict for a ticker, or None if not found."""
    _ensure_loaded()
    return _ticker_to_meta.get(ticker)
