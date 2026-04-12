"""
news_fetcher.py — Multi-source news aggregator for SGX stocks.

Built-in free sources (always active, no API key needed):
  1. Google News RSS           — company name search
  2. Business Times Singapore  — search page scrape
  3. Business Times RSS        — structured RSS feeds (Companies & Markets,
                                 Banking & Finance, All News) filtered by name
  4. The Straits Times         — scrape
  5. SGX Announcements API     — official filings via api2.sgx.com
  6. Yahoo Finance news        — scrape (ticker.SI)

External API providers (enable in config.yaml with API key):
  7.  Marketaux    — financial news + per-entity sentiment
  8.  StockNewsAPI — stock/financial news feed
  9.  EODHD        — financial news + sentiment, keyword search
  10. Finnhub      — market news, keyword filter
  11. StockGeist   — sentiment signals

All sources run concurrently. Results are deduplicated by headline similarity
(rapidfuzz) and filtered to the last 24 hours.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import quote_plus

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from loguru import logger
from rapidfuzz import fuzz
from tenacity import retry, stop_after_attempt, wait_exponential

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}
TIMEOUT           = aiohttp.ClientTimeout(total=20)
DEDUP_THRESHOLD   = 85   # rapidfuzz score — headlines more similar than this are dupes
MAX_AGE_HOURS     = 24

# Business Times RSS feeds — all public, no auth required
BT_RSS_FEEDS: list[tuple[str, str]] = [
    ("Companies & Markets", "https://www.businesstimes.com.sg/rss/companies-markets"),
    ("Banking & Finance",   "https://www.businesstimes.com.sg/rss/banking-finance"),
    ("All News",            "https://www.businesstimes.com.sg/rss/all-news"),
]


class NewsFetcher:
    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        provider_registry: Optional[Any] = None,
    ):
        """
        Args:
            rate_limit_delay:  Seconds to wait between requests to the same domain.
            provider_registry: Optional ProviderRegistry. When supplied, all enabled
                               API providers are queried alongside the free scrapers.
        """
        self.rate_limit_delay = rate_limit_delay
        self.registry = provider_registry
        self._domain_last_hit: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def fetch_all(
        self,
        ticker: str,
        name: str = "",
    ) -> list[dict[str, Any]]:
        """
        Fetch and deduplicate news for a single SGX ticker from all sources.

        Args:
            ticker: SGX ticker code, e.g. "D05"
            name:   Company name, e.g. "DBS Group" — improves search quality
                    on Google News, Business Times, Straits Times, and all API
                    providers that support keyword/name-based queries.

        Returns:
            Deduplicated list of news dicts sorted newest-first, last 24h only.
        """
        # Use company name as primary search term where possible — gives far
        # better results than the bare ticker code on news sites.
        search_term = name if name else ticker

        # ── Free scraped sources (always run) ────────────────────────────
        scrape_tasks = [
            self._fetch_google_news(search_term),
            self._fetch_business_times(search_term),
            self._fetch_business_times_rss(ticker, name),  # structured RSS feeds
            self._fetch_straits_times(search_term),
            self._fetch_sgx_announcements(ticker),   # SGX API uses bare ticker
            self._fetch_yahoo_finance(ticker),        # Yahoo uses ticker.SI
        ]
        scrape_names = [
            "Google News", "Business Times", "Business Times RSS",
            "Straits Times", "SGX Announcements", "Yahoo Finance",
        ]

        # ── External API providers (concurrent with scraping) ────────────
        async def _no_providers() -> list:
            return []

        provider_task = (
            self.registry.fetch_news(ticker, name=name)
            if self.registry else _no_providers()
        )

        scrape_results, provider_articles = await asyncio.gather(
            asyncio.gather(*scrape_tasks, return_exceptions=True),
            provider_task,
            return_exceptions=True,
        )

        all_articles: list[dict[str, Any]] = []

        # Collect scraped results
        if isinstance(scrape_results, list):
            for src_name, result in zip(scrape_names, scrape_results):
                if isinstance(result, Exception):
                    logger.warning(f"[{ticker}] {src_name} fetch error: {result}")
                elif result:
                    logger.debug(f"[{ticker}] {src_name}: {len(result)} articles")
                    all_articles.extend(result)

        # Collect provider results
        if isinstance(provider_articles, list):
            all_articles.extend(provider_articles)
        elif isinstance(provider_articles, Exception):
            logger.warning(f"[{ticker}] Provider registry error: {provider_articles}")

        # ── Filter to last 24 hours ───────────────────────────────────────
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=MAX_AGE_HOURS)
        recent = [a for a in all_articles if self._is_recent(a, cutoff)]

        # ── Relevance filter — remove off-topic articles ──────────────────
        # SGX Announcements and Yahoo Finance are already ticker-specific; every
        # other source (Google News, BT search/RSS, ST, API providers) can return
        # general-market articles.  Keep an article only when its headline+summary
        # contains at least one identifier for this stock.
        relevant = [
            a for a in recent
            if self._is_relevant_to_stock(a, ticker, name)
        ]
        dropped = len(recent) - len(relevant)
        if dropped:
            logger.info(
                f"[{ticker}] Relevance filter dropped {dropped} off-topic articles "
                f"({len(relevant)} remain)"
            )

        # ── Deduplicate by headline similarity ───────────────────────────
        deduped = self._deduplicate(relevant)

        # ── Sort newest-first ─────────────────────────────────────────────
        deduped.sort(key=lambda x: x.get("published_at") or "", reverse=True)

        # ── Source breakdown log ──────────────────────────────────────────
        source_counts: dict[str, int] = {}
        for a in deduped:
            src = a.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        breakdown = " | ".join(f"{k}:{v}" for k, v in sorted(source_counts.items()))

        logger.info(
            f"[{ticker}] News: {len(all_articles)} raw → "
            f"{len(recent)} recent → {len(relevant)} relevant → {len(deduped)} after dedup"
            + (f" | {breakdown}" if breakdown else "")
        )
        return deduped

    # ------------------------------------------------------------------
    # Source 1 — Google News RSS
    # ------------------------------------------------------------------

    async def _fetch_google_news(self, search_term: str) -> list[dict[str, Any]]:
        query = quote_plus(f"{search_term} SGX")
        url   = f"https://news.google.com/rss/search?q={query}&hl=en-SG&gl=SG&ceid=SG:en"
        raw   = await self._get_text(url)
        if not raw:
            return []

        feed = feedparser.parse(raw)
        articles: list[dict[str, Any]] = []
        for entry in feed.entries:
            articles.append({
                "source":       "Google News",
                "headline":     entry.get("title", ""),
                "summary":      BeautifulSoup(
                    entry.get("summary", ""), "html.parser"
                ).get_text(" ", strip=True)[:500],
                "url":          entry.get("link", ""),
                "published_at": self._parse_rss_date(entry.get("published", "")),
            })
        return articles

    # ------------------------------------------------------------------
    # Source 2 — Business Times Singapore
    # ------------------------------------------------------------------

    async def _fetch_business_times(self, search_term: str) -> list[dict[str, Any]]:
        url  = f"https://www.businesstimes.com.sg/search?q={quote_plus(search_term)}"
        html = await self._get_text(url, domain="businesstimes.com.sg")
        if not html:
            return []
        return self._parse_bt_html(html)

    def _parse_bt_html(self, html: str) -> list[dict[str, Any]]:
        soup     = BeautifulSoup(html, "html.parser")
        articles: list[dict[str, Any]] = []
        for item in soup.select("article, .search-result-item, .media"):
            headline_el = item.find(["h2", "h3", "h4", "a"])
            headline    = headline_el.get_text(strip=True) if headline_el else ""
            if not headline:
                continue
            link_el = item.find("a", href=True)
            url     = link_el["href"] if link_el else ""
            if url and not url.startswith("http"):
                url = "https://www.businesstimes.com.sg" + url
            summary_el  = item.find("p")
            summary     = summary_el.get_text(strip=True)[:500] if summary_el else ""
            date_el     = item.find(["time", "span"], class_=re.compile(r"date|time|published"))
            published_at = self._parse_html_date(date_el)
            articles.append({
                "source":       "Business Times",
                "headline":     headline,
                "summary":      summary,
                "url":          url,
                "published_at": published_at,
            })
        return articles

    # ------------------------------------------------------------------
    # Source 3 — Business Times RSS feeds (Companies & Markets, etc.)
    # ------------------------------------------------------------------

    # Generic words too common to use as standalone keywords
    _CORPORATE_STOPWORDS: frozenset[str] = frozenset({
        "group", "holdings", "holding", "limited", "corporation",
        "corp", "incorporated", "trust", "reit", "fund", "capital",
        "partners", "partner", "ventures", "venture", "industries",
        "industry", "international", "global", "asia", "pacific",
        "singapore", "services", "technology", "technologies",
        "investment", "investments", "management", "solutions",
        "properties", "property", "real", "estate",
        "bank", "banks", "banking", "finance", "financial",
    })

    @staticmethod
    def _bt_rss_keywords(ticker: str, name: str) -> tuple[list[str], list[str]]:
        """
        Return (must_match_all, any_one_suffices) keyword lists for BT RSS
        article filtering.

        Strategy:
          • Uppercase abbreviations in the name (≥3 chars, e.g. "DBS", "OCBC")
            and the bare ticker are "strong" identifiers — a single match is
            enough to include an article.
          • Ordinary words ≥5 chars that are not corporate stopwords are
            collected as "context" keywords.  When two or more context keywords
            are found, ALL of them must appear in the article (AND logic),
            which avoids false positives from common words like "united".
          • If no context keywords survive, the strong identifiers alone are
            used.
        """
        stopwords = NewsFetcher._CORPORATE_STOPWORDS
        tokens    = re.split(r"\W+", name)

        # Strong identifiers: all-uppercase tokens ≥3 chars (abbreviations)
        strong = [t for t in tokens if t == t.upper() and len(t) >= 3 and t.isalpha()]
        strong_lower = [s.lower() for s in strong] + [ticker.lower()]

        # Context keywords: mixed-case words ≥5 chars, not stopwords
        context = [
            t.lower() for t in tokens
            if len(t) >= 5
            and t.lower() not in stopwords
            and t != t.upper()   # skip abbreviations already in strong
        ]

        return strong_lower, context

    async def _fetch_business_times_rss(
        self, ticker: str, name: str
    ) -> list[dict[str, Any]]:
        """
        Fetch all BT_RSS_FEEDS concurrently, then keep only articles that
        mention the company.  Matching rules (evaluated in order):

          1. Any strong identifier (ticker or uppercase abbreviation like "DBS",
             "OCBC") appears in headline+summary → include.
          2. If the name yields ≥2 context keywords (ordinary words ≥5 chars,
             not corporate suffixes), ALL must be present → include.
          3. If only 1 context keyword, it alone is sufficient.

        This avoids false positives from generic words (e.g. "bank", "group")
        while still catching tickers whose full name is only stop-words.

        Returns merged, deduplicated list.
        """
        strong, context = self._bt_rss_keywords(ticker, name)
        logger.debug(
            f"[BT RSS] {ticker} keywords — strong={strong} context={context}"
        )

        tasks = [
            self._fetch_bt_rss_feed(label, url)
            for label, url in BT_RSS_FEEDS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen_urls: set[str] = set()
        articles: list[dict[str, Any]] = []

        for label, result in zip([lbl for lbl, _ in BT_RSS_FEEDS], results):
            if isinstance(result, Exception):
                logger.debug(f"[BT RSS] {label} error: {result}")
                continue
            for article in result:
                url = article.get("url", "")
                if url in seen_urls:
                    continue

                text = (
                    (article.get("headline") or "") + " " +
                    (article.get("summary")  or "")
                ).lower()

                # Rule 1: strong identifier match (single keyword sufficient)
                if any(sid in text for sid in strong):
                    seen_urls.add(url)
                    articles.append(article)
                    continue

                # Rule 2 / 3: context keyword match
                if context:
                    matched = (
                        all(kw in text for kw in context)  # AND when ≥2 keywords
                        if len(context) >= 2
                        else context[0] in text             # OR when only 1
                    )
                    if matched:
                        seen_urls.add(url)
                        articles.append(article)

        logger.debug(
            f"[BT RSS] {ticker}: {len(articles)} articles matched across "
            f"{len(BT_RSS_FEEDS)} feeds"
        )
        return articles

    async def _fetch_bt_rss_feed(
        self, label: str, url: str
    ) -> list[dict[str, Any]]:
        """Fetch a single BT RSS feed URL and parse entries."""
        raw = await self._get_text(url, domain="businesstimes.com.sg")
        if not raw:
            return []

        feed     = feedparser.parse(raw)
        articles: list[dict[str, Any]] = []
        for entry in feed.entries:
            summary_html = entry.get("summary", "") or entry.get("description", "")
            summary      = BeautifulSoup(summary_html, "html.parser").get_text(
                " ", strip=True
            )[:500]
            articles.append({
                "source":       f"Business Times RSS ({label})",
                "headline":     entry.get("title", ""),
                "summary":      summary,
                "url":          entry.get("link", ""),
                "published_at": self._parse_rss_date(entry.get("published", "")),
            })
        return articles

    # ------------------------------------------------------------------
    # Source 4 — The Straits Times
    # ------------------------------------------------------------------

    async def _fetch_straits_times(self, search_term: str) -> list[dict[str, Any]]:
        url  = f"https://www.straitstimes.com/search/{quote_plus(search_term)}"
        html = await self._get_text(url, domain="straitstimes.com")
        if not html:
            return []
        return self._parse_st_html(html)

    def _parse_st_html(self, html: str) -> list[dict[str, Any]]:
        soup     = BeautifulSoup(html, "html.parser")
        articles: list[dict[str, Any]] = []
        for item in soup.select(".search-result-item, article, .card"):
            headline_el = item.find(["h2", "h3", "h4", "a"])
            headline    = headline_el.get_text(strip=True) if headline_el else ""
            if not headline:
                continue
            link_el = item.find("a", href=True)
            url     = link_el["href"] if link_el else ""
            if url and not url.startswith("http"):
                url = "https://www.straitstimes.com" + url
            summary_el  = item.find("p")
            summary     = summary_el.get_text(strip=True)[:500] if summary_el else ""
            date_el     = item.find(["time", "span"], class_=re.compile(r"date|time|published"))
            published_at = self._parse_html_date(date_el)
            articles.append({
                "source":       "Straits Times",
                "headline":     headline,
                "summary":      summary,
                "url":          url,
                "published_at": published_at,
            })
        return articles

    # ------------------------------------------------------------------
    # Source 5 — SGX Announcements API
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def _fetch_sgx_announcements(self, ticker: str) -> list[dict[str, Any]]:
        url = f"https://api2.sgx.com/announcements?code={ticker}"
        try:
            async with aiohttp.ClientSession(headers=HEADERS, timeout=TIMEOUT) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.debug(f"SGX announcements fetch failed for {ticker}: {exc}")
            return []

        articles: list[dict[str, Any]] = []
        items = data if isinstance(data, list) else data.get("data", data.get("announcements", []))
        for item in (items if isinstance(items, list) else []):
            headline = (
                item.get("headline")
                or item.get("title")
                or item.get("subject")
                or ""
            )
            if not headline:
                continue
            articles.append({
                "source":       "SGX Announcements",
                "headline":     headline,
                "summary":      item.get("summary") or item.get("description") or "",
                "url":          item.get("url") or item.get("link") or "",
                "published_at": self._normalise_iso(
                    item.get("publishedAt")
                    or item.get("date")
                    or item.get("announcementDate")
                    or ""
                ),
            })
        return articles

    # ------------------------------------------------------------------
    # Source 6 — Yahoo Finance news
    # ------------------------------------------------------------------

    async def _fetch_yahoo_finance(self, ticker: str) -> list[dict[str, Any]]:
        yf_ticker = f"{ticker}.SI"
        url       = f"https://finance.yahoo.com/quote/{yf_ticker}/news/"
        html      = await self._get_text(url, domain="finance.yahoo.com")
        if not html:
            return []
        return self._parse_yahoo_html(html)

    def _parse_yahoo_html(self, html: str) -> list[dict[str, Any]]:
        soup     = BeautifulSoup(html, "html.parser")
        articles: list[dict[str, Any]] = []
        for item in soup.select("li[class*='stream-item'], div[class*='Ov(h)'], article"):
            headline_el = item.find(["h3", "h2", "a"])
            headline    = headline_el.get_text(strip=True) if headline_el else ""
            if not headline:
                continue
            link_el = item.find("a", href=True)
            url     = link_el["href"] if link_el else ""
            if url and url.startswith("/"):
                url = "https://finance.yahoo.com" + url
            summary_el  = item.find("p")
            summary     = summary_el.get_text(strip=True)[:500] if summary_el else ""
            date_el     = item.find("time")
            published_at = self._parse_html_date(date_el)
            articles.append({
                "source":       "Yahoo Finance",
                "headline":     headline,
                "summary":      summary,
                "url":          url,
                "published_at": published_at,
            })
        return articles

    # ------------------------------------------------------------------
    # Stock-relevance filter
    # ------------------------------------------------------------------

    @staticmethod
    def _is_relevant_to_stock(
        article: dict[str, Any],
        ticker: str,
        name: str,
    ) -> bool:
        """
        Return True if the article is likely about this specific stock.

        Sources that are already ticker-specific (SGX Announcements, Yahoo
        Finance, StockGeist) are always passed through.  For all other sources
        we match against:
          1. The bare ticker code (e.g. "D05")
          2. Strong abbreviations from the company name (all-caps ≥3 chars,
             e.g. "DBS", "OCBC", "SIA")
          3. Context keywords (meaningful words ≥5 chars, not corporate
             stop-words) — ALL must match when there are ≥2 of them

        This prevents generic Straits Times / Google News / Business Times
        articles from reaching the LLM prompt.
        """
        # Always pass ticker-specific sources
        source = (article.get("source") or "").lower()
        if any(s in source for s in ("sgx announcements", "yahoo finance", "stockgeist")):
            return True

        text = (
            (article.get("headline") or "") + " " +
            (article.get("summary")  or "")
        ).lower()

        # 1. Bare ticker match (word-boundary)
        if re.search(rf"\b{re.escape(ticker.lower())}\b", text):
            return True

        # 2 & 3. Name-derived keywords (reuse existing BT RSS logic)
        strong, context = NewsFetcher._bt_rss_keywords(ticker, name)

        if any(sid in text for sid in strong):
            return True

        if context:
            if len(context) >= 2:
                if all(kw in text for kw in context):
                    return True
            else:
                if context[0] in text:
                    return True

        return False

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove articles whose headline is too similar to an already-kept one."""
        kept: list[dict[str, Any]] = []
        for article in articles:
            headline = article.get("headline", "")
            is_dup   = any(
                fuzz.token_sort_ratio(headline, k.get("headline", "")) >= DEDUP_THRESHOLD
                for k in kept
            )
            if not is_dup:
                kept.append(article)
        return kept

    # ------------------------------------------------------------------
    # HTTP helper with per-domain rate limiting
    # ------------------------------------------------------------------

    async def _get_text(self, url: str, domain: Optional[str] = None) -> Optional[str]:
        import time as _time

        if domain:
            last = self._domain_last_hit.get(domain, 0)
            gap  = _time.monotonic() - last
            if gap < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - gap)
            self._domain_last_hit[domain] = _time.monotonic()

        for attempt in range(3):
            try:
                async with aiohttp.ClientSession(headers=HEADERS, timeout=TIMEOUT) as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            return await resp.text()
                        logger.debug(f"HTTP {resp.status} for {url}")
                        return None
            except asyncio.TimeoutError:
                logger.debug(f"Timeout fetching {url} (attempt {attempt + 1})")
            except Exception as exc:
                logger.debug(f"Error fetching {url}: {exc}")
            await asyncio.sleep(2 ** attempt)
        return None

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_rss_date(date_str: str) -> str:
        if not date_str:
            return ""
        import email.utils
        try:
            parsed = email.utils.parsedate_to_datetime(date_str)
            return parsed.astimezone(timezone.utc).isoformat()
        except Exception:
            return date_str

    @staticmethod
    def _parse_html_date(el: Any) -> str:
        if el is None:
            return ""
        dt_attr = el.get("datetime") or el.get("data-timestamp") or ""
        if dt_attr:
            return NewsFetcher._normalise_iso(dt_attr)
        return NewsFetcher._normalise_iso(el.get_text(strip=True))

    @staticmethod
    def _normalise_iso(value: str) -> str:
        if not value:
            return ""
        try:
            from dateutil import parser as dp
            parsed = dp.parse(value, fuzzy=True)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        except Exception:
            return value

    @staticmethod
    def _is_recent(article: dict[str, Any], cutoff: datetime) -> bool:
        pub = article.get("published_at", "")
        if not pub:
            return True   # no date → include rather than discard
        try:
            from dateutil import parser as dp
            parsed = dp.parse(pub)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed >= cutoff
        except Exception:
            return True


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _main() -> None:
        fetcher = NewsFetcher()
        news = await fetcher.fetch_all("D05", name="DBS Group")
        print(f"Fetched {len(news)} articles for D05 (DBS Group)")
        for n in news[:5]:
            print(f"  [{n['source']}] {n['headline'][:70]}")

    asyncio.run(_main())
