"""
finnhub.py — Finnhub provider.

Free plan coverage for SGX:
  - Per-ticker quote (/quote with TICKER:SI): returns zeros — SGX not on free plan
  - Per-ticker news (/company-news): returns empty — SGX not covered
  - General market news (/news?category=general): returns 100 articles ✓

Strategy: Use Finnhub's general news feed, then filter for articles mentioning
the company name or ticker. This gives real financial news without requiring
a paid SGX data subscription.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
from loguru import logger

from .base import BaseProvider, ProviderArticle

BASE_URL = "https://finnhub.io/api/v1"
TIMEOUT = aiohttp.ClientTimeout(total=15)


class FinnhubProvider(BaseProvider):
    NAME = "Finnhub"
    CATEGORY = "news"

    def _headers(self) -> dict[str, str]:
        return {"X-Finnhub-Token": self.api_key}

    # ------------------------------------------------------------------
    # News — fetch general news, filter for ticker/name mentions
    # ------------------------------------------------------------------

    async def fetch(self, ticker: str, name: str = "") -> list[ProviderArticle]:
        if not self._guard():
            return []

        articles: list[ProviderArticle] = []

        # Try per-ticker endpoint first (works on paid plans / some global stocks)
        per_ticker = await self._fetch_company_news(ticker)
        if per_ticker:
            logger.info(f"[{self.NAME}] {len(per_ticker)} per-ticker articles for {ticker}")
            return per_ticker

        # Fall back to general news filtered by ticker/name keyword
        general = await self._fetch_general_news(ticker, name)
        if general:
            logger.info(f"[{self.NAME}] {len(general)} general-news articles for {ticker} (keyword match)")
        else:
            logger.debug(f"[{self.NAME}] No articles found for {ticker}")
        return general

    async def _fetch_company_news(self, ticker: str) -> list[ProviderArticle]:
        """Try the per-ticker endpoint (returns empty for SGX on free plan)."""
        today     = datetime.now(tz=timezone.utc)
        date_from = (today - timedelta(days=2)).strftime("%Y-%m-%d")
        date_to   = today.strftime("%Y-%m-%d")
        fh_symbol = f"{ticker}:SI"

        try:
            async with aiohttp.ClientSession(headers=self._headers(), timeout=TIMEOUT) as session:
                async with session.get(
                    f"{BASE_URL}/company-news",
                    params={"symbol": fh_symbol, "from": date_from, "to": date_to},
                ) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Company news request failed for {ticker}: {exc}")
            return []

        if not isinstance(data, list) or not data:
            return []

        return [
            self._article(
                headline=item.get("headline", ""),
                url=item.get("url", ""),
                summary=item.get("summary", ""),
                published_at=self._normalise_iso(str(item.get("datetime", ""))),
                ticker=ticker,
                tags=[item.get("category", "")],
            )
            for item in data
            if item.get("headline")
        ]

    async def _fetch_general_news(self, ticker: str, name: str = "") -> list[ProviderArticle]:
        """
        Fetch the Finnhub general market news feed and filter by keyword.
        Works on free plan — returns up to 100 recent articles.
        """
        try:
            async with aiohttp.ClientSession(headers=self._headers(), timeout=TIMEOUT) as session:
                async with session.get(
                    f"{BASE_URL}/news",
                    params={"category": "general"},
                ) as resp:
                    if resp.status != 200:
                        logger.debug(f"[{self.NAME}] General news HTTP {resp.status}")
                        return []
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.debug(f"[{self.NAME}] General news request failed: {exc}")
            return []

        if not isinstance(data, list):
            return []

        # Filter for articles that mention the ticker or company name
        keywords = {ticker.lower()}
        if name:
            # Add individual meaningful words from the company name (skip short words)
            keywords.update(w.lower() for w in name.split() if len(w) > 3)

        matched: list[ProviderArticle] = []
        for item in data:
            text = (
                (item.get("headline") or "")
                + " "
                + (item.get("summary") or "")
            ).lower()
            if any(kw in text for kw in keywords):
                matched.append(
                    self._article(
                        headline=item.get("headline", ""),
                        url=item.get("url", ""),
                        summary=item.get("summary", ""),
                        published_at=self._normalise_iso(str(item.get("datetime", ""))),
                        ticker=ticker,
                        tags=[item.get("category", "general")],
                    )
                )

        return matched

    # ------------------------------------------------------------------
    # Quote — returns zeros for SGX on free plan; kept for completeness
    # but callers should prefer Yahoo Finance / SGX API prices
    # ------------------------------------------------------------------

    async def fetch_quote(self, ticker: str) -> dict[str, Any]:
        """
        Attempt a real-time quote from Finnhub.
        NOTE: Returns zeros for SGX stocks on the free plan.
        The caller in main.py falls back to the Yahoo/SGX cache when last_price==0.
        """
        if not self._guard():
            return {}

        fh_symbol = f"{ticker}:SI"
        try:
            async with aiohttp.ClientSession(headers=self._headers(), timeout=TIMEOUT) as session:
                async with session.get(
                    f"{BASE_URL}/quote", params={"symbol": fh_symbol}
                ) as resp:
                    if resp.status != 200:
                        return {}
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Quote request failed for {ticker}: {exc}")
            return {}

        # c=0 means no data — return empty so caller falls back to Yahoo
        close = float(data.get("c") or 0)
        if close == 0:
            logger.debug(f"[{self.NAME}] Quote for {ticker} returned zeros — not covered on free plan")
            return {}

        return {
            "ticker":     ticker,
            "last_price": close,
            "open":       float(data.get("o") or 0),
            "high":       float(data.get("h") or 0),
            "low":        float(data.get("l") or 0),
            "prev_close": float(data.get("pc") or 0),
            "change":     float(data.get("d") or 0),
            "change_pct": float(data.get("dp") or 0),
            "timestamp":  data.get("t", ""),
            "source":     self.NAME,
        }
