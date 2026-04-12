"""
stocknewsapi.py — StockNewsAPI provider.

API docs : https://stocknewsapi.com/documentation
Auth     : ?token=<key>  (query param)
Endpoint : https://stocknewsapi.com/api/v1
Params   : tickers, items, date (MMDDYYYY), token
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import aiohttp
from loguru import logger

from .base import BaseProvider, ProviderArticle

API_URL = "https://stocknewsapi.com/api/v1"
TIMEOUT = aiohttp.ClientTimeout(total=15)


class StockNewsAPIProvider(BaseProvider):
    NAME = "StockNewsAPI"
    CATEGORY = "news"

    async def fetch(self, ticker: str) -> list[ProviderArticle]:
        if not self._guard():
            logger.debug(f"[{self.NAME}] Skipped (disabled or no API key)")
            return []

        # StockNewsAPI accepts raw tickers; try both bare and .SI forms
        params = {
            "tickers": f"{ticker}.SI,{ticker}",
            "items": str(self._cfg.get("items", 20)),
            "token": self.api_key,
        }

        # Optional date filter — restrict to today
        date_filter = datetime.now(tz=timezone.utc).strftime("%m%d%Y")
        params["date"] = date_filter

        logger.debug(f"[{self.NAME}] Fetching news for {ticker}")
        try:
            async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
                async with session.get(API_URL, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"[{self.NAME}] HTTP {resp.status} for {ticker}")
                        return []
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning(f"[{self.NAME}] Request failed for {ticker}: {exc}")
            return []

        articles: list[ProviderArticle] = []
        for item in data.get("data", []):
            articles.append(
                self._article(
                    headline=item.get("title", ""),
                    url=item.get("news_url", ""),
                    summary=item.get("text", ""),
                    published_at=self._normalise_iso(item.get("date", "")),
                    sentiment=self._map_sentiment(item.get("sentiment")),
                    ticker=ticker,
                    tags=item.get("topics", []),
                )
            )

        logger.info(f"[{self.NAME}] {len(articles)} articles for {ticker}")
        return articles

    @staticmethod
    def _map_sentiment(value: Any) -> float | None:
        """Map StockNewsAPI sentiment string to float."""
        mapping = {"Positive": 0.6, "Negative": -0.6, "Neutral": 0.0}
        if isinstance(value, str):
            return mapping.get(value)
        if isinstance(value, (int, float)):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
        return None
