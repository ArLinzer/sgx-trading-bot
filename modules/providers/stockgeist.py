"""
stockgeist.py — StockGeist sentiment provider.

API docs : https://docs.stockgeist.ai
Auth     : ?token=<key>  (query param)
Endpoints used:
  /stocks/{ticker}/messages/time-series  — real-time + historical sentiment
  /stocks/{ticker}/messages/now          — latest snapshot

Returns CATEGORY="sentiment" — the registry exposes fetch_sentiment() for consumers.
fetch() also returns articles shaped as ProviderArticle so news_fetcher can
include the sentiment headline snippets in the LLM context.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
from loguru import logger

from .base import BaseProvider, ProviderArticle

BASE_URL = "https://api.stockgeist.ai"
TIMEOUT = aiohttp.ClientTimeout(total=15)


class StockGeistProvider(BaseProvider):
    NAME = "StockGeist"
    CATEGORY = "sentiment"

    # ------------------------------------------------------------------
    # fetch() — return sentiment snapshots as ProviderArticle-shaped dicts
    # ------------------------------------------------------------------

    async def fetch(self, ticker: str) -> list[ProviderArticle]:
        """
        Fetch the latest sentiment snapshot and wrap it as a ProviderArticle
        so the LLM receives it as part of the news context.
        """
        if not self._guard():
            logger.debug(f"[{self.NAME}] Skipped (disabled or no API key)")
            return []

        snapshot = await self.fetch_sentiment_now(ticker)
        if not snapshot:
            return []

        score = snapshot.get("score")
        label = self._score_label(score)

        headline = (
            f"{ticker} sentiment ({label}): "
            f"score={score:.2f}, "
            f"pos={snapshot.get('pos_count', 0)} / "
            f"neg={snapshot.get('neg_count', 0)} mentions in last hour"
        )

        return [
            self._article(
                headline=headline,
                summary=f"StockGeist real-time sentiment for {ticker}. "
                        f"Total mentions: {snapshot.get('total_count', 0)}. "
                        f"Sentiment score: {score:.3f} ({label}).",
                published_at=snapshot.get("timestamp", ""),
                sentiment=score,
                ticker=ticker,
                tags=["sentiment", "social"],
            )
        ]

    # ------------------------------------------------------------------
    # Sentiment time series
    # ------------------------------------------------------------------

    async def fetch_sentiment_series(
        self,
        ticker: str,
        hours: int = 24,
        granularity: str = "1h",
    ) -> list[dict[str, Any]]:
        """
        Fetch hourly sentiment time series for the past N hours.

        Returns list of:
          { timestamp, score, pos_count, neg_count, total_count }
        """
        if not self._guard():
            return []

        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(hours=hours)
        params = {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": granularity,
            "token": self.api_key,
        }

        url = f"{BASE_URL}/stocks/{ticker}/messages/time-series"
        logger.debug(f"[{self.NAME}] Fetching sentiment series for {ticker}")
        try:
            async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"[{self.NAME}] HTTP {resp.status} for {ticker}")
                        return []
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning(f"[{self.NAME}] Series request failed for {ticker}: {exc}")
            return []

        series: list[dict[str, Any]] = []
        for point in data.get("data", data if isinstance(data, list) else []):
            series.append(
                {
                    "timestamp": self._normalise_iso(
                        point.get("timestamp") or point.get("time") or ""
                    ),
                    "score": float(point.get("score") or point.get("sentiment") or 0),
                    "pos_count": int(point.get("pos_count") or point.get("positive") or 0),
                    "neg_count": int(point.get("neg_count") or point.get("negative") or 0),
                    "total_count": int(point.get("total_count") or point.get("total") or 0),
                }
            )

        logger.info(f"[{self.NAME}] {len(series)} sentiment data points for {ticker}")
        return series

    # ------------------------------------------------------------------
    # Latest snapshot
    # ------------------------------------------------------------------

    async def fetch_sentiment_now(self, ticker: str) -> dict[str, Any]:
        """
        Fetch the current-moment sentiment snapshot.

        Returns:
          { ticker, score, pos_count, neg_count, total_count, timestamp }
        """
        if not self._guard():
            return {}

        url = f"{BASE_URL}/stocks/{ticker}/messages/now"
        params = {"token": self.api_key}

        try:
            async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"[{self.NAME}] HTTP {resp.status} snapshot for {ticker}")
                        return {}
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning(f"[{self.NAME}] Snapshot request failed for {ticker}: {exc}")
            return {}

        item = data.get("data", data) if isinstance(data, dict) else {}
        if not item:
            return {}

        return {
            "ticker": ticker,
            "score": float(item.get("score") or item.get("sentiment") or 0),
            "pos_count": int(item.get("pos_count") or item.get("positive") or 0),
            "neg_count": int(item.get("neg_count") or item.get("negative") or 0),
            "total_count": int(item.get("total_count") or item.get("total") or 0),
            "timestamp": self._normalise_iso(
                item.get("timestamp") or item.get("time") or ""
            ),
            "source": self.NAME,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_label(score: Any) -> str:
        try:
            f = float(score)
            if f >= 0.3:
                return "bullish"
            if f <= -0.3:
                return "bearish"
            return "neutral"
        except (TypeError, ValueError):
            return "unknown"
