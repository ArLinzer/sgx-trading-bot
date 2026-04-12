"""
registry.py — Provider registry and aggregation manager.

Responsibilities:
  - Instantiate every provider from config
  - Expose enable/disable toggles at runtime
  - Aggregate fetch() results from all enabled news providers
  - Expose dedicated fetch_ohlcv() and fetch_sentiment() routers

Usage:
    registry = ProviderRegistry(cfg["providers"])
    articles = await registry.fetch_news(ticker="D05")
    bars     = await registry.fetch_ohlcv(ticker="D05")
    score    = await registry.fetch_sentiment(ticker="D05")
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from .base import BaseProvider, ProviderArticle
from .eodhd import EODHDProvider
from .finnhub import FinnhubProvider
from .marketaux import MarketauxProvider
from .marketstack import MarketstackProvider
from .stockgeist import StockGeistProvider
from .stocknewsapi import StockNewsAPIProvider

# Registry of all known provider classes keyed by config block name
_PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "marketaux":    MarketauxProvider,
    "stocknewsapi": StockNewsAPIProvider,
    "eodhd":        EODHDProvider,
    "marketstack":  MarketstackProvider,
    "finnhub":      FinnhubProvider,
    "stockgeist":   StockGeistProvider,
}


class ProviderRegistry:
    def __init__(self, providers_cfg: dict[str, Any]):
        """
        Args:
            providers_cfg: The ``providers:`` block from config.yaml.
                           Each key is a provider name; each value is its config dict.
        """
        self._providers: dict[str, BaseProvider] = {}

        for name, cls in _PROVIDER_CLASSES.items():
            block = providers_cfg.get(name, {})
            instance = cls(block)
            self._providers[name] = instance
            status = "ENABLED" if instance.enabled else "disabled"
            logger.info(f"[Registry] {name:14s} → {status}")

    # ------------------------------------------------------------------
    # Runtime toggle
    # ------------------------------------------------------------------

    def enable(self, name: str) -> None:
        """Enable a provider by name at runtime."""
        p = self._providers.get(name)
        if p:
            p.enabled = True
            logger.info(f"[Registry] {name} ENABLED")
        else:
            logger.warning(f"[Registry] Unknown provider: {name}")

    def disable(self, name: str) -> None:
        """Disable a provider by name at runtime."""
        p = self._providers.get(name)
        if p:
            p.enabled = False
            logger.info(f"[Registry] {name} DISABLED")
        else:
            logger.warning(f"[Registry] Unknown provider: {name}")

    def status(self) -> dict[str, bool]:
        """Return a name → enabled mapping for all providers."""
        return {name: p.enabled for name, p in self._providers.items()}

    def get(self, name: str) -> BaseProvider | None:
        return self._providers.get(name)

    # ------------------------------------------------------------------
    # Aggregated news fetch
    # ------------------------------------------------------------------

    async def fetch_news(self, ticker: str, name: str = "") -> list[ProviderArticle]:
        """
        Fetch news from all enabled news providers concurrently.

        Args:
            ticker: SGX stock code, e.g. "D05"
            name:   Company name, e.g. "DBS Group" — passed to providers that
                    do keyword searches (Finnhub general news, EODHD tag search)

        Returns:
            Combined, unsorted list of ProviderArticle dicts.
            Deduplication is handled by NewsFetcher.
        """
        news_providers = [
            p for p in self._providers.values()
            if p.enabled and p.CATEGORY == "news"
        ]

        if not news_providers:
            logger.debug(f"[Registry] No news providers enabled for {ticker}")
            return []

        names = [p.NAME for p in news_providers]
        logger.info(f"[Registry] Fetching news for {ticker} ({name or '?'}) from: {', '.join(names)}")

        # Pass name= to providers that accept it (Finnhub, EODHD keyword search)
        tasks = []
        for p in news_providers:
            import inspect
            sig = inspect.signature(p.fetch)
            if "name" in sig.parameters:
                tasks.append(p.fetch(ticker, name=name))
            else:
                tasks.append(p.fetch(ticker))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        articles: list[ProviderArticle] = []
        for provider, result in zip(news_providers, results):
            if isinstance(result, Exception):
                logger.warning(f"[Registry] {provider.NAME} raised: {result}")
            elif isinstance(result, list):
                articles.extend(result)
                logger.debug(f"[Registry] {provider.NAME} → {len(result)} articles")

        logger.info(f"[Registry] {ticker}: {len(articles)} total articles from {len(news_providers)} providers")
        return articles

    # ------------------------------------------------------------------
    # OHLCV fetch (Marketstack, or fallback to any enabled ohlcv provider)
    # ------------------------------------------------------------------

    async def fetch_ohlcv(self, ticker: str, days: int = 5) -> list[dict[str, Any]]:
        """
        Fetch OHLCV bars. Priority:
          1. Marketstack (or any enabled ohlcv provider)
          2. Yahoo Finance via yfinance (always available, no API key needed)

        Returns list of OHLCV bar dicts sorted oldest-first, or [] on failure.
        """
        ohlcv_providers = [
            p for p in self._providers.values()
            if p.enabled and p.CATEGORY == "ohlcv"
        ]

        # Try registered providers first
        for provider in ohlcv_providers:
            try:
                if hasattr(provider, "fetch_ohlcv"):
                    bars = await provider.fetch_ohlcv(ticker, days=days)  # type: ignore[attr-defined]
                else:
                    bars = await provider.fetch(ticker)
                if bars:
                    logger.info(f"[Registry] OHLCV for {ticker} from {provider.NAME}: {len(bars)} bars")
                    return bars
            except Exception as exc:
                logger.warning(f"[Registry] OHLCV {provider.NAME} failed for {ticker}: {exc}")

        # Fallback: Yahoo Finance via yfinance (no API key, always works)
        logger.info(f"[Registry] OHLCV providers returned nothing for {ticker} — falling back to Yahoo Finance (yfinance)")
        return await self._fetch_ohlcv_yfinance(ticker, days)

    @staticmethod
    async def _fetch_ohlcv_yfinance(ticker: str, days: int) -> list[dict[str, Any]]:
        """Fetch OHLCV bars from Yahoo Finance using yfinance (single-ticker download)."""
        try:
            import asyncio as _asyncio
            import yfinance as yf

            yf_symbol = f"{ticker}.SI"

            def _download() -> list[dict[str, Any]]:
                # Use Ticker object for single-stock history — avoids MultiIndex columns
                t = yf.Ticker(yf_symbol)
                df = t.history(period=f"{days + 7}d", interval="1d", auto_adjust=True)
                if df.empty:
                    return []
                bars = []
                for ts, row in df.tail(days).iterrows():
                    trade_date = ts.date() if hasattr(ts, "date") else ts
                    bars.append({
                        "date":   str(trade_date),
                        "open":   float(row.get("Open",   0) or 0),
                        "high":   float(row.get("High",   0) or 0),
                        "low":    float(row.get("Low",    0) or 0),
                        "close":  float(row.get("Close",  0) or 0),
                        "volume": float(row.get("Volume", 0) or 0),
                        "source": "Yahoo Finance",
                    })
                return bars

            loop = _asyncio.get_event_loop()
            bars = await loop.run_in_executor(None, _download)
            if bars:
                logger.info(f"[Registry] OHLCV for {ticker} from Yahoo Finance: {len(bars)} bars")
            else:
                logger.debug(f"[Registry] Yahoo Finance returned no OHLCV bars for {ticker}")
            return bars

        except Exception as exc:
            logger.warning(f"[Registry] Yahoo Finance OHLCV fallback failed for {ticker}: {exc}")
            return []

    # ------------------------------------------------------------------
    # Sentiment fetch (StockGeist, or any enabled sentiment provider)
    # ------------------------------------------------------------------

    async def fetch_sentiment(self, ticker: str) -> dict[str, Any]:
        """
        Fetch the latest sentiment snapshot from the first enabled sentiment provider.

        Returns:
            Sentiment dict or {} if unavailable.
        """
        sentiment_providers = [
            p for p in self._providers.values()
            if p.enabled and p.CATEGORY == "sentiment"
        ]
        if not sentiment_providers:
            return {}

        for provider in sentiment_providers:
            try:
                if hasattr(provider, "fetch_sentiment_now"):
                    snap = await provider.fetch_sentiment_now(ticker)  # type: ignore[attr-defined]
                else:
                    articles = await provider.fetch(ticker)
                    snap = articles[0] if articles else {}
                if snap:
                    logger.info(f"[Registry] Sentiment for {ticker} from {provider.NAME}: score={snap.get('score')}")
                    return snap
            except Exception as exc:
                logger.warning(f"[Registry] Sentiment {provider.NAME} failed for {ticker}: {exc}")

        return {}

    # ------------------------------------------------------------------
    # Real-time quote (Finnhub, or any enabled quote-capable provider)
    # ------------------------------------------------------------------

    async def fetch_quote(self, ticker: str) -> dict[str, Any]:
        """Fetch a real-time quote from any enabled provider that supports it."""
        for name, provider in self._providers.items():
            if provider.enabled and hasattr(provider, "fetch_quote"):
                try:
                    quote = await provider.fetch_quote(ticker)  # type: ignore[attr-defined]
                    if quote:
                        logger.debug(f"[Registry] Quote for {ticker} from {provider.NAME}")
                        return quote
                except Exception as exc:
                    logger.warning(f"[Registry] Quote {provider.NAME} failed for {ticker}: {exc}")
        return {}
