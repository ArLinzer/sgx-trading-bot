"""
marketstack.py — Marketstack OHLCV provider.

API docs : https://marketstack.com/documentation
Auth     : ?access_key=<key>
Endpoints used:
  /v1/eod      — end-of-day OHLCV (historical)
  /v2/eod      — v2 endpoint (same, for paid plans)
Params   : symbols, date_from, limit, access_key

SGX stocks use MIC suffix: TICKER.XSES  (e.g. D05.XSES)

NOTE: Marketstack free plan requires HTTPS and may have limited exchange coverage.
If the API key returns 401, the provider logs a clear message and returns [].
OHLCV falls back to yfinance (via SGXScanner.refresh_prices) automatically.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
from loguru import logger

from .base import BaseProvider, ProviderArticle

EOD_URL  = "https://api.marketstack.com/v1/eod"
TIMEOUT  = aiohttp.ClientTimeout(total=15)

# Set once on first 401 to avoid repeating the same error on every ticker
_AUTH_FAILED: bool = False


class MarketstackProvider(BaseProvider):
    NAME     = "Marketstack"
    CATEGORY = "ohlcv"

    async def fetch(self, ticker: str) -> list[ProviderArticle]:
        return await self.fetch_ohlcv(ticker)  # type: ignore[return-value]

    async def fetch_ohlcv(self, ticker: str, days: int = 5) -> list[dict[str, Any]]:
        """
        Fetch recent EOD OHLCV bars for an SGX ticker.

        Returns list of bar dicts:
          { date, open, high, low, close, volume, adj_close, symbol, source }
        sorted oldest-first.
        """
        global _AUTH_FAILED

        if not self._guard():
            return []

        if _AUTH_FAILED:
            logger.debug(f"[{self.NAME}] Skipping {ticker} — API key auth failed previously")
            return []

        # Marketstack uses MIC suffix; SGX = XSES
        symbol    = f"{ticker}.XSES"
        date_from = (
            datetime.now(tz=timezone.utc) - timedelta(days=days + 7)
        ).strftime("%Y-%m-%d")

        params = {
            "symbols":    symbol,
            "date_from":  date_from,
            "limit":      str(days + 7),
            "access_key": self.api_key,
        }

        logger.debug(f"[{self.NAME}] Fetching EOD OHLCV for {ticker} ({symbol})")
        try:
            async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
                async with session.get(EOD_URL, params=params) as resp:
                    if resp.status == 401:
                        _AUTH_FAILED = True
                        logger.warning(
                            f"[{self.NAME}] API key authentication failed (HTTP 401). "
                            "Check your Marketstack access key at https://marketstack.com/dashboard. "
                            "OHLCV data will fall back to Yahoo Finance (yfinance)."
                        )
                        return []
                    if resp.status != 200:
                        logger.warning(f"[{self.NAME}] HTTP {resp.status} for {ticker}")
                        return []
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning(f"[{self.NAME}] Request failed for {ticker}: {exc}")
            return []

        # Check for API-level error in JSON response
        if "error" in data:
            err = data["error"]
            if err.get("code") == "invalid_access_key":
                _AUTH_FAILED = True
                logger.warning(
                    f"[{self.NAME}] Invalid API key: {err.get('message','')}"
                    " — OHLCV will fall back to Yahoo Finance."
                )
            else:
                logger.warning(f"[{self.NAME}] API error for {ticker}: {err}")
            return []

        bars: list[dict[str, Any]] = []
        for item in data.get("data", []):
            bars.append({
                "date":      item.get("date", ""),
                "open":      float(item.get("open")      or 0),
                "high":      float(item.get("high")      or 0),
                "low":       float(item.get("low")       or 0),
                "close":     float(item.get("close")     or 0),
                "volume":    float(item.get("volume")    or 0),
                "adj_close": float(item.get("adj_close") or 0),
                "symbol":    item.get("symbol", ""),
                "source":    self.NAME,
            })

        bars.sort(key=lambda x: x.get("date", ""))
        logger.info(f"[{self.NAME}] {len(bars)} OHLCV bars for {ticker}")
        return bars[-days:]
