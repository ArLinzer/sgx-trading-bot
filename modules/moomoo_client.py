"""
moomoo_client.py — STUB (Moomoo integration disabled).

Moomoo has been replaced by:
  - Yahoo Finance (yfinance)  — live price refresh, OHLCV
  - Finnhub                   — quotes, market news
  - Marketaux                 — financial news with sentiment
  - EODHD                     — financial news
  - Google News RSS           — free news scraping
  - SGX Announcements API     — official SGX filings

This stub exists solely so any stale import of MoomooClient doesn't
cause an ImportError. All methods return empty results immediately.
"""

from __future__ import annotations

from typing import Any


class MoomooClient:
    """No-op stub — Moomoo integration is disabled."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def check_gateway(self) -> bool:
        return False

    async def get_stock_quote(self, ticker_list: list[str]) -> list[dict[str, Any]]:
        return []

    async def get_news_feed(self, ticker: str, max_items: int = 20) -> list[dict[str, Any]]:
        return []

    async def get_order_book(self, ticker: str, num_levels: int = 10) -> dict[str, Any]:
        return {}

    async def get_historical_kline(
        self, ticker: str, days: int = 5, ktype: str = "K_DAY"
    ) -> list[dict[str, Any]]:
        return []
