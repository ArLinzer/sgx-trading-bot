"""
sgx_scanner.py — Fetch all SGX-listed stocks.

Primary source: SGX official API (api2.sgx.com)
Fallback:       Scrape sgx.com/securities/equities
Cache:          Refreshed once per trading day at 08:30 SGT
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import date, datetime, timezone
from typing import Any, Optional

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

SGX_API_URL = "https://api.sgx.com/securities/v1.1/stocks"
SGX_EQUITIES_URL = "https://www.sgx.com/securities/equities"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
}

TIMEOUT = aiohttp.ClientTimeout(total=30)


class SGXScanner:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f"sgx_stocks_{date.today()}.json")
        os.makedirs(cache_dir, exist_ok=True)

    async def get_stock_list(self) -> list[dict[str, Any]]:
        """Return full list of active SGX-listed stocks, using cache if available."""
        cached = self._load_cache()
        if cached:
            logger.info(f"SGX stock list loaded from cache ({len(cached)} stocks)")
            return cached

        stocks = await self._fetch_from_api()
        if not stocks:
            logger.warning("SGX API returned no data, falling back to scraper")
            stocks = await self._fetch_from_scraper()

        if stocks:
            self._save_cache(stocks)
            logger.info(f"SGX stock list fetched and cached ({len(stocks)} stocks)")
        else:
            logger.error("Failed to fetch SGX stock list from all sources")

        return stocks

    def pre_filter_stocks(
        self,
        stocks: list[dict[str, Any]],
        top_n: int = 100,
        min_turnover_sgd: float = 5_000_000,
    ) -> list[dict[str, Any]]:
        """
        Reduce the full SGX stock list to the most liquid, non-stale counters
        before sending to the LLM.  This keeps the LLM prompt small and focused.

        Filtering steps (in order):
          1. Drop price_stale = True           — no trading in >5 calendar days
          2. Drop turnover_sgd < min_turnover_sgd — SGD value traded below floor
          3. Sort by turnover_sgd descending   — highest activity first
          4. Return top_n

        ``turnover_sgd`` comes directly from the SGX API ``v`` field and
        represents the total SGD value of shares traded in the last session.
        It is the most reliable liquidity measure (combines price × volume).

        Args:
            stocks:          Full SGX stock list from get_stock_list()
            top_n:           Maximum number of stocks to return (default 100)
            min_turnover_sgd: Minimum SGD turnover; stocks below this are
                             excluded (default SGD 5,000,000).  Set to 0 to
                             disable the floor filter.

        Returns:
            Filtered and sorted list of up to top_n stocks.
        """
        # Step 1: Drop stale counters (price >5 calendar days old = illiquid)
        active = [s for s in stocks if not s.get("price_stale")]
        dropped_stale = len(stocks) - len(active)

        # Step 2: Drop below SGD turnover floor
        if min_turnover_sgd > 0:
            before_tov = len(active)
            active = [
                s for s in active
                if (s.get("turnover_sgd") or 0) >= min_turnover_sgd
            ]
            dropped_tov = before_tov - len(active)
        else:
            dropped_tov = 0

        # Step 3: Sort by SGD turnover descending
        active.sort(key=lambda s: s.get("turnover_sgd") or 0, reverse=True)

        # Step 4: Take top N
        result = active[:top_n]

        logger.info(
            f"Pre-filter: {len(stocks)} total → "
            f"{dropped_stale} stale dropped → "
            f"{dropped_tov} below SGD {min_turnover_sgd:,.0f} turnover dropped → "
            f"{len(active)} qualify → top {len(result)} sent to LLM"
        )
        for i, s in enumerate(result[:5], 1):
            logger.info(
                f"  #{i:>3}  {s['ticker']:6s}  {s.get('name',''):30s}  "
                f"turnover=SGD {s.get('turnover_sgd') or 0:>16,.0f}  "
                f"price={s.get('last_price') or 'N/A'}"
            )
        if len(result) > 5:
            logger.info(f"  ... and {len(result) - 5} more")

        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch_from_api(self) -> list[dict[str, Any]]:
        """Fetch stock list from SGX official API."""
        try:
            async with aiohttp.ClientSession(headers=HEADERS, timeout=TIMEOUT) as session:
                async with session.get(SGX_API_URL) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)

            return self._parse_api_response(data)
        except Exception as exc:
            logger.warning(f"SGX API fetch failed: {exc}")
            return []

    def _parse_api_response(self, data: Any) -> list[dict[str, Any]]:
        """Normalise the SGX API JSON into a flat list of stock dicts.

        Current SGX API (api.sgx.com/securities/v1.1/stocks) shape:
          { "meta": {...}, "data": { "prices": [ {...}, ... ] } }

        Key field mappings:
          nc           → ticker
          issuer-name  → name
          lt           → last_price (last traded)
          o            → open
          pv           → prev_close
          h / l        → high / low
          c            → change
          vl           → volume_lots (number of board lots traded)
          v            → turnover_sgd (SGD value of shares traded — native from SGX)
          m            → board ("MAINBOARD" / "CATALIST")
          cur          → currency
          sc           → sector code
        """
        stocks: list[dict[str, Any]] = []

        # Unwrap { data: { prices: [...] } }
        items: list[Any] = []
        if isinstance(data, dict):
            inner = data.get("data", data)
            if isinstance(inner, dict):
                items = inner.get("prices", [])
            elif isinstance(inner, list):
                items = inner
        elif isinstance(data, list):
            items = data

        today = date.today()

        for item in items:
            if not isinstance(item, dict):
                continue

            ticker = (item.get("nc") or item.get("stockCode") or item.get("code") or "").strip()
            if not ticker:
                continue

            # Parse trading_time: "20260410_091600" → date + age
            raw_tt = item.get("trading_time") or item.get("ptd") or ""
            price_date, price_age_days = self._parse_trading_time(raw_tt, today)

            stocks.append(
                {
                    "ticker":       ticker,
                    "name":         item.get("issuer-name") or item.get("companyName") or item.get("n") or "",
                    "board":        item.get("m") or "",              # MAINBOARD / CATALIST
                    "sector":       item.get("sc") or item.get("sector") or "Unknown",
                    "last_price":   self._safe_float(item.get("lt") or item.get("pv") or item.get("lastPrice")),
                    "open":         self._safe_float(item.get("o")),
                    "prev_close":   self._safe_float(item.get("pv")),
                    "change":       self._safe_float(item.get("c")),
                    "change_pct":   self._safe_float(item.get("change_vs_pc_percentage")),
                    "high":         self._safe_float(item.get("h")),
                    "low":          self._safe_float(item.get("l")),
                    # vl = board lots traded (1 lot = 100 shares for stocks ≥ S$0.20)
                    "volume":       self._safe_float(item.get("vl") or item.get("volume")),
                    # v = SGD turnover (native from SGX API — most reliable liquidity measure)
                    "turnover_sgd": self._safe_float(item.get("v") or item.get("turnover")),
                    "currency":     item.get("cur") or item.get("currency") or "SGD",
                    "isin":         item.get("isin") or "",
                    "price_date":       price_date,         # "2026-04-10"
                    "price_age_days":   price_age_days,     # calendar days since last trade
                    "price_stale":      self._is_stale(price_age_days),
                }
            )

        # Log staleness summary
        stale_count = sum(1 for s in stocks if s.get("price_stale"))
        if stale_count:
            logger.warning(
                f"SGX scan: {len(stocks)} stocks | "
                f"{stale_count} with stale prices (>5 trading days old — likely illiquid)"
            )

        return stocks

    @staticmethod
    def _parse_trading_time(raw: str, today: date) -> tuple:
        """
        Parse SGX trading_time string into (date_str, age_days).
        Format: "20260410_091600" or plain "20260410"
        """
        if not raw:
            return ("", None)
        try:
            date_part = raw[:8]
            trade_date = datetime.strptime(date_part, "%Y%m%d").date()
            age = (today - trade_date).days
            return (trade_date.isoformat(), age)
        except (ValueError, TypeError):
            return (str(raw), None)

    @staticmethod
    def _is_stale(age_days: Any) -> bool:
        """
        True if the price is more than 5 calendar days old.
        5 days covers a long weekend; anything beyond that is genuinely illiquid.
        """
        if age_days is None:
            return False
        try:
            return int(age_days) > 5
        except (TypeError, ValueError):
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch_from_scraper(self) -> list[dict[str, Any]]:
        """Scrape SGX equities page as fallback."""
        try:
            async with aiohttp.ClientSession(headers=HEADERS, timeout=TIMEOUT) as session:
                async with session.get(SGX_EQUITIES_URL) as resp:
                    resp.raise_for_status()
                    html = await resp.text()

            return self._parse_equities_html(html)
        except Exception as exc:
            logger.error(f"SGX scraper fallback failed: {exc}")
            return []

    def _parse_equities_html(self, html: str) -> list[dict[str, Any]]:
        """Extract stock rows from SGX equities HTML table."""
        soup = BeautifulSoup(html, "html.parser")
        stocks: list[dict[str, Any]] = []

        # SGX renders equities via a JavaScript data store; look for JSON blobs
        for script in soup.find_all("script"):
            text = script.string or ""
            if "stockCode" in text or '"code"' in text:
                try:
                    start = text.find("[")
                    end = text.rfind("]") + 1
                    if start != -1 and end > start:
                        items = json.loads(text[start:end])
                        parsed = self._parse_api_response(items)
                        if parsed:
                            return parsed
                except (json.JSONDecodeError, ValueError):
                    pass

        # Table row fallback
        for row in soup.select("table tr"):
            cells = row.find_all("td")
            if len(cells) >= 4:
                ticker = cells[0].get_text(strip=True)
                name = cells[1].get_text(strip=True)
                if ticker and name:
                    stocks.append(
                        {
                            "ticker": ticker,
                            "name": name,
                            "market_cap": None,
                            "sector": "Unknown",
                            "last_price": self._safe_float(cells[2].get_text(strip=True)),
                            "volume": self._safe_float(cells[3].get_text(strip=True)),
                            "currency": "SGD",
                            "isin": "",
                        }
                    )

        return stocks

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> Optional[list]:
        """Return today's cached stock list, or None if stale / missing."""
        if not os.path.exists(self.cache_file):
            return None
        try:
            with open(self.cache_file) as f:
                data = json.load(f)
            # Invalidate if the cache is older than 23 hours
            if time.time() - data.get("_ts", 0) > 82800:
                return None
            return data.get("stocks", [])
        except Exception:
            return None

    def _save_cache(self, stocks: list[dict[str, Any]]) -> None:
        try:
            with open(self.cache_file, "w") as f:
                json.dump({"_ts": time.time(), "stocks": stocks}, f)
        except Exception as exc:
            logger.warning(f"Failed to write SGX cache: {exc}")

    # ------------------------------------------------------------------
    # Yahoo Finance live price refresh
    # ------------------------------------------------------------------

    async def refresh_prices(
        self, stocks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Fetch live prices via yfinance (handles Yahoo Finance auth/cookies/rate-limits).

        Designed for the 10-20 LLM-selected tickers — one bulk yfinance download call.
        Updates last_price, volume, change_pct, high, low, price_date in-place.
        Returns the updated list.
        """
        if not stocks:
            return stocks

        try:
            import yfinance as yf
        except ImportError:
            logger.warning("[PriceRefresh] yfinance not installed — run: pip3 install yfinance")
            return stocks

        tickers = [s["ticker"] for s in stocks]
        yf_symbols = [f"{t}.SI" for t in tickers]

        logger.info(f"[PriceRefresh] Downloading live prices via yfinance for: {', '.join(tickers)}")

        try:
            price_map = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._yfinance_bulk(yf_symbols)
            )
        except Exception as exc:
            logger.warning(f"[PriceRefresh] yfinance bulk download failed: {exc}")
            return stocks

        refreshed = 0
        for stock in stocks:
            ticker = stock["ticker"]
            fresh = price_map.get(ticker)
            if fresh and fresh.get("last_price"):
                old = stock.get("last_price")
                stock.update(fresh)
                logger.info(
                    f"[PriceRefresh] {ticker}: SGX={old} → Yahoo={fresh['last_price']} "
                    f"| chg={fresh.get('change_pct', 'N/A')}% "
                    f"| vol={fresh.get('volume')} "
                    f"| date={fresh.get('price_date')} (age={fresh.get('price_age_days')}d)"
                )
                refreshed += 1
            else:
                logger.warning(
                    f"[PriceRefresh] {ticker}: no live price from Yahoo — "
                    f"keeping SGX API price ({stock.get('last_price')} from {stock.get('price_date')})"
                )

        logger.info(f"[PriceRefresh] Done — {refreshed}/{len(stocks)} prices refreshed from Yahoo Finance")
        return stocks

    @staticmethod
    def _yfinance_bulk(yf_symbols: list[str]) -> dict[str, dict[str, Any]]:
        """
        Synchronous yfinance bulk download — run in executor to avoid blocking.
        Returns dict keyed by bare SGX ticker (e.g. 'D05').
        """
        import yfinance as yf

        # download() returns a DataFrame with MultiIndex columns
        df = yf.download(
            tickers=" ".join(yf_symbols),
            period="2d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        result: dict[str, dict[str, Any]] = {}
        today = date.today()

        for sym in yf_symbols:
            ticker = sym.replace(".SI", "")
            try:
                # Multi-ticker download nests columns under (ticker, field)
                if hasattr(df.columns, "levels") and sym in df.columns.get_level_values(0):
                    sub = df[sym].dropna(how="all")
                else:
                    # Single-ticker download — flat columns
                    sub = df.dropna(how="all")

                if sub.empty:
                    continue

                last = sub.iloc[-1]
                prev = sub.iloc[-2] if len(sub) >= 2 else None

                trade_date = sub.index[-1].date() if hasattr(sub.index[-1], "date") else None
                price_date = trade_date.isoformat() if trade_date else ""
                age = (today - trade_date).days if trade_date else None

                close = float(last.get("Close") or last.get("close") or 0)
                prev_close = float(prev.get("Close") or prev.get("close") or 0) if prev is not None else None
                change_pct = round((close - prev_close) / prev_close * 100, 2) if prev_close else None

                result[ticker] = {
                    "last_price":     close or None,
                    "prev_close":     prev_close,
                    "high":           float(last.get("High") or last.get("high") or 0) or None,
                    "low":            float(last.get("Low")  or last.get("low")  or 0) or None,
                    "volume":         float(last.get("Volume") or last.get("volume") or 0) or None,
                    "change_pct":     change_pct,
                    "price_date":     price_date,
                    "price_age_days": age,
                    "price_stale":    SGXScanner._is_stale(age),
                    "price_source":   "Yahoo Finance",
                }
            except Exception as exc:
                logger.debug(f"[PriceRefresh] {ticker}: parse error — {exc}")

        return result

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _main() -> None:
        scanner = SGXScanner()
        stocks = await scanner.get_stock_list()
        print(f"Fetched {len(stocks)} stocks")
        for s in stocks[:5]:
            print(s)

    asyncio.run(_main())
