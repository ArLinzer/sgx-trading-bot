"""
fundamental_analyzer.py — Fetch fundamental data for SGX stocks via yfinance.

Metrics returned (all sourced from yfinance .info + .financials):
  Valuation  : market cap, P/E (trailing + forward), P/B, EV/EBITDA
  Income     : revenue TTM, net income TTM, EPS, net margin, revenue/earnings growth
  Balance    : debt/equity, current ratio, book value/share, ROE, ROA
  Dividends  : yield, DPS, payout ratio, ex-dividend date
  Technicals : 52-week high/low, beta, avg volume (10-day)
  Score      : simple composite quality score (0–100)

All network calls run in an executor to avoid blocking the asyncio event loop.
Results are disk-cached for 24 hours so repeated lookups are instant.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, date, timezone
from typing import Any, Optional

from loguru import logger

# Cache TTL: 24 hours (fundamentals don't change intraday)
CACHE_TTL_SECONDS = 86_400


def _safe(val: Any, multiplier: float = 1.0, decimals: int = 2) -> Optional[float]:
    """Return rounded float or None for missing/NaN/Inf values."""
    try:
        v = float(val) * multiplier
        if v != v or abs(v) == float("inf"):  # NaN or Inf check
            return None
        return round(v, decimals)
    except (TypeError, ValueError):
        return None


def _fmt_large(val: Any) -> Optional[str]:
    """Format large numbers (market cap, revenue) as human-readable string."""
    v = _safe(val)
    if v is None:
        return None
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"S${v / 1e12:.2f}T"
    if abs_v >= 1e9:
        return f"S${v / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"S${v / 1e6:.2f}M"
    return f"S${v:,.0f}"


def _pct(val: Any) -> Optional[str]:
    """Format decimal ratio as percentage string."""
    v = _safe(val)
    if v is None:
        return None
    return f"{v * 100:.1f}%"


def _fetch_yf_info(yf_symbol: str) -> dict[str, Any]:
    """
    Synchronous yfinance fetch — runs in executor.
    Returns a normalised fundamentals dict.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance not installed — run: pip install yfinance"}

    try:
        tkr  = yf.Ticker(yf_symbol)
        info = tkr.info or {}
    except Exception as exc:
        return {"error": f"yfinance fetch failed: {exc}"}

    if not info or info.get("quoteType") is None:
        return {"error": f"No data available for {yf_symbol} (not listed or delisted)"}

    # ── Valuation ──────────────────────────────────────────────────────
    market_cap       = _safe(info.get("marketCap"))
    trailing_pe      = _safe(info.get("trailingPE"))
    forward_pe       = _safe(info.get("forwardPE"))
    price_to_book    = _safe(info.get("priceToBook"))
    ev_to_ebitda     = _safe(info.get("enterpriseToEbitda"))
    enterprise_value = _safe(info.get("enterpriseValue"))

    # ── Income ─────────────────────────────────────────────────────────
    revenue          = _safe(info.get("totalRevenue"))
    net_income       = _safe(info.get("netIncomeToCommon"))
    eps_trailing     = _safe(info.get("trailingEps"))
    eps_forward      = _safe(info.get("forwardEps"))
    net_margin       = _safe(info.get("profitMargins"))
    revenue_growth   = _safe(info.get("revenueGrowth"))
    earnings_growth  = _safe(info.get("earningsGrowth"))
    gross_margin     = _safe(info.get("grossMargins"))
    operating_margin = _safe(info.get("operatingMargins"))

    # ── Balance sheet ──────────────────────────────────────────────────
    debt_to_equity   = _safe(info.get("debtToEquity"))
    current_ratio    = _safe(info.get("currentRatio"))
    book_value       = _safe(info.get("bookValue"))
    roe              = _safe(info.get("returnOnEquity"))
    roa              = _safe(info.get("returnOnAssets"))
    free_cashflow    = _safe(info.get("freeCashflow"))
    op_cashflow      = _safe(info.get("operatingCashflow"))

    # ── Dividends ──────────────────────────────────────────────────────
    # yfinance sometimes returns dividendYield as 5.34 (percent) instead
    # of 0.0534 (decimal) for SGX stocks — normalise to decimal if > 1.
    raw_yield = info.get("dividendYield")
    if raw_yield is not None:
        try:
            raw_yield = float(raw_yield)
            if raw_yield > 1:
                raw_yield = raw_yield / 100
        except (TypeError, ValueError):
            raw_yield = None
    div_yield        = _safe(raw_yield)
    div_rate         = _safe(info.get("dividendRate"))           # annual DPS
    payout_ratio     = _safe(info.get("payoutRatio"))
    ex_div_date_raw  = info.get("exDividendDate")
    try:
        ex_div_date = (
            datetime.fromtimestamp(ex_div_date_raw).strftime("%d %b %Y")
            if ex_div_date_raw else None
        )
    except Exception:
        ex_div_date = str(ex_div_date_raw) if ex_div_date_raw else None

    # ── Technicals ─────────────────────────────────────────────────────
    wk52_high        = _safe(info.get("fiftyTwoWeekHigh"))
    wk52_low         = _safe(info.get("fiftyTwoWeekLow"))
    beta             = _safe(info.get("beta"))
    avg_vol_10d      = _safe(info.get("averageVolume10days"), decimals=0)
    current_price    = _safe(info.get("currentPrice") or info.get("regularMarketPrice"))

    # ── 52-week position (0% = at 52W low, 100% = at 52W high) ────────
    wk52_position: Optional[float] = None
    if wk52_high and wk52_low and current_price and (wk52_high - wk52_low) > 0:
        wk52_position = round((current_price - wk52_low) / (wk52_high - wk52_low) * 100, 1)

    # ── Composite quality score (0–100) ────────────────────────────────
    # Simple heuristic: rewards low P/E, high dividend yield, low debt, positive ROE
    score_parts: list[float] = []

    if trailing_pe is not None and trailing_pe > 0:
        # Lower P/E = better; 10→100pts, 30→0pts, cap at 100
        score_parts.append(max(0.0, min(100.0, (30 - trailing_pe) * 5)))
    if div_yield is not None:
        score_parts.append(min(100.0, div_yield * 1000))        # 5% yield → 50 pts
    if roe is not None:
        score_parts.append(min(100.0, max(0.0, roe * 200)))     # 15% ROE → 30 pts
    if debt_to_equity is not None:
        score_parts.append(max(0.0, min(100.0, 100 - debt_to_equity * 0.5)))
    if net_margin is not None:
        score_parts.append(min(100.0, max(0.0, net_margin * 300)))  # 10% margin → 30 pts

    quality_score = round(sum(score_parts) / max(len(score_parts), 1), 1) if score_parts else None

    # ── Build result ───────────────────────────────────────────────────
    ticker_clean = yf_symbol.replace(".SI", "")
    return {
        "ticker":          ticker_clean,
        "name":            info.get("longName") or info.get("shortName") or ticker_clean,
        "sector":          info.get("sector") or info.get("industry") or "—",
        "exchange":        info.get("exchange") or "SGX",
        "currency":        info.get("currency") or "SGD",
        "fetched_at":      datetime.now(tz=timezone.utc).isoformat(),

        "valuation": {
            "market_cap":        market_cap,
            "market_cap_fmt":    _fmt_large(market_cap),
            "enterprise_value":  enterprise_value,
            "ev_fmt":            _fmt_large(enterprise_value),
            "trailing_pe":       trailing_pe,
            "forward_pe":        forward_pe,
            "price_to_book":     price_to_book,
            "ev_to_ebitda":      ev_to_ebitda,
        },

        "income": {
            "revenue_ttm":       revenue,
            "revenue_fmt":       _fmt_large(revenue),
            "net_income_ttm":    net_income,
            "net_income_fmt":    _fmt_large(net_income),
            "eps_trailing":      eps_trailing,
            "eps_forward":       eps_forward,
            "net_margin":        net_margin,
            "net_margin_pct":    _pct(net_margin),
            "gross_margin_pct":  _pct(gross_margin),
            "operating_margin_pct": _pct(operating_margin),
            "revenue_growth":    revenue_growth,
            "revenue_growth_pct": _pct(revenue_growth),
            "earnings_growth":   earnings_growth,
            "earnings_growth_pct": _pct(earnings_growth),
        },

        "balance": {
            "debt_to_equity":    debt_to_equity,
            "current_ratio":     current_ratio,
            "book_value_per_share": book_value,
            "roe":               roe,
            "roe_pct":           _pct(roe),
            "roa":               roa,
            "roa_pct":           _pct(roa),
            "free_cashflow":     free_cashflow,
            "free_cashflow_fmt": _fmt_large(free_cashflow),
            "operating_cashflow": op_cashflow,
        },

        "dividends": {
            "yield":             div_yield,
            "yield_pct":         _pct(div_yield),
            "annual_dps":        div_rate,
            "payout_ratio":      payout_ratio,
            "payout_ratio_pct":  _pct(payout_ratio),
            "ex_dividend_date":  ex_div_date,
        },

        "technicals": {
            "current_price":     current_price,
            "wk52_high":         wk52_high,
            "wk52_low":          wk52_low,
            "wk52_position_pct": wk52_position,
            "beta":              beta,
            "avg_volume_10d":    avg_vol_10d,
        },

        "quality_score":     quality_score,
    }


class FundamentalAnalyzer:
    """
    Async wrapper around the synchronous yfinance fundamentals fetcher.
    Caches results to disk for 24 hours.
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, ticker: str) -> str:
        return os.path.join(self.cache_dir, f"fundamentals_{ticker}.json")

    def _load_cache(self, ticker: str) -> Optional[dict]:
        path = self._cache_path(ticker)
        try:
            with open(path) as f:
                data = json.load(f)
            fetched_at = data.get("fetched_at", "")
            if fetched_at:
                cached_dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
                age = (datetime.now(tz=cached_dt.tzinfo) - cached_dt).total_seconds()
                if age < CACHE_TTL_SECONDS:
                    logger.debug(f"[Fundamentals] {ticker} — cache hit ({age/3600:.1f}h old)")
                    return data
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass
        return None

    def _save_cache(self, ticker: str, data: dict) -> None:
        try:
            with open(self._cache_path(ticker), "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as exc:
            logger.debug(f"[Fundamentals] Cache write failed for {ticker}: {exc}")

    async def fetch(self, ticker: str) -> dict[str, Any]:
        """
        Return fundamental data for *ticker* (bare SGX code, e.g. 'D05').
        Reads from cache if fresh; otherwise fetches via yfinance in an executor.
        """
        ticker = ticker.upper()

        # Check cache first
        cached = self._load_cache(ticker)
        if cached:
            return cached

        yf_symbol = f"{ticker}.SI"
        logger.info(f"[Fundamentals] Fetching {yf_symbol} via yfinance…")
        try:
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, _fetch_yf_info, yf_symbol)
        except Exception as exc:
            logger.warning(f"[Fundamentals] Executor error for {ticker}: {exc}")
            data = {"error": str(exc), "ticker": ticker}

        if "error" not in data:
            self._save_cache(ticker, data)
            logger.info(f"[Fundamentals] {ticker} — fetched OK (score={data.get('quality_score')})")
        else:
            logger.warning(f"[Fundamentals] {ticker} — {data['error']}")

        return data

    async def fetch_batch(self, tickers: list[str]) -> dict[str, dict]:
        """Fetch fundamentals for multiple tickers concurrently (respects cache)."""
        results = await asyncio.gather(
            *[self.fetch(t) for t in tickers], return_exceptions=False
        )
        return {t.upper(): r for t, r in zip(tickers, results)}
