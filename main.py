"""
main.py — SGX Day Trading Signal Bot orchestrator.

Data sources (in priority order):
  1. SGX official API      — full stock list + prices
  2. Yahoo Finance (yfinance) — live price refresh for selected tickers
  3. Finnhub               — real-time quotes, news, fundamentals (enabled in config)
  4. EODHD                 — news with sentiment (enabled in config)
  5. Marketstack           — OHLCV bars (enabled in config)
  6. DuckDuckGo            — web/news search via Ollama tool-calling (Call B only)
  7. Free RSS scrapers      — SGX announcements, Business Times, etc.

Schedule (Asia/Singapore timezone):
  08:30  Fetch SGX stock list + LLM stock selection (Call A)
  09:00  Fetch news + LLM strategy (Call B) → send morning signals
  12:00  Re-run news fetch + LLM refresh → send midday updates
  14:00  Afternoon session signals
  16:30  EOD sell reminder
  17:15  Daily summary

CLI flags:
  --dry-run   Print signals to stdout instead of sending Telegram messages
  --once      Run a single full cycle immediately (for testing)
  --force     Bypass the trading day check (useful for weekend testing)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any, Optional, Tuple

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from zoneinfo import ZoneInfo

from modules.fundamental_analyzer import FundamentalAnalyzer
from modules.llm_analyst import LLMAnalyst
from modules.news_fetcher import NewsFetcher
from modules.providers.registry import ProviderRegistry
from modules.sgx_scanner import SGXScanner
from modules.signal_engine import SESSION_AFTERNOON, SESSION_MORNING, SignalEngine
from modules.telegram_bot import TelegramNotifier
from modules.watchlist import SignalStore, WatchlistManager, WSManager


# ---------------------------------------------------------------------------
# API request models
# ---------------------------------------------------------------------------

class WatchlistAddRequest(BaseModel):
    ticker: str
    name: str

class SettingsUpdateRequest(BaseModel):
    telegram_enabled: Optional[bool] = None
    watchlist_interval_hours: Optional[int] = None

SGT = ZoneInfo("Asia/Singapore")

# ---------------------------------------------------------------------------
# SGX market holidays (update annually)
# Source: https://www.sgx.com/securities/market-calendar
# ---------------------------------------------------------------------------
SGX_HOLIDAYS_2025 = {
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 29),  # Chinese New Year
    date(2025, 1, 30),  # Chinese New Year
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 1),   # Labour Day
    date(2025, 5, 12),  # Vesak Day
    date(2025, 6, 7),   # Hari Raya Haji
    date(2025, 8, 9),   # National Day
    date(2025, 10, 20), # Deepavali
    date(2025, 12, 25), # Christmas Day
}

SGX_HOLIDAYS_2026 = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 2, 17),  # Chinese New Year
    date(2026, 2, 18),  # Chinese New Year
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 1),   # Labour Day
    date(2026, 6, 1),   # Vesak Day (approximate)
    date(2026, 8, 9),   # National Day
    date(2026, 12, 25), # Christmas Day
}

SGX_HOLIDAYS = SGX_HOLIDAYS_2025 | SGX_HOLIDAYS_2026


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Trading day guard
# ---------------------------------------------------------------------------

_force_trading_day: bool = False


def is_trading_day() -> bool:
    if _force_trading_day:
        return True
    today = date.today()
    if today.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return today not in SGX_HOLIDAYS


# ---------------------------------------------------------------------------
# Bot state (shared across scheduled tasks)
# ---------------------------------------------------------------------------

class BotState:
    def __init__(self) -> None:
        self.selected_stocks: list[dict[str, Any]] = []
        self.signals_today: list[dict[str, Any]] = []
        self.start_time = datetime.now(tz=SGT)
        self.last_run: str = "never"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class TradingBot:
    def __init__(self, cfg: dict[str, Any], dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.state = BotState()

        cache_dir = cfg.get("cache", {}).get("dir", "data/cache")
        log_dir   = cfg.get("logging", {}).get("log_dir", "data/logs")
        data_dir  = os.path.dirname(cache_dir)   # "data"
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(log_dir,   exist_ok=True)
        os.makedirs(data_dir,  exist_ok=True)

        self._setup_logging(log_dir, cfg.get("logging", {}).get("level", "INFO"))

        llm_cfg = cfg.get("llm", {})
        flt_cfg = cfg.get("filters", {})
        tg_cfg  = cfg.get("telegram", {})

        self.scanner          = SGXScanner(cache_dir=cache_dir)
        self.fundamentals     = FundamentalAnalyzer(cache_dir=cache_dir)
        self.provider_registry = ProviderRegistry(cfg.get("providers", {}))
        self.news_fetcher     = NewsFetcher(provider_registry=self.provider_registry)
        self.analyst          = LLMAnalyst(
            model=llm_cfg.get("model", "qwen3.5:9b"),
            fallback_models=llm_cfg.get("fallback_models", ["mistral", "phi3"]),
            ollama_host=llm_cfg.get("ollama_host", "http://localhost:11434"),
            request_timeout=llm_cfg.get("request_timeout", 120),
        )
        self.signal_engine    = SignalEngine(
            min_confidence=flt_cfg.get("min_confidence", 0.65),
            min_volume_ratio=flt_cfg.get("min_volume_ratio", 1.2),
            max_signals_per_session=flt_cfg.get("max_signals_per_session", 5),
            cache_dir=cache_dir,
        )
        self.telegram = TelegramNotifier(
            bot_token=tg_cfg.get("bot_token", ""),
            chat_id=tg_cfg.get("chat_id", ""),
            dry_run=dry_run,
        )

        # Web UI state
        self.watchlist_mgr = WatchlistManager(data_dir=data_dir)
        self.signal_store  = SignalStore()
        self.ws_manager    = WSManager()

        # Stop/resume flag — set True to gracefully halt running analyses
        self._stopped: bool = False
        self._current_task: Any = None   # asyncio.Task of any running analysis

        # Path to config.yaml — set by run() after construction so we can save edits
        self.config_path: str = ""

        # Log active providers
        provider_status = self.provider_registry.status()
        active = [n for n, enabled in provider_status.items() if enabled]
        logger.info(f"Active data providers: {', '.join(active) if active else 'none'}")
        logger.info(f"Web UI: http://0.0.0.0:{cfg.get('health_check', {}).get('port', 8080)}")

    # ------------------------------------------------------------------
    # Scheduled task: 08:30 — Stock selection
    # ------------------------------------------------------------------

    async def task_stock_selection(self) -> None:
        if not is_trading_day():
            logger.info("Not a trading day — skipping stock selection")
            return

        logger.info("=== TASK: Stock Selection (08:30) ===")
        self.state.last_run = "08:30 stock_selection"

        # ── 1. Full SGX stock list ────────────────────────────────────────
        stock_list = await self.scanner.get_stock_list()
        if not stock_list:
            logger.error("Failed to fetch SGX stock list — aborting stock selection")
            return

        stale = [s for s in stock_list if s.get("price_stale")]
        logger.info(
            f"SGX API: {len(stock_list)} stocks total | "
            f"{len(stale)} stale (>5 days old, illiquid) | "
            f"{len(stock_list) - len(stale)} active"
        )

        # ── 2. Pre-filter to top N by volume ─────────────────────────────
        flt_cfg        = self.cfg.get("filters", {})
        pre_filter_n   = int(flt_cfg.get("pre_filter_top_n", 100))
        max_llm_select = int(flt_cfg.get("max_llm_select", 20))
        min_turnover   = float(flt_cfg.get("min_turnover_sgd", 5_000_000))
        stock_list     = self.scanner.pre_filter_stocks(
            stock_list, top_n=pre_filter_n, min_turnover_sgd=min_turnover
        )

        # ── 3. Market context ─────────────────────────────────────────────
        market_context = await self._get_market_context()

        # ── 4. LLM Call A — pick best tickers ────────────────────────────
        selected = await self.analyst.select_stocks(
            stock_list, market_context, max_select=max_llm_select
        )

        if selected:
            # ── 5. Refresh live prices via Yahoo Finance ──────────────────
            selected_raw = [
                next((s for s in stock_list if s["ticker"] == sel.ticker), {"ticker": sel.ticker})
                for sel in selected
            ]
            logger.info(f"Refreshing live prices for {len(selected_raw)} selected tickers via Yahoo Finance...")
            refreshed = await self.scanner.refresh_prices(selected_raw)

            # Merge refreshed prices with LLM selection metadata
            price_map = {s["ticker"]: s for s in refreshed}
            self.state.selected_stocks = [
                {**price_map.get(sel.ticker, {"ticker": sel.ticker}), **sel.model_dump()}
                for sel in selected
            ]
        else:
            self.state.selected_stocks = []

        self._log_to_file("stock_selection", self.state.selected_stocks)
        logger.info(f"Stock selection complete: {len(selected)} stocks chosen")

    # ------------------------------------------------------------------
    # Scheduled task: 09:00 / 14:00 — Signal generation
    # ------------------------------------------------------------------

    async def task_generate_signals(self, session: str, skip_day_check: bool = False) -> None:
        if not skip_day_check and not is_trading_day():
            logger.info(f"Not a trading day — skipping {session} signals")
            await self.ws_manager.broadcast({
                "type": "scheduled_status",
                "data": {"message": "Not a trading day — skipping scheduled run", "in_progress": False},
            })
            return

        logger.info(f"=== TASK: Signal Generation ({session}) ===")
        self.state.last_run = f"{session} signal_generation"

        await self.ws_manager.broadcast({
            "type": "scheduled_status",
            "data": {"message": f"Running {session} signal scan…", "in_progress": True},
        })

        if not self.state.selected_stocks:
            logger.warning("No selected stocks — running stock selection first")
            await self.ws_manager.broadcast({
                "type": "scheduled_status",
                "data": {"message": "Fetching SGX stock list…", "in_progress": True},
            })
            await self.task_stock_selection()

        if not self.state.selected_stocks:
            logger.error("Still no selected stocks — cannot generate signals")
            await self.ws_manager.broadcast({
                "type": "scheduled_status",
                "data": {"message": "No stocks selected — cannot generate signals", "in_progress": False},
            })
            return

        raw_signals: list = []
        volume_data: dict[str, Any] = {}
        stocks = self.state.selected_stocks
        total  = len(stocks)

        logger.info(f"Analysing {total} stock(s) — batches of 5 (sequential to avoid Ollama overload)")
        await self.ws_manager.broadcast({
            "type": "scheduled_status",
            "data": {"message": f"Analysing {total} stock(s)…", "in_progress": True},
        })

        # Track this task for cancellation
        self._current_task = asyncio.current_task()

        batch_size = 5
        for batch_start in range(0, total, batch_size):
            # Check stop flag before each batch
            if self._stopped:
                logger.info(f"[Scheduled] Stop requested — halting after batch {batch_start // batch_size}/{(total + batch_size - 1) // batch_size}")
                await self.ws_manager.broadcast({
                    "type": "scheduled_status",
                    "data": {"message": f"Stopped — {len(raw_signals)} signal(s) before halt", "in_progress": False},
                })
                self._current_task = None
                return

            batch     = stocks[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            batch_tot = (total + batch_size - 1) // batch_size
            logger.info(f"--- Batch {batch_num}/{batch_tot}: {[s['ticker'] for s in batch]}")

            tasks   = [self._analyse_single_stock(s) for s in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for stock, result in zip(batch, results):
                ticker = stock["ticker"]
                if isinstance(result, Exception):
                    logger.warning(f"[{ticker}] Analysis failed: {result}")
                elif result is not None:
                    signal, vol = result
                    if signal:
                        raw_signals.append(signal)
                        logger.info(
                            f"[{ticker}] Signal → {signal.action} | "
                            f"conf={signal.confidence:.0%} | "
                            f"entry={signal.entry_price} | target={signal.target_price} | stop={signal.stop_loss}"
                        )
                    if vol:
                        volume_data[ticker] = vol

        logger.info(f"All batches complete — {len(raw_signals)} raw signals before filtering")

        # ── Filter and rank ───────────────────────────────────────────────
        top_signals = self.signal_engine.process_signals(raw_signals, volume_data, session)

        logger.info(f"--- Final signals for {session} session ({len(top_signals)}) ---")
        for i, sig in enumerate(top_signals, 1):
            logger.info(
                f"  #{i} {sig.ticker} | {sig.action} | conf={sig.confidence:.0%} "
                f"| entry={sig.entry_price} | target={sig.target_price} | stop={sig.stop_loss}"
            )

        for sig in top_signals:
            await self.telegram.send_signal(sig, catalyst="")
            stored = self.signal_store.add(sig.model_dump(), source="scheduled")
            await self.ws_manager.broadcast({"type": "signal", "data": stored})

        self._log_to_file(f"signals_{session}", [s.model_dump() for s in top_signals])
        logger.info(f"{session.title()} session complete — {len(top_signals)} signals dispatched")

        self._current_task = None
        finished_at = datetime.now(tz=SGT).strftime("%d %b %Y, %H:%M SGT")
        await self.ws_manager.broadcast({
            "type": "scheduled_status",
            "data": {
                "message": f"Last run: {finished_at} — {len(top_signals)} signal(s) from {total} stock(s)",
                "in_progress": False,
                "finished_at": finished_at,
                "signal_count": len(top_signals),
                "stock_count": total,
            },
        })

    # ------------------------------------------------------------------
    # Single-stock analysis (Call B data fetch + LLM)
    # ------------------------------------------------------------------

    async def _analyse_single_stock(
        self, stock: dict[str, Any]
    ) -> Optional[Tuple[Any, dict]]:
        ticker = stock["ticker"]
        name   = stock.get("name", ticker)

        logger.info(f"[{ticker}] ── Analysing {name} ──")

        # ── 1. Real-time quote (Finnhub → Yahoo/SGX cache fallback) ──────
        logger.info(f"[{ticker}] Fetching real-time quote from providers...")
        quote_data: dict[str, Any] = {}
        try:
            quote_data = await self.provider_registry.fetch_quote(ticker)
            if quote_data:
                logger.info(
                    f"[{ticker}] Quote ({quote_data.get('source','?')}) → "
                    f"last={quote_data.get('last_price')} | "
                    f"open={quote_data.get('open')} | "
                    f"high={quote_data.get('high')} | "
                    f"low={quote_data.get('low')} | "
                    f"chg={quote_data.get('change_pct')}% | "
                    f"prev={quote_data.get('prev_close')}"
                )
        except Exception as exc:
            logger.warning(f"[{ticker}] Provider quote failed: {exc}")

        if not quote_data:
            # Use the price data already refreshed by Yahoo Finance in task_stock_selection
            quote_data = {
                "ticker":     ticker,
                "last_price": stock.get("last_price"),
                "prev_close": stock.get("prev_close"),
                "high":       stock.get("high"),
                "low":        stock.get("low"),
                "volume":     stock.get("volume"),
                "change_pct": stock.get("change_pct"),
                "price_date": stock.get("price_date"),
                "source":     stock.get("price_source", "Yahoo Finance / SGX cache"),
            }
            logger.info(
                f"[{ticker}] Using cached price: "
                f"last={quote_data.get('last_price')} "
                f"({quote_data.get('price_date')}) "
                f"source={quote_data.get('source')}"
            )

        # ── 2. OHLCV bars (Marketstack → empty fallback) ──────────────────
        logger.info(f"[{ticker}] Fetching OHLCV from providers (Marketstack)...")
        kline: list[dict[str, Any]] = []
        try:
            kline = await self.provider_registry.fetch_ohlcv(ticker, days=5)
            if kline:
                logger.info(f"[{ticker}] OHLCV → {len(kline)} bars from provider")
            else:
                logger.info(f"[{ticker}] No OHLCV bars from providers — LLM will use quote data only")
        except Exception as exc:
            logger.warning(f"[{ticker}] OHLCV fetch failed: {exc}")

        # ── 3. News (Finnhub + EODHD + RSS scrapers) ─────────────────────
        logger.info(f"[{ticker}] Fetching news (all sources) — search term: '{name or ticker}'")
        news: list[dict[str, Any]] = []
        try:
            news = await self.news_fetcher.fetch_all(ticker, name=name)
            logger.info(f"[{ticker}] News → {len(news)} articles total")
            for article in news[:5]:
                logger.info(f"  [{article.get('source','?')}] {article.get('headline','')[:80]}")
            if len(news) > 5:
                logger.info(f"  ... and {len(news) - 5} more")
        except Exception as exc:
            logger.warning(f"[{ticker}] News fetch failed: {exc}")

        # ── 4. Sentiment (StockGeist if enabled) ─────────────────────────
        try:
            sentiment = await self.provider_registry.fetch_sentiment(ticker)
            if sentiment:
                logger.info(
                    f"[{ticker}] Sentiment → score={sentiment.get('score')} "
                    f"source={sentiment.get('source')}"
                )
                # Inject sentiment as a synthetic news item so the LLM sees it
                news.insert(0, {
                    "source": sentiment.get("source", "Sentiment"),
                    "headline": sentiment.get("headline", ""),
                    "summary": sentiment.get("summary", ""),
                    "published_at": sentiment.get("published_at", ""),
                })
        except Exception:
            pass

        # ── 5. Fundamental analysis (yfinance, 24-hour cache) ────────────
        logger.info(f"[{ticker}] Fetching fundamental data (yfinance)…")
        fundamentals: dict[str, Any] = {}
        try:
            fundamentals = await self.fundamentals.fetch(ticker)
            if "error" not in fundamentals:
                qs = fundamentals.get("quality_score")
                val = fundamentals.get("valuation", {})
                div = fundamentals.get("dividends", {})
                logger.info(
                    f"[{ticker}] Fundamentals → "
                    f"quality_score={qs} | "
                    f"pe={val.get('trailing_pe')} | "
                    f"pb={val.get('price_to_book')} | "
                    f"div_yield={div.get('yield_pct')} | "
                    f"market_cap={val.get('market_cap_fmt')}"
                )
            else:
                logger.warning(f"[{ticker}] Fundamentals not available: {fundamentals.get('error')}")
                fundamentals = {}
        except Exception as exc:
            logger.warning(f"[{ticker}] Fundamentals fetch failed: {exc}")
            fundamentals = {}

        # ── 6. Data summary before LLM call ──────────────────────────────
        logger.info(
            f"[{ticker}] Data ready → "
            f"ohlcv={len(kline)} bars | "
            f"quote={quote_data.get('source','?')} | "
            f"order_book=none | "
            f"news={len(news)} articles | "
            f"fundamentals={'yes' if fundamentals else 'no'}"
        )

        # ── 7. Volume info for signal engine filtering ────────────────────
        volume    = quote_data.get("volume") or stock.get("volume")
        vol_info: dict[str, Any] = {"volume": volume, "avg_volume_20d": None}
        if kline:
            vols = [b.get("volume") for b in kline if b.get("volume")]
            if vols:
                vol_info["avg_volume_20d"] = sum(vols) / len(vols)

        # ── 8. LLM Call B — generate signal (with fundamentals) ───────────
        signal = await self.analyst.analyse_stock(
            ticker=ticker,
            name=name,
            ohlcv=kline,
            quote=quote_data,
            order_book={},   # no order book without Moomoo L2
            news=news,
            fundamentals=fundamentals,
        )
        return signal, vol_info

    # ------------------------------------------------------------------
    # Scheduled task: 12:00 — Midday refresh
    # ------------------------------------------------------------------

    async def task_midday_refresh(self) -> None:
        if not is_trading_day():
            return
        logger.info("=== TASK: Midday Refresh (12:00) ===")
        await self.task_generate_signals(SESSION_MORNING)
        await self._check_and_send_exits()

    # ------------------------------------------------------------------
    # Scheduled task: 16:30 — EOD sell reminder
    # ------------------------------------------------------------------

    async def task_eod_reminder(self) -> None:
        if not is_trading_day():
            return
        logger.info("=== TASK: EOD Sell Reminder (16:30) ===")
        open_sigs = self.signal_engine.get_all_open_signals()
        if open_sigs:
            await self.telegram.send_eod_reminder(open_sigs)

    # ------------------------------------------------------------------
    # Scheduled task: 17:15 — Daily summary
    # ------------------------------------------------------------------

    async def task_daily_summary(self) -> None:
        if not is_trading_day():
            return
        logger.info("=== TASK: Daily Summary (17:15) ===")
        self.signal_engine.expire_all()
        summary = self.signal_engine.get_daily_summary()
        await self.telegram.send_daily_summary(summary)
        self._log_to_file("daily_summary", summary)
        self.state.selected_stocks = []

    # ------------------------------------------------------------------
    # Watchlist analysis — runs hourly + on-demand from web UI
    # ------------------------------------------------------------------

    async def task_analyze_watchlist(self) -> None:
        """Analyse all watchlist stocks and broadcast results to the web UI."""
        if self._stopped:
            logger.info("Bot is stopped — skipping watchlist analysis")
            return

        watchlist = self.watchlist_mgr.get_all()
        if not watchlist:
            await self.ws_manager.broadcast({
                "type": "status",
                "data": {"message": "Watchlist is empty — add stocks via the web UI", "in_progress": False},
            })
            return

        settings      = self.watchlist_mgr.get_settings()
        telegram_on   = settings.get("telegram_enabled", True)
        total         = len(watchlist)

        logger.info(f"=== TASK: Watchlist Analysis ({total} stocks, telegram={'on' if telegram_on else 'off'}) ===")

        # Track this task so it can be cancelled via /api/bot/stop
        self._current_task = asyncio.current_task()

        # Clear previous watchlist signals from both the store and the UI
        self.signal_store.clear_by_source("watchlist")
        await self.ws_manager.broadcast({"type": "clear_watchlist"})

        await self.ws_manager.broadcast({
            "type": "status",
            "data": {"message": f"Analysing {total} watchlist stock(s)…", "in_progress": True},
        })

        # Fetch SGX stock list for price data (use cache — don't refetch if fresh)
        try:
            stock_list = await self.scanner.get_stock_list()
            by_ticker  = {s["ticker"]: s for s in stock_list}
        except Exception:
            by_ticker = {}

        completed = 0
        try:
            for wl_entry in watchlist:
                # Check stop flag before each stock
                if self._stopped:
                    logger.info(f"[Watchlist] Stop requested — halting after {completed}/{total} stocks")
                    await self.ws_manager.broadcast({
                        "type": "status",
                        "data": {"message": f"Stopped — {completed}/{total} stock(s) analysed", "in_progress": False},
                    })
                    return

                ticker = wl_entry["ticker"]
                name   = wl_entry["name"]
                stock  = {**by_ticker.get(ticker, {}), "ticker": ticker, "name": name}

                await self.ws_manager.broadcast({
                    "type": "status",
                    "data": {"message": f"Analysing {ticker} ({completed+1}/{total})…", "in_progress": True},
                })

                try:
                    result = await self._analyse_single_stock(stock)
                    if result:
                        signal, _ = result
                        if signal:
                            stored = self.signal_store.add(signal.model_dump(), source="watchlist")
                            await self.ws_manager.broadcast({"type": "signal", "data": stored})
                            if telegram_on and not self.dry_run:
                                await self.telegram.send_signal(signal)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(f"[Watchlist] {ticker} analysis failed: {exc}")

                completed += 1

        except asyncio.CancelledError:
            logger.info(f"[Watchlist] Analysis cancelled after {completed}/{total} stocks")
            await self.ws_manager.broadcast({
                "type": "status",
                "data": {"message": f"Analysis cancelled — {completed}/{total} stock(s) done", "in_progress": False},
            })
            return
        finally:
            self._current_task = None

        finished_at = datetime.now(tz=SGT).strftime("%d %b %Y, %H:%M SGT")
        await self.ws_manager.broadcast({
            "type": "status",
            "data": {
                "message": f"Last run: {finished_at} — {completed} stock(s) analysed",
                "in_progress": False,
                "finished_at": finished_at,
            },
        })
        logger.info(f"Watchlist analysis complete — {completed}/{total} stocks processed")

    # ------------------------------------------------------------------
    # Exit check — uses provider quote + Yahoo Finance for live prices
    # ------------------------------------------------------------------

    async def _check_and_send_exits(self) -> None:
        """Check open signals against live prices; fire exit alerts if hit."""
        open_sigs = self.signal_engine.get_all_open_signals()
        if not open_sigs:
            return

        tickers = [s.signal.ticker for s in open_sigs]
        logger.info(f"[ExitCheck] Checking live prices for {len(tickers)} open signals: {tickers}")

        # Fetch live prices: try provider registry first, then Yahoo Finance
        live_prices: dict[str, float] = {}
        for ticker in tickers:
            try:
                q = await self.provider_registry.fetch_quote(ticker)
                if q and q.get("last_price"):
                    live_prices[ticker] = float(q["last_price"])
                    continue
            except Exception:
                pass

        # Fall back to yfinance for any tickers still missing
        missing = [t for t in tickers if t not in live_prices]
        if missing:
            try:
                refreshed = await self.scanner.refresh_prices(
                    [{"ticker": t} for t in missing]
                )
                for s in refreshed:
                    if s.get("last_price"):
                        live_prices[s["ticker"]] = float(s["last_price"])
            except Exception as exc:
                logger.warning(f"[ExitCheck] Yahoo Finance refresh failed: {exc}")

        if not live_prices:
            logger.debug("[ExitCheck] No live prices available — skipping exit check")
            return

        logger.info(
            f"[ExitCheck] Live prices: "
            + " | ".join(f"{t}={p}" for t, p in live_prices.items())
        )
        exits = self.signal_engine.check_exits(live_prices)
        for open_sig, reason in exits:
            await self.telegram.send_exit_alert(open_sig, reason)

    # ------------------------------------------------------------------
    # Market context
    # ------------------------------------------------------------------

    async def _get_market_context(self) -> str:
        """Fetch STI index data from Yahoo Finance for the LLM stock selection prompt."""
        import aiohttp

        context_parts: list[str] = []

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ESTI?interval=1d&range=2d"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        result = data.get("chart", {}).get("result", [{}])[0]
                        closes = (
                            result
                            .get("indicators", {})
                            .get("quote", [{}])[0]
                            .get("close", [])
                        )
                        if len(closes) >= 2 and closes[-2] and closes[-1]:
                            chg  = (closes[-1] - closes[-2]) / closes[-2] * 100
                            sign = "+" if chg >= 0 else ""
                            context_parts.append(
                                f"STI index: {closes[-1]:.0f} ({sign}{chg:.1f}% yesterday)"
                            )
        except Exception:
            pass

        if not context_parts:
            context_parts.append("No real-time index data available.")

        today_str = datetime.now(tz=SGT).strftime("%A, %d %B %Y")
        context_parts.insert(0, f"Date: {today_str}")
        return " | ".join(context_parts)

    # ------------------------------------------------------------------
    # File logging
    # ------------------------------------------------------------------

    def _log_to_file(self, event: str, data: Any) -> None:
        cache_dir = self.cfg.get("cache", {}).get("dir", "data/cache")
        today     = date.today().isoformat()
        path      = os.path.join(cache_dir, f"{today}_{event}.json")
        try:
            with open(path, "w") as f:
                json.dump(
                    {"event": event, "timestamp": datetime.now(tz=SGT).isoformat(), "data": data},
                    f, indent=2,
                )
        except Exception as exc:
            logger.warning(f"Failed to write log file {path}: {exc}")

    # ------------------------------------------------------------------
    # Loguru setup
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_logging(log_dir: str, level: str = "INFO") -> None:
        logger.remove()
        logger.add(
            sys.stdout,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            colorize=True,
        )
        logger.add(
            os.path.join(log_dir, "sgx_bot_{time:YYYY-MM-DD}.log"),
            level="DEBUG",
            rotation="00:00",
            retention="30 days",
            compression="gz",
        )


# ---------------------------------------------------------------------------
# Singapore REIT master list
# Tickers are SGX stock codes (e.g. "A17U"); names are display names.
# Update annually as mergers / de-listings occur.
# ---------------------------------------------------------------------------

SGX_REITS: list[dict[str, str]] = [
    # ── Large-cap / Blue-chip ──────────────────────────────────────────────
    {"ticker": "A17U",  "name": "Ascendas REIT"},
    {"ticker": "C38U",  "name": "CapitaLand Integrated Commercial Trust"},
    {"ticker": "ME8U",  "name": "Mapletree Industrial Trust"},
    {"ticker": "M44U",  "name": "Mapletree Logistics Trust"},
    {"ticker": "N2IU",  "name": "Mapletree Pan Asia Commercial Trust"},
    # ── Mid-cap ───────────────────────────────────────────────────────────
    {"ticker": "CY6U",  "name": "CapitaLand Ascott Trust"},
    {"ticker": "J69U",  "name": "Frasers Centrepoint Trust"},
    {"ticker": "BUOU",  "name": "Frasers Logistics & Commercial Trust"},
    {"ticker": "K71U",  "name": "Keppel REIT"},
    {"ticker": "T82U",  "name": "Suntec REIT"},
    {"ticker": "P40U",  "name": "Parkway Life REIT"},
    {"ticker": "J91U",  "name": "ESR-LOGOS REIT"},
    # ── Smaller S-REITs ───────────────────────────────────────────────────
    {"ticker": "AW9U",  "name": "AIMS APAC REIT"},
    {"ticker": "RW0U",  "name": "Starhill Global REIT"},
    {"ticker": "SK6U",  "name": "Sabana Industrial REIT"},
    {"ticker": "OXMU",  "name": "OUE REIT"},
    {"ticker": "SV3U",  "name": "Sasseur REIT"},
    {"ticker": "EC1N",  "name": "CapitaLand China Trust"},
    {"ticker": "BWCU",  "name": "Daiwa House Logistics Trust"},
    {"ticker": "DIOU",  "name": "Digital Core REIT"},
    {"ticker": "LREIT", "name": "Lendlease Global Commercial REIT"},
    {"ticker": "CWBU",  "name": "Cromwell European REIT"},
    # ── US-focused S-REITs ────────────────────────────────────────────────
    {"ticker": "AJBU",  "name": "Keppel Pacific Oak US REIT"},
    {"ticker": "42SK",  "name": "Prime US REIT"},
    {"ticker": "MXNU",  "name": "Manulife US REIT"},
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, updates: dict) -> None:
    """Recursively merge *updates* into *base* in-place (updates wins on conflict)."""
    for k, v in updates.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# FastAPI health-check app
# ---------------------------------------------------------------------------

def create_health_app(bot: TradingBot) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(title="SGX Trading Bot", lifespan=lifespan)

    # ── Static files + root UI ─────────────────────────────────────────
    _static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(_static_dir):
        app.mount("/static", StaticFiles(directory=_static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def ui():
        return FileResponse(os.path.join(_static_dir, "index.html"))

    @app.get("/config", include_in_schema=False)
    async def config_ui():
        return FileResponse(os.path.join(_static_dir, "config.html"))

    # ── WebSocket ──────────────────────────────────────────────────────
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await bot.ws_manager.connect(ws)
        # Send current state to the newly connected client
        try:
            await ws.send_json({
                "type": "settings_updated",
                "data": bot.watchlist_mgr.get_settings(),
            })
            await ws.send_json({
                "type": "watchlist_updated",
                "data": {"watchlist": bot.watchlist_mgr.get_all()},
            })
            await ws.send_json({
                "type": "bot_status",
                "data": {"stopped": bot._stopped},
            })
        except Exception:
            pass
        async def _keep_alive():
            try:
                while True:
                    await asyncio.sleep(25)
                    await ws.send_json({"type": "ping"})
            except Exception:
                pass

        ping_task = asyncio.create_task(_keep_alive())
        try:
            while True:
                await ws.receive_text()   # keep alive; client sends nothing meaningful
        except (WebSocketDisconnect, Exception):
            bot.ws_manager.disconnect(ws)
        finally:
            ping_task.cancel()

    # ── Health / legacy ────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        return {
            "status":          "running",
            "trading_day":     is_trading_day(),
            "selected_stocks": len(bot.state.selected_stocks),
            "last_run":        bot.state.last_run,
            "uptime_seconds":  (datetime.now(tz=SGT) - bot.state.start_time).total_seconds(),
            "providers":       bot.provider_registry.status(),
            "ws_clients":      bot.ws_manager.client_count,
            "watchlist_count": len(bot.watchlist_mgr.get_all()),
        }

    @app.get("/signals")
    async def signals_legacy():
        return bot.signal_engine.get_daily_summary()

    # ── Watchlist API ──────────────────────────────────────────────────
    @app.get("/api/watchlist")
    async def get_watchlist():
        return bot.watchlist_mgr.get_all()

    @app.post("/api/watchlist", status_code=201)
    async def add_to_watchlist(req: WatchlistAddRequest):
        entry = bot.watchlist_mgr.add(req.ticker, req.name)
        await bot.ws_manager.broadcast({
            "type": "watchlist_updated",
            "data": {"watchlist": bot.watchlist_mgr.get_all()},
        })
        return entry

    @app.delete("/api/watchlist/{ticker}")
    async def remove_from_watchlist(ticker: str):
        removed = bot.watchlist_mgr.remove(ticker.upper())
        if not removed:
            return JSONResponse(status_code=404, content={"detail": f"{ticker} not in watchlist"})
        await bot.ws_manager.broadcast({
            "type": "watchlist_updated",
            "data": {"watchlist": bot.watchlist_mgr.get_all()},
        })
        return {"removed": ticker.upper()}

    # ── Signals API ────────────────────────────────────────────────────
    @app.get("/api/signals")
    async def get_signals(limit: int = 100):
        return bot.signal_store.get_all(limit=limit)

    # ── On-demand analysis trigger ─────────────────────────────────────
    @app.post("/api/analyze")
    async def trigger_analyze():
        if not bot.watchlist_mgr.get_all():
            return JSONResponse(status_code=400, content={"detail": "Watchlist is empty"})
        asyncio.create_task(bot.task_analyze_watchlist())
        return {"status": "started", "watchlist_count": len(bot.watchlist_mgr.get_all())}

    # ── Settings API ───────────────────────────────────────────────────
    @app.get("/api/settings")
    async def get_settings():
        return bot.watchlist_mgr.get_settings()

    @app.put("/api/settings")
    async def update_settings(req: SettingsUpdateRequest):
        updates = req.model_dump(exclude_none=True)
        settings = bot.watchlist_mgr.update_settings(updates)
        await bot.ws_manager.broadcast({"type": "settings_updated", "data": settings})
        return settings

    # ── Bot stop / resume ─────────────────────────────────────────────
    @app.post("/api/bot/stop")
    async def stop_bot():
        """Gracefully stop any running analysis and prevent new ones from starting."""
        bot._stopped = True
        # Cancel the active asyncio task if one is running
        task = bot._current_task
        if task and not task.done():
            task.cancel()
            logger.info("Bot stop requested — active analysis task cancelled")
        else:
            logger.info("Bot stop requested — no active task running")
        await bot.ws_manager.broadcast({"type": "bot_status", "data": {"stopped": True}})
        return {"status": "stopped"}

    @app.post("/api/bot/resume")
    async def resume_bot():
        """Clear the stop flag so analyses can run again."""
        bot._stopped = False
        logger.info("Bot resumed")
        await bot.ws_manager.broadcast({"type": "bot_status", "data": {"stopped": False}})
        return {"status": "running"}

    @app.get("/api/bot/status")
    async def bot_status():
        return {
            "stopped":       bot._stopped,
            "active_task":   bot._current_task is not None and not bot._current_task.done(),
        }

    # ── REIT list ──────────────────────────────────────────────────────
    @app.get("/api/reits")
    async def get_reits():
        """Return the master list of SGX-listed REITs."""
        return SGX_REITS

    @app.post("/api/watchlist/add-reits", status_code=201)
    async def add_all_reits():
        """Bulk-add all SGX REITs to the watchlist (skips duplicates)."""
        added = []
        for reit in SGX_REITS:
            entry = bot.watchlist_mgr.add(reit["ticker"], reit["name"])
            added.append(entry)
        await bot.ws_manager.broadcast({
            "type": "watchlist_updated",
            "data": {"watchlist": bot.watchlist_mgr.get_all()},
        })
        return {"added": len(added), "watchlist": bot.watchlist_mgr.get_all()}

    # ── On-demand scheduled analysis trigger ───────────────────────────
    @app.post("/api/analyze/scheduled")
    async def trigger_scheduled():
        """Manually trigger scheduled signal generation (bypasses trading-day check)."""
        asyncio.create_task(
            bot.task_generate_signals(SESSION_MORNING, skip_day_check=True)
        )
        return {"status": "started", "session": SESSION_MORNING}

    # ── Stock search (for watchlist autocomplete) ──────────────────────
    @app.get("/api/stocks/search")
    async def search_stocks(q: str = ""):
        if not q or len(q) < 1:
            return []
        q_lower = q.lower()

        # 1. Search the live SGX stock list (may be empty/stale on weekends)
        try:
            stocks = await bot.scanner.get_stock_list()
        except Exception:
            stocks = []

        seen: set[str] = set()
        matches: list[dict] = []

        for s in stocks:
            ticker = s["ticker"].upper()
            name   = (s.get("name") or "").strip()
            if q_lower in ticker.lower() or q_lower in name.lower():
                seen.add(ticker)
                matches.append({"ticker": ticker, "name": name, "reit": False})

        # 2. Always overlay the hardcoded REIT list so REITs are searchable
        #    even on weekends / when SGX cache is stale or filtered out.
        for r in SGX_REITS:
            ticker = r["ticker"].upper()
            if ticker not in seen:
                if q_lower in ticker.lower() or q_lower in r["name"].lower():
                    matches.append({"ticker": ticker, "name": r["name"], "reit": True})

        # Sort: REITs that match ticker prefix first, then name matches
        def rank(m: dict) -> tuple:
            t = m["ticker"].lower()
            n = m["name"].lower()
            return (
                0 if t.startswith(q_lower) else 1,   # ticker-prefix first
                0 if m.get("reit") else 1,             # REITs bubble up within tier
                n,
            )

        matches.sort(key=rank)
        return matches[:15]

    # ── Fundamental analysis ───────────────────────────────────────────
    @app.get("/api/fundamentals/{ticker}")
    async def get_fundamentals(ticker: str):
        """
        Return fundamental data for a single SGX ticker.
        Cached on disk for 24 h; first call may take ~3 s (yfinance network).
        """
        data = await bot.fundamentals.fetch(ticker.upper())
        if "error" in data:
            return JSONResponse(status_code=404, content=data)
        return data

    @app.get("/api/fundamentals")
    async def get_fundamentals_batch(tickers: str = ""):
        """
        Return fundamentals for a comma-separated list of tickers.
        E.g. /api/fundamentals?tickers=D05,O39,A17U
        """
        if not tickers.strip():
            return {}
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        return await bot.fundamentals.fetch_batch(ticker_list)

    @app.delete("/api/fundamentals/cache/{ticker}")
    async def clear_fundamentals_cache(ticker: str):
        """Force-clear the cached fundamental data for one ticker."""
        import glob as _glob
        path = bot.fundamentals._cache_path(ticker.upper())
        try:
            os.remove(path)
            return {"cleared": ticker.upper()}
        except FileNotFoundError:
            return JSONResponse(status_code=404, content={"detail": "No cache entry found"})

    # ── Config read / write ────────────────────────────────────────────
    @app.get("/api/config")
    async def get_config():
        """Return the live config dict (bot.cfg) as JSON."""
        return bot.cfg

    @app.put("/api/config")
    async def save_config(request: Request):
        """
        Accept a full or partial config dict, deep-merge into bot.cfg,
        write to config.yaml, and apply what can be applied immediately.
        """
        try:
            updates = await request.json()
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": f"Invalid JSON: {exc}"})

        # Normalise fallback_models: UI sends a comma-string, YAML expects a list
        if isinstance(updates.get("llm", {}).get("fallback_models"), str):
            raw = updates["llm"]["fallback_models"]
            updates["llm"]["fallback_models"] = [m.strip() for m in raw.split(",") if m.strip()]

        # Coerce numeric fields that the browser might send as strings
        flt = updates.get("filters", {})
        for key in ("min_confidence", "min_volume_ratio"):
            if key in flt:
                try: flt[key] = float(flt[key])
                except (TypeError, ValueError): pass
        for key in ("max_signals_per_session", "pre_filter_top_n", "max_llm_select", "min_turnover_sgd"):
            if key in flt:
                try: flt[key] = int(flt[key])
                except (TypeError, ValueError): pass
        if "request_timeout" in updates.get("llm", {}):
            try: updates["llm"]["request_timeout"] = int(updates["llm"]["request_timeout"])
            except (TypeError, ValueError): pass
        for pname, pdata in updates.get("providers", {}).items():
            if isinstance(pdata, dict):
                for lkey in ("limit", "items"):
                    if lkey in pdata:
                        try: pdata[lkey] = int(pdata[lkey])
                        except (TypeError, ValueError): pass

        # Deep-merge into the live config
        _deep_merge(bot.cfg, updates)

        # ── Persist to disk ───────────────────────────────────────────
        if bot.config_path:
            try:
                with open(bot.config_path, "w") as f:
                    yaml.dump(bot.cfg, f, default_flow_style=False,
                              allow_unicode=True, sort_keys=False)
            except Exception as exc:
                return JSONResponse(status_code=500, content={"detail": f"Save failed: {exc}"})

        # ── Apply immediate effects (no restart needed) ───────────────
        # 1. Telegram credentials
        tg_cfg = bot.cfg.get("telegram", {})
        bot.telegram.bot_token = tg_cfg.get("bot_token", "")
        bot.telegram.chat_id   = str(tg_cfg.get("chat_id", ""))

        # 2. Signal engine thresholds
        flt_live = bot.cfg.get("filters", {})
        bot.signal_engine.min_confidence       = float(flt_live.get("min_confidence", 0.65))
        bot.signal_engine.min_volume_ratio     = float(flt_live.get("min_volume_ratio", 1.2))
        bot.signal_engine.max_signals_per_session = int(flt_live.get("max_signals_per_session", 5))

        logger.info("Config updated and saved via web UI")
        return {"status": "saved", "config": bot.cfg}

    return app


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def setup_scheduler(bot: TradingBot, cfg: dict[str, Any]) -> AsyncIOScheduler:
    tz       = ZoneInfo(cfg.get("scheduler", {}).get("timezone", "Asia/Singapore"))
    sched_cfg = cfg.get("scheduler", {})

    scheduler = AsyncIOScheduler(timezone=tz)

    def parse_time(t: str) -> tuple[int, int]:
        h, m = t.split(":")
        return int(h), int(m)

    h, m = parse_time(sched_cfg.get("morning_scan",    "08:30"))
    scheduler.add_job(bot.task_stock_selection,
                      CronTrigger(hour=h, minute=m, timezone=tz), id="stock_selection")

    h, m = parse_time(sched_cfg.get("morning_signal",  "09:00"))
    scheduler.add_job(
        lambda: asyncio.create_task(bot.task_generate_signals(SESSION_MORNING)),
        CronTrigger(hour=h, minute=m, timezone=tz), id="morning_signals",
    )

    h, m = parse_time(sched_cfg.get("midday_refresh",  "12:00"))
    scheduler.add_job(bot.task_midday_refresh,
                      CronTrigger(hour=h, minute=m, timezone=tz), id="midday_refresh")

    h, m = parse_time(sched_cfg.get("afternoon_signal","14:00"))
    scheduler.add_job(
        lambda: asyncio.create_task(bot.task_generate_signals(SESSION_AFTERNOON)),
        CronTrigger(hour=h, minute=m, timezone=tz), id="afternoon_signals",
    )

    h, m = parse_time(sched_cfg.get("eod_reminder",    "16:30"))
    scheduler.add_job(bot.task_eod_reminder,
                      CronTrigger(hour=h, minute=m, timezone=tz), id="eod_reminder")

    h, m = parse_time(sched_cfg.get("daily_summary",   "17:15"))
    scheduler.add_job(bot.task_daily_summary,
                      CronTrigger(hour=h, minute=m, timezone=tz), id="daily_summary")

    # Price exit check every 5 minutes during market hours
    scheduler.add_job(
        bot._check_and_send_exits,
        CronTrigger(minute="*/5", hour="9-12,14-17", timezone=tz),
        id="exit_check",
    )

    # Watchlist analysis — every hour (configurable interval read at runtime)
    scheduler.add_job(
        bot.task_analyze_watchlist,
        CronTrigger(minute=0, timezone=tz),   # top of every hour
        id="watchlist_analysis",
    )

    return scheduler


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    global _force_trading_day
    _force_trading_day = args.force
    cfg = load_config(args.config)
    bot = TradingBot(cfg, dry_run=args.dry_run)
    bot.config_path = args.config   # stored so /api/config can save edits

    # Validate Telegram credentials immediately — fail fast with a clear message
    await bot.telegram.verify()

    if args.once:
        logger.info("Running single cycle (--once mode)")
        await bot.task_stock_selection()
        await bot.task_generate_signals(SESSION_MORNING)
        await bot.task_daily_summary()
        return

    scheduler = setup_scheduler(bot, cfg)
    scheduler.start()
    logger.info("Scheduler started")

    import uvicorn
    health_cfg = cfg.get("health_check", {})
    app    = create_health_app(bot)
    config = uvicorn.Config(
        app,
        host=health_cfg.get("host", "0.0.0.0"),
        port=health_cfg.get("port", 8080),
        log_level="warning",
    )
    server = uvicorn.Server(config)

    await bot.telegram.send_status("SGX Trading Bot started. Next event: 08:30 SGT stock selection.")
    logger.info("SGX Trading Bot running. Press Ctrl+C to stop.")

    try:
        await server.serve()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
        scheduler.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="SGX Day Trading Signal Bot")
    parser.add_argument("--config",   default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print signals to stdout instead of sending Telegram messages")
    parser.add_argument("--once",     action="store_true",
                        help="Run a single full cycle immediately (for testing)")
    parser.add_argument("--force",    action="store_true",
                        help="Bypass the trading day check (useful for weekend testing)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
