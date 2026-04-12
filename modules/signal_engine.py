"""
signal_engine.py — Signal aggregation, filtering, and ranking.

Responsibilities:
  - Apply hard quality filters to raw LLM signals
  - Rank by (confidence × expected_return)
  - Return top N signals per session
  - Track open signals and emit SELL alerts when price hits target/stop-loss
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from .llm_analyst import TradingSignal


# ---------------------------------------------------------------------------
# Session type
# ---------------------------------------------------------------------------

SESSION_MORNING = "morning"    # 09:00 – 12:00 SGT
SESSION_AFTERNOON = "afternoon"  # 14:00 – 17:00 SGT


# ---------------------------------------------------------------------------
# Open-signal tracker (in-memory, persisted to JSON for restart resilience)
# ---------------------------------------------------------------------------

@dataclass
class OpenSignal:
    signal: TradingSignal
    session: str
    issued_at: str
    current_price: Optional[float] = None
    status: str = "open"   # open | hit_target | hit_stop | expired


# ---------------------------------------------------------------------------
# SignalEngine
# ---------------------------------------------------------------------------

class SignalEngine:
    def __init__(
        self,
        min_confidence: float = 0.65,
        min_volume_ratio: float = 1.2,
        max_signals_per_session: int = 5,
        cache_dir: str = "data/cache",
    ):
        self.min_confidence = min_confidence
        self.min_volume_ratio = min_volume_ratio
        self.max_signals = max_signals_per_session
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # ticker → OpenSignal
        self._open_signals: dict[str, OpenSignal] = {}
        self._load_open_signals()

    # ------------------------------------------------------------------
    # Main entry: filter + rank + persist
    # ------------------------------------------------------------------

    def process_signals(
        self,
        raw_signals: list[TradingSignal],
        volume_data: dict[str, dict[str, Any]],  # ticker → {volume, avg_volume_20d}
        session: str = SESSION_MORNING,
    ) -> list[TradingSignal]:
        """
        Filter raw LLM signals, rank them, and return the top N.

        Args:
            raw_signals:  List of TradingSignal objects from llm_analyst
            volume_data:  Dict mapping ticker to volume metrics from Moomoo/scanner
            session:      "morning" or "afternoon"

        Returns:
            Filtered + ranked list of up to max_signals signals.
        """
        passed: list[TradingSignal] = []
        for sig in raw_signals:
            reason = self._filter(sig, volume_data.get(sig.ticker, {}))
            if reason:
                logger.debug(f"[{sig.ticker}] Filtered out: {reason}")
            else:
                passed.append(sig)

        ranked = sorted(passed, key=self._score, reverse=True)
        top = ranked[: self.max_signals]

        # Register as open signals
        now = datetime.now(tz=timezone.utc).isoformat()
        for sig in top:
            self._open_signals[sig.ticker] = OpenSignal(
                signal=sig, session=session, issued_at=now
            )
        self._persist_open_signals()

        logger.info(
            f"Signal engine: {len(raw_signals)} raw → "
            f"{len(passed)} passed filters → {len(top)} selected"
        )
        return top

    # ------------------------------------------------------------------
    # Price update + SELL alert detection
    # ------------------------------------------------------------------

    def check_exits(
        self, live_prices: dict[str, float]
    ) -> list[tuple[OpenSignal, str]]:
        """
        Compare live prices against open signal targets / stop-losses.

        Args:
            live_prices: ticker → current last price

        Returns:
            List of (OpenSignal, reason) tuples for signals that should exit.
            Reason is "target" or "stop_loss".
        """
        alerts: list[tuple[OpenSignal, str]] = []
        for ticker, open_sig in list(self._open_signals.items()):
            if open_sig.status != "open":
                continue
            price = live_prices.get(ticker)
            if price is None:
                continue

            open_sig.current_price = price
            sig = open_sig.signal

            if sig.action == "BUY":
                if sig.target_price and price >= sig.target_price:
                    open_sig.status = "hit_target"
                    alerts.append((open_sig, "target"))
                    logger.info(f"[{ticker}] TARGET hit at {price:.3f}")
                elif sig.stop_loss and price <= sig.stop_loss:
                    open_sig.status = "hit_stop"
                    alerts.append((open_sig, "stop_loss"))
                    logger.info(f"[{ticker}] STOP LOSS hit at {price:.3f}")

            elif sig.action == "SELL":
                if sig.target_price and price <= sig.target_price:
                    open_sig.status = "hit_target"
                    alerts.append((open_sig, "target"))
                elif sig.stop_loss and price >= sig.stop_loss:
                    open_sig.status = "hit_stop"
                    alerts.append((open_sig, "stop_loss"))

        self._persist_open_signals()
        return alerts

    def get_all_open_signals(self) -> list[OpenSignal]:
        return [s for s in self._open_signals.values() if s.status == "open"]

    def expire_all(self) -> None:
        """Mark all open signals as expired (called at EOD)."""
        for sig in self._open_signals.values():
            if sig.status == "open":
                sig.status = "expired"
        self._persist_open_signals()

    def get_daily_summary(self) -> list[dict[str, Any]]:
        """Return a summary of all signals issued today for the daily report."""
        return [
            {
                "ticker": s.signal.ticker,
                "action": s.signal.action,
                "entry_price": s.signal.entry_price,
                "target_price": s.signal.target_price,
                "stop_loss": s.signal.stop_loss,
                "confidence": s.signal.confidence,
                "strategy": s.signal.strategy,
                "session": s.session,
                "issued_at": s.issued_at,
                "status": s.status,
                "current_price": s.current_price,
            }
            for s in self._open_signals.values()
        ]

    # ------------------------------------------------------------------
    # Hard filters
    # ------------------------------------------------------------------

    def _filter(
        self, sig: TradingSignal, vol: dict[str, Any]
    ) -> Optional[str]:
        """Return a rejection reason string, or None if the signal passes."""

        # Confidence threshold
        if sig.confidence < self.min_confidence:
            return f"confidence {sig.confidence:.0%} < {self.min_confidence:.0%}"

        if sig.action == "BUY":
            # Target must be above entry
            if sig.entry_price and sig.target_price:
                if sig.target_price <= sig.entry_price:
                    return "target_price not above entry_price"
            # Stop loss must be set and below entry
            if sig.entry_price and sig.stop_loss:
                if sig.stop_loss >= sig.entry_price:
                    return "stop_loss not below entry_price"
            elif not sig.stop_loss:
                return "stop_loss not set"

        if sig.action == "SELL":
            if sig.entry_price and sig.target_price:
                if sig.target_price >= sig.entry_price:
                    return "target_price not below entry_price for SELL"

        # Volume filter
        volume = vol.get("volume")
        avg_volume = vol.get("avg_volume_20d")
        if volume is not None and avg_volume and avg_volume > 0:
            ratio = volume / avg_volume
            if ratio < self.min_volume_ratio:
                return f"volume ratio {ratio:.2f} < {self.min_volume_ratio:.2f}"

        return None

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _score(sig: TradingSignal) -> float:
        """
        Rank by: confidence × expected_return (absolute % move to target).
        Higher is better.
        """
        if sig.entry_price and sig.target_price and sig.entry_price > 0:
            expected_return = abs(sig.target_price - sig.entry_price) / sig.entry_price
        else:
            expected_return = 0.0
        return sig.confidence * expected_return

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_open_signals(self) -> None:
        path = os.path.join(self.cache_dir, "open_signals.json")
        try:
            data = {}
            for ticker, os_ in self._open_signals.items():
                data[ticker] = {
                    "signal": os_.signal.model_dump(),
                    "session": os_.session,
                    "issued_at": os_.issued_at,
                    "current_price": os_.current_price,
                    "status": os_.status,
                }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning(f"Failed to persist open signals: {exc}")

    def _load_open_signals(self) -> None:
        path = os.path.join(self.cache_dir, "open_signals.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for ticker, d in data.items():
                from .llm_analyst import TradingSignal as TS
                self._open_signals[ticker] = OpenSignal(
                    signal=TS(**d["signal"]),
                    session=d.get("session", ""),
                    issued_at=d.get("issued_at", ""),
                    current_price=d.get("current_price"),
                    status=d.get("status", "open"),
                )
            logger.info(f"Loaded {len(self._open_signals)} open signals from cache")
        except Exception as exc:
            logger.warning(f"Failed to load open signals from cache: {exc}")
