"""
watchlist.py — User watchlist, signal store and WebSocket manager for the web UI.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from loguru import logger


class WatchlistManager:
    """Persistent user-defined watchlist + UI settings."""

    DEFAULT_SETTINGS: dict[str, Any] = {
        "telegram_enabled": True,
        "watchlist_interval_hours": 1,
    }

    def __init__(self, data_dir: str = "data") -> None:
        os.makedirs(data_dir, exist_ok=True)
        self._wl_path  = os.path.join(data_dir, "watchlist.json")
        self._cfg_path = os.path.join(data_dir, "ui_settings.json")
        self._entries: list[dict[str, Any]] = []
        self._settings: dict[str, Any] = dict(self.DEFAULT_SETTINGS)
        self._load()

    # ── persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self._wl_path):
            try:
                with open(self._wl_path) as f:
                    self._entries = json.load(f)
            except Exception:
                self._entries = []
        if os.path.exists(self._cfg_path):
            try:
                with open(self._cfg_path) as f:
                    self._settings = {**self.DEFAULT_SETTINGS, **json.load(f)}
            except Exception:
                pass

    def _save_wl(self) -> None:
        try:
            with open(self._wl_path, "w") as f:
                json.dump(self._entries, f, indent=2)
        except Exception as exc:
            logger.warning(f"[Watchlist] save failed: {exc}")

    def _save_cfg(self) -> None:
        try:
            with open(self._cfg_path, "w") as f:
                json.dump(self._settings, f, indent=2)
        except Exception as exc:
            logger.warning(f"[Watchlist] settings save failed: {exc}")

    # ── CRUD ───────────────────────────────────────────────────────────

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def add(self, ticker: str, name: str) -> dict[str, Any]:
        ticker = ticker.upper().strip()
        existing = next((e for e in self._entries if e["ticker"] == ticker), None)
        if existing:
            return existing
        entry: dict[str, Any] = {
            "ticker":   ticker,
            "name":     name,
            "added_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._entries.append(entry)
        self._save_wl()
        logger.info(f"[Watchlist] + {ticker} ({name})")
        return entry

    def remove(self, ticker: str) -> bool:
        ticker = ticker.upper().strip()
        before = len(self._entries)
        self._entries = [e for e in self._entries if e["ticker"] != ticker]
        if len(self._entries) < before:
            self._save_wl()
            logger.info(f"[Watchlist] - {ticker}")
            return True
        return False

    # ── settings ───────────────────────────────────────────────────────

    def get_settings(self) -> dict[str, Any]:
        return dict(self._settings)

    def update_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        for k, v in updates.items():
            if k in self.DEFAULT_SETTINGS:
                self._settings[k] = v
        self._save_cfg()
        return dict(self._settings)


class SignalStore:
    """In-memory ring buffer of recent signals for the web UI."""

    def __init__(self, max_signals: int = 200) -> None:
        self._signals: list[dict[str, Any]] = []
        self.max_signals = max_signals

    def add(self, signal: dict[str, Any], source: str = "scheduled") -> dict[str, Any]:
        entry = {
            **signal,
            "received_at": datetime.now(tz=timezone.utc).isoformat(),
            "ui_source":   source,   # "scheduled" | "watchlist"
        }
        self._signals.insert(0, entry)
        if len(self._signals) > self.max_signals:
            self._signals = self._signals[: self.max_signals]
        return entry

    def get_all(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._signals[:limit]

    def clear(self) -> None:
        self._signals.clear()

    def clear_by_source(self, source: str) -> int:
        before = len(self._signals)
        self._signals = [s for s in self._signals if s.get("ui_source") != source]
        return before - len(self._signals)


class WSManager:
    """Broadcast JSON payloads to all connected WebSocket clients."""

    def __init__(self) -> None:
        self._conns: list[Any] = []

    async def connect(self, ws: Any) -> None:
        await ws.accept()
        self._conns.append(ws)
        logger.debug(f"[WS] +client  ({len(self._conns)} total)")

    def disconnect(self, ws: Any) -> None:
        try:
            self._conns.remove(ws)
        except ValueError:
            pass
        logger.debug(f"[WS] -client ({len(self._conns)} remaining)")

    async def broadcast(self, payload: dict[str, Any]) -> None:
        dead: list[Any] = []
        for ws in list(self._conns):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def client_count(self) -> int:
        return len(self._conns)
