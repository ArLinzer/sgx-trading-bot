"""
telegram_bot.py — Telegram notification sender.

Sends:
  - BUY / SELL / HOLD / WATCH signal messages
  - SELL exit alerts (target hit or stop-loss hit)
  - EOD sell reminder (16:30 SGT)
  - Daily summary (17:15 SGT)

Uses python-telegram-bot v20+ (async).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    from telegram.error import TelegramError

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning(
        "python-telegram-bot not found. "
        "Install it with: pip install python-telegram-bot>=20.0"
    )

from .llm_analyst import TradingSignal
from .signal_engine import OpenSignal


def _sgt_now() -> str:
    """Return current SGT time as a formatted string."""
    from zoneinfo import ZoneInfo
    return datetime.now(tz=ZoneInfo("Asia/Singapore")).strftime("%H:%M SGT")


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, dry_run: bool = False):
        """
        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id:   Target chat or group ID
            dry_run:   If True, print messages to stdout instead of sending
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.dry_run = dry_run
        self._bot: Any = None

        if dry_run:
            logger.info("Telegram: dry-run mode — messages will print to stdout, NOT sent to Telegram")
            return

        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram: python-telegram-bot not installed — messages disabled")
            return

        missing = []
        if not bot_token or bot_token.startswith("YOUR_"):
            missing.append("bot_token")
        if not chat_id or str(chat_id).startswith("YOUR_"):
            missing.append("chat_id")
        if missing:
            logger.error(
                f"Telegram: {', '.join(missing)} not configured in config.yaml — "
                "messages will NOT be sent. "
                "Get your bot token from @BotFather and chat_id from @userinfobot."
            )
            return

        self._bot = Bot(token=bot_token)
        logger.info(f"Telegram: bot initialised (chat_id={chat_id})")

    # ------------------------------------------------------------------
    # Startup connectivity check
    # ------------------------------------------------------------------

    async def verify(self) -> bool:
        """
        Ping the Telegram API at startup to validate credentials.
        Logs a clear pass/fail — call this once from main before the scheduler starts.
        """
        if self.dry_run:
            logger.info("Telegram verify: skipped (dry-run mode)")
            return True
        if not self._bot:
            logger.warning("Telegram verify: skipped (bot not initialised)")
            return False
        try:
            me = await self._bot.get_me()
            logger.info(
                f"Telegram ✅ connected — bot=@{me.username} "
                f"(id={me.id}), sending to chat_id={self.chat_id}"
            )
            return True
        except TelegramError as exc:
            logger.error(f"Telegram ❌ connectivity check failed [{type(exc).__name__}]: {exc}")
            self._log_telegram_hint(exc)
            return False

    # ------------------------------------------------------------------
    # Signal message
    # ------------------------------------------------------------------

    async def send_signal(self, signal: TradingSignal, catalyst: str = "") -> bool:
        """
        Send a formatted trading signal message.

        Args:
            signal:   The TradingSignal to announce
            catalyst: One-line news catalyst summary
        """
        msg = self._format_signal(signal, catalyst)
        return await self._send(msg)

    @staticmethod
    def _esc(text: str) -> str:
        """Escape text for Telegram HTML mode (only &, <, > need escaping)."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _format_signal(self, signal: TradingSignal, catalyst: str = "") -> str:
        action = signal.action
        emoji = {
            "BUY": "🟢",
            "SELL": "🔴",
            "HOLD": "🟡",
            "WATCH": "👀",
        }.get(action, "⚪")

        lines = [f"{emoji} <b>{action} SIGNAL — ${self._esc(signal.ticker)}</b>"]

        if signal.entry_price:
            lines.append("━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"Entry:       S${signal.entry_price:.3f}")
        if signal.target_price and signal.entry_price:
            pct  = (signal.target_price - signal.entry_price) / signal.entry_price * 100
            sign = "+" if pct >= 0 else ""
            lines.append(f"Target:      S${signal.target_price:.3f}  ({sign}{pct:.1f}%)")
        if signal.stop_loss and signal.entry_price:
            pct  = (signal.stop_loss - signal.entry_price) / signal.entry_price * 100
            sign = "+" if pct >= 0 else ""
            lines.append(f"Stop Loss:   S${signal.stop_loss:.3f}  ({sign}{pct:.1f}%)")

        lines.append(f"Confidence:  {signal.confidence:.0%}")
        if signal.strategy:
            lines.append(f"Strategy:    {signal.strategy.replace('_', ' ').title()}")
        lines.append(f"Exit by EOD: {'✅ Yes' if signal.exit_before_eod else '❌ No'}")

        if signal.reasoning:
            lines.append("")
            lines.append(f"📊 <b>Rationale:</b> {self._esc(signal.reasoning)}")

        if catalyst:
            lines.append("")
            lines.append(f"📰 <b>Catalyst:</b> {self._esc(catalyst)}")

        if signal.news_sources:
            src = self._esc(" + ".join(signal.news_sources[:3]))
            lines.append(f"⚡ <b>Source:</b>   {src}")

        lines.append("")
        lines.append(f"<i>[Generated at {_sgt_now()}]</i>")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Exit alerts
    # ------------------------------------------------------------------

    async def send_exit_alert(
        self,
        open_signal: OpenSignal,
        reason: str,
    ) -> bool:
        """
        Send an alert when a signal hits target or stop-loss.

        Args:
            open_signal: The tracked OpenSignal
            reason:      "target" or "stop_loss"
        """
        sig = open_signal.signal
        price = open_signal.current_price

        if reason == "target":
            emoji = "🎯"
            label = "TARGET HIT"
        else:
            emoji = "🛑"
            label = "STOP LOSS HIT"

        pnl_str = ""
        if price and sig.entry_price:
            pnl_pct = (price - sig.entry_price) / sig.entry_price * 100
            sign = "+" if pnl_pct >= 0 else ""
            pnl_str = f"  ({sign}{pnl_pct:.1f}%)"

        lines = [
            f"{emoji} <b>{label} — ${self._esc(sig.ticker)}</b>",
            "━━━━━━━━━━━━━━━━━━━━",
            f"Entry:    S${sig.entry_price:.3f}" if sig.entry_price else "",
            f"Current:  S${price:.3f}{self._esc(pnl_str)}" if price else "",
            "Action:   Close / Exit position",
            "",
            f"<i>[Alert at {_sgt_now()}]</i>",
        ]
        msg = "\n".join(l for l in lines if l)
        return await self._send(msg)

    # ------------------------------------------------------------------
    # EOD sell reminder
    # ------------------------------------------------------------------

    async def send_eod_reminder(self, open_signals: list[OpenSignal]) -> bool:
        """
        Send a sell reminder 30 minutes before close for all open signals.
        """
        if not open_signals:
            return True

        tickers = ", ".join(f"${s.signal.ticker}" for s in open_signals)
        lines = [
            "⏰ <b>EOD SELL REMINDER</b>",
            "━━━━━━━━━━━━━━━━━━━━",
            "Market closes in ~30 minutes.",
            f"Close all open positions: {self._esc(tickers)}",
            "",
            "<i>All signals are intraday — exit before 17:00 SGT.</i>",
            "",
            f"<i>[{_sgt_now()}]</i>",
        ]
        return await self._send("\n".join(lines))

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    async def send_daily_summary(self, summary: list[dict[str, Any]]) -> bool:
        """
        Send the end-of-day summary of all signals issued.
        """
        if not summary:
            msg = (
                "📋 <b>Daily Summary</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "No signals were issued today.\n\n"
                f"<i>[{_sgt_now()}]</i>"
            )
            return await self._send(msg)

        lines = [
            "📋 <b>Daily Summary</b>",
            "━━━━━━━━━━━━━━━━━━━━",
        ]
        for s in summary:
            status_emoji = {
                "hit_target": "✅",
                "hit_stop":   "❌",
                "expired":    "⏱",
                "open":       "🔵",
            }.get(s.get("status", ""), "⚪")

            entry   = s.get("entry_price")
            current = s.get("current_price")
            pnl = ""
            if entry and current:
                pnl_pct = (current - entry) / entry * 100
                sign = "+" if pnl_pct >= 0 else ""
                pnl = f" ({sign}{pnl_pct:.1f}%)"

            lines.append(
                f"{status_emoji} ${self._esc(s['ticker'])}  "
                f"{s['action']}  conf={s['confidence']:.0%}  "
                f"{self._esc(s.get('strategy', 'N/A'))}{self._esc(pnl)}"
            )

        lines += ["", f"<i>[{_sgt_now()}]</i>"]
        return await self._send("\n".join(lines))

    # ------------------------------------------------------------------
    # Health / status message
    # ------------------------------------------------------------------

    async def send_status(self, message: str) -> bool:
        return await self._send(f"ℹ️ {self._esc(message)}")

    # ------------------------------------------------------------------
    # Core send with error handling
    # ------------------------------------------------------------------

    async def _send(self, text: str) -> bool:
        if self.dry_run:
            print("\n" + "=" * 50)
            print("[DRY RUN] Telegram message:")
            print(text)
            print("=" * 50 + "\n")
            return True

        if not TELEGRAM_AVAILABLE or not self._bot:
            logger.warning("Telegram bot not available — message not sent")
            return False

        # Guard: catch missing / placeholder credentials early
        if not self.bot_token or self.bot_token.startswith("YOUR_"):
            logger.error(
                "Telegram bot_token is not configured. "
                "Set telegram.bot_token in config.yaml"
            )
            return False
        if not self.chat_id or str(self.chat_id).startswith("YOUR_"):
            logger.error(
                "Telegram chat_id is not configured. "
                "Set telegram.chat_id in config.yaml"
            )
            return False

        try:
            msg = await self._bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            logger.info(
                f"Telegram ✅ sent (message_id={msg.message_id}, "
                f"chat={self.chat_id}, chars={len(text)})"
            )
            return True
        except TelegramError as exc:
            logger.error(f"Telegram send failed [{type(exc).__name__}]: {exc}")
            self._log_telegram_hint(exc)
            # Fallback: strip HTML tags and retry as plain text
            try:
                import re as _re
                plain = _re.sub(r"<[^>]+>", "", text).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                msg = await self._bot.send_message(
                    chat_id=self.chat_id,
                    text=plain,
                    disable_web_page_preview=True,
                )
                logger.info(
                    f"Telegram ✅ sent (plain fallback, message_id={msg.message_id})"
                )
                return True
            except TelegramError as exc2:
                logger.error(f"Telegram fallback send also failed [{type(exc2).__name__}]: {exc2}")
                return False

    @staticmethod
    def _log_telegram_hint(exc: "TelegramError") -> None:
        """Log a human-readable fix hint for common Telegram errors."""
        msg = str(exc).lower()
        if "unauthorized" in msg:
            logger.error(
                "  → Hint: bot_token is invalid or revoked. "
                "Get a new token from @BotFather on Telegram."
            )
        elif "not found" in msg or "chat not found" in msg:
            logger.error(
                "  → Hint: chat_id is wrong, or the bot has never been started. "
                "Open Telegram, find your bot, and send /start. "
                "Then confirm your chat_id by messaging @userinfobot."
            )
        elif "forbidden" in msg or "blocked" in msg:
            logger.error(
                "  → Hint: The bot was blocked by the user. "
                "Unblock it in Telegram and send /start again."
            )
        elif "flood" in msg or "retry" in msg:
            logger.warning(
                "  → Hint: Telegram rate limit hit. Messages will retry automatically."
            )
