"""
llm_analyst.py — Local LLM strategy engine via Ollama.

Two LLM calls are implemented:

  Call A — Stock Selection
    Input:  Full SGX stock list + market context
    Output: JSON list of up to 20 tickers to focus on today

  Call B — Strategy per Stock
    Input:  Price/volume data, aggregated news, order book snapshot
    Output: JSON trading signal with action, entry, target, stop-loss, etc.

All prompts explicitly instruct the model to output ONLY valid JSON.
JSON responses are parsed with error handling (markdown-fence stripping,
schema validation via Pydantic).
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

from .web_search import ALL_TOOLS, DDG_AVAILABLE, execute_tool

try:
    import ollama as _ollama  # type: ignore
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama package not found. Install it with: pip install ollama")


# ---------------------------------------------------------------------------
# Pydantic schemas for LLM output validation
# ---------------------------------------------------------------------------

class SelectedStock(BaseModel):
    ticker: str
    name: str = ""
    reason: str = ""
    priority: Literal["high", "medium", "low"] = "medium"


class TradingSignal(BaseModel):
    ticker: str
    action: Literal["BUY", "SELL", "HOLD", "WATCH"] = "WATCH"
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy: str = ""
    reasoning: str = ""
    time_horizon: str = "intraday"
    exit_before_eod: bool = True
    news_sources: List[str] = Field(default_factory=list)

    @field_validator("action", mode="before")
    @classmethod
    def normalise_action(cls, v: Any) -> str:
        """
        Accept common LLM variants and map them to the canonical set.
        E.g. "buy", "Strong Buy", "STRONG_BUY", "BULLISH" → "BUY"
        """
        if v is None:
            return "WATCH"
        s = str(v).strip().upper().replace("-", "_").replace(" ", "_")
        # Map common synonyms
        if s in ("BUY", "STRONG_BUY", "BULLISH", "LONG", "ACCUMULATE", "OUTPERFORM"):
            return "BUY"
        if s in ("SELL", "STRONG_SELL", "BEARISH", "SHORT", "REDUCE", "UNDERPERFORM"):
            return "SELL"
        if s in ("HOLD", "NEUTRAL", "MARKET_PERFORM", "EQUAL_WEIGHT"):
            return "HOLD"
        if s in ("WATCH", "MONITOR", "OBSERVE", "WATCHLIST"):
            return "WATCH"
        # Partial match — try starts-with
        for canon in ("BUY", "SELL", "HOLD", "WATCH"):
            if s.startswith(canon):
                return canon
        return "WATCH"  # safe default

    @field_validator("entry_price", "target_price", "stop_loss", mode="before")
    @classmethod
    def coerce_nullable_price(cls, v: Any) -> Optional[float]:
        """
        Convert non-numeric LLM strings (e.g. 'NEEDS_LIVE_DATA', 'N/A', 'TBD', 'null')
        to None, and parse valid numeric strings into float.
        """
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v) if v != 0 else None
        s = str(v).strip().upper().replace(",", "")
        # Treat these as "no value"
        null_tokens = {"", "NULL", "NONE", "N/A", "NA", "TBD", "TBC",
                       "NEEDS_LIVE_DATA", "NEEDS_DATA", "UNKNOWN", "-", "0"}
        if s in null_tokens:
            return None
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        try:
            f = float(v)
            return max(0.0, min(1.0, f))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("news_sources", mode="before")
    @classmethod
    def coerce_news_sources(cls, v: Any) -> List[str]:
        """Accept None, a plain string, or a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        if isinstance(v, list):
            return [str(x) for x in v if x]
        return []

    @model_validator(mode="after")
    def enforce_eod_exit(self) -> "TradingSignal":
        self.exit_before_eod = True  # Always enforce for day trading
        return self


# ---------------------------------------------------------------------------
# LLM Analyst
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a professional quantitative analyst specialising in SGX (Singapore Exchange) "
    "day trading. You produce concise, data-driven trading signals. "
    "You ALWAYS respond with ONLY valid JSON — no prose, no markdown fences, no explanations "
    "outside the JSON structure."
)

STOCK_SELECTION_PROMPT_TEMPLATE = """You are analysing SGX stocks for intraday trading opportunities today.

MARKET CONTEXT:
{market_context}

TOP {top_n} SGX STOCKS BY VOLUME (ticker, name, sector, market_cap_SGD_millions, last_price, change_pct, volume, price_date, stale):
{stock_list_csv}

IMPORTANT DATA QUALITY NOTES:
- "price_date" shows when the last trade occurred.
- "stale=Y" means the price is more than 5 days old — this stock is illiquid and should be AVOIDED.
- Prefer stocks with price_date = today or yesterday and high volume.

Task: Select EXACTLY {max_select} SGX stocks with the best intraday trading potential today.
Prioritise stocks with:
- High relative volume (unusual activity)
- Upcoming catalysts (earnings, dividends, M&A rumours)
- Technical breakout setups
- Recent news-driven momentum
- Fresh prices (stale=N)

Return ONLY a JSON array. Each element must have exactly these keys:
  "ticker"   : string  — SGX stock code
  "name"     : string  — company name
  "reason"   : string  — 1-2 sentence rationale
  "priority" : string  — one of "high", "medium", "low"

Example:
[
  {{"ticker": "D05", "name": "DBS Group", "reason": "Breaking 52-week high on strong volume after earnings beat.", "priority": "high"}},
  {{"ticker": "O39", "name": "OCBC Bank", "reason": "Sector rotation into financials; near key support.", "priority": "medium"}}
]

IMPORTANT: Output ONLY the JSON array. No other text."""


STRATEGY_PROMPT_TEMPLATE = """You are a day trading analyst specialising in SGX (Singapore Exchange) stocks.

╔══════════════════════════════════════════════════════╗
║  STOCK BEING ANALYSED:  {ticker}  —  {name}          ║
║  Exchange: SGX (Singapore)  |  Currency: SGD         ║
╚══════════════════════════════════════════════════════╝

Your task: synthesise ALL six data sections below into ONE trading signal for {ticker} ({name}).
Every number, price level, and recommendation you output MUST be for {ticker} ONLY.

━━━ 1. PRICE & VOLUME — OHLCV (last 5 days, oldest → newest) ━━━
{ohlcv_json}

━━━ 2. TECHNICAL INDICATORS (pre-computed from OHLCV + quote) ━━━
{technicals_text}

━━━ 3. INTRADAY QUOTE ━━━
{quote_json}

━━━ 4. NEWS & ANNOUNCEMENTS (last 24 h, newest first, with sentiment where available) ━━━
{news_json}

━━━ 5. SENTIMENT ANALYSIS (aggregate across all news providers) ━━━
{sentiment_text}

━━━ 6. FUNDAMENTAL DATA (Yahoo Finance, refreshed daily) ━━━
{fundamentals_text}

━━━ HOW TO WEIGHT EACH DATA SOURCE ━━━
Priority for day trading (highest → lowest):
  1. News catalysts  — earnings, SGX announcements, M&A, regulatory changes → drives intraday moves
  2. Technical setup — price vs SMA, volume ratio, range position → confirms entry timing
  3. Sentiment trend — StockGeist score + 24h trend, news sentiment aggregate → crowd momentum
  4. Intraday quote  — live price, today's range, change % → exact entry/stop calibration
  5. Fundamentals    — quality score, P/E, dividends, debt → VALIDATE or CONTRADICT the signal

SPECIFIC RULES:
- Fundamentals quality score ≥60: raise confidence +0.05; ≤30: lower confidence −0.10 on BUY.
- StockGeist score >0.3 (bullish) + rising 24h trend: supports BUY momentum.
- StockGeist score <−0.3 (bearish) + falling 24h trend: supports SELL / reduces BUY confidence.
- Avg news sentiment >+0.3 (3+ articles): add "news_catalyst" to reasoning. <−0.3: flag as risk.
- SMA crossover (price crosses above SMA20 on above-avg volume): strong BUY signal.
- Volume ratio >2×: confirms breakout or reversal. <0.5×: low conviction, prefer WATCH.
- Ex-dividend within 5 days: consider "dividend_capture" strategy (BUY before, exit same day).
- 52W position >85% + strong fundamentals + positive news: momentum continuation BUY.
- 52W position <15% + good fundamentals + no negative news: potential mean-reversion BUY.
- Debt/Equity >200 (non-financials): downside risk — reduce confidence on BUY by −0.08.
- Beta >1.5: widen stop-loss by ≥1.5× normal range.

Trading constraints:
- Day trading only — all positions MUST be closed before market close (17:00 SGT)
- SGX market hours: 09:00–17:00 SGT (lunch break 12:00–14:00)
- Singapore dollars (SGD)

REMINDER: You are generating a signal for {ticker} ({name}) ONLY.
If any news headline is clearly unrelated to {ticker} or {name}, ignore it entirely.

Task: Synthesise ALL six data sources above into ONE trading signal for {ticker}.
Return ONLY a JSON object with exactly these keys:
  "ticker"          : string  — SGX stock code
  "action"          : string  — one of "BUY", "SELL", "HOLD", "WATCH"
  "entry_price"     : number  — suggested entry price (null if HOLD/WATCH)
  "target_price"    : number  — profit target price (null if HOLD/WATCH)
  "stop_loss"       : number  — stop loss price (null if HOLD/WATCH)
  "confidence"      : number  — confidence score 0.0–1.0 (apply the rules above)
  "strategy"        : string  — one of "momentum", "news_catalyst", "breakout", "mean_reversion", "range_trade", "fundamental_value", "dividend_capture", "sentiment_driven"
  "reasoning"       : string  — 3-4 sentences covering: (a) primary catalyst from news, (b) technical confirmation, (c) sentiment posture, (d) how fundamentals validated or flagged risk
  "time_horizon"    : string  — always "intraday"
  "exit_before_eod" : boolean — always true
  "news_sources"    : array   — names of all sources that materially influenced the signal

Example:
{{
  "ticker": "D05",
  "action": "BUY",
  "entry_price": 38.50,
  "target_price": 39.10,
  "stop_loss": 38.20,
  "confidence": 0.83,
  "strategy": "news_catalyst",
  "reasoning": "SGX announcement of Q3 earnings beat (+12% NII) is the primary catalyst. Price sits 2% above SMA20 on 2.1× avg volume — strong technical confirmation of breakout. StockGeist score +0.41 (bullish, trending up from +0.18 yesterday) and avg news sentiment +0.38 across 6 articles confirm positive crowd momentum. Fundamentals (quality score 77, P/E 15×, ROE 16%) validate the move — stock is fairly valued, not overextended.",
  "time_horizon": "intraday",
  "exit_before_eod": true,
  "news_sources": ["SGX Announcements", "Business Times", "Marketaux", "StockGeist"]
}}

IMPORTANT: Output ONLY the JSON object. No other text."""


class LLMAnalyst:
    def __init__(
        self,
        model: str = "qwen3.5:9b",
        fallback_models: Optional[List[str]] = None,
        ollama_host: str = "http://localhost:11434",
        request_timeout: int = 120,
    ):
        self.model = model
        self.fallback_models = fallback_models or ["mistral", "phi3"]
        self.ollama_host = ollama_host
        self.request_timeout = request_timeout
        self._active_model: Optional[str] = None

    # ------------------------------------------------------------------
    # Call A — Stock Selection
    # ------------------------------------------------------------------

    async def select_stocks(
        self,
        stock_list: list[dict[str, Any]],
        market_context: str = "",
        max_select: int = 20,
    ) -> list[SelectedStock]:
        """
        Ask the LLM to pick up to 20 tickers with the best day-trading potential.

        Args:
            stock_list:     Full SGX stock list from sgx_scanner
            market_context: Brief string describing overnight global market sentiment

        Returns:
            Validated list of SelectedStock objects.
        """
        stock_csv = self._build_stock_csv(stock_list)
        prompt = STOCK_SELECTION_PROMPT_TEMPLATE.format(
            market_context=market_context or "No global market context available.",
            stock_list_csv=stock_csv,
            top_n=len(stock_list),
            max_select=max_select,
        )

        logger.info(f"[LLM] Call A — Stock Selection | model={self.model} | input={len(stock_list)} stocks | max_select={max_select}")
        logger.info(f"[LLM] Market context: {market_context or 'N/A'}")

        raw = await self._call_llm(prompt, label="Stock Selection", think=False, use_tools=False)
        if not raw:
            return []

        parsed = self._parse_json(raw, expected="list")
        if not isinstance(parsed, list):
            logger.warning(f"[LLM] Stock selection output was not a JSON list: {raw[:300]}")
            return []

        results: list[SelectedStock] = []
        for item in parsed:
            try:
                results.append(SelectedStock(**item))
            except Exception as exc:
                logger.debug(f"[LLM] SelectedStock validation error: {exc} | item={item}")

        logger.info(f"[LLM] Stock selection complete — {len(results)} tickers chosen:")
        for s in results:
            logger.info(f"  [{s.priority.upper()}] {s.ticker} — {s.reason}")
        return results

    # ------------------------------------------------------------------
    # Call B — Strategy per Stock
    # ------------------------------------------------------------------

    async def analyse_stock(
        self,
        ticker: str,
        name: str,
        ohlcv: list[dict[str, Any]],
        quote: dict[str, Any],
        order_book: dict[str, Any],
        news: list[dict[str, Any]],
        fundamentals: Optional[dict[str, Any]] = None,
        stockgeist: Optional[dict[str, Any]] = None,
    ) -> Optional[TradingSignal]:
        """
        Generate a single trading signal for a stock using all available data.

        Args:
            fundamentals: Optional dict from FundamentalAnalyzer.fetch() — injected
                          into the prompt so the LLM can factor in valuation, quality
                          score, dividends, 52-week position, etc.

        Returns:
            Validated TradingSignal, or None on error.
        """
        # Slim news — include sentiment score where available
        slimmed_news = [
            {
                "source":       n.get("source", ""),
                "headline":     n.get("headline", ""),
                "summary":      (n.get("summary") or "")[:200],
                "published_at": n.get("published_at", ""),
                **({"sentiment": round(float(n["sentiment"]), 3)}
                   if n.get("sentiment") is not None else {}),
            }
            for n in news[:20]
        ]

        tech_text  = self._format_technicals(ohlcv, quote)
        sent_text  = self._format_sentiment_summary(news, stockgeist)
        fund_text  = self._format_fundamentals(fundamentals)

        logger.info(
            f"[LLM] Call B — Strategy | ticker={ticker} ({name}) | model={self.model} "
            f"| ohlcv_bars={len(ohlcv)} | news_items={len(slimmed_news)} "
            f"| scored_news={sum(1 for n in news if n.get('sentiment') is not None)} "
            f"| stockgeist={'yes' if stockgeist else 'no'} "
            f"| fundamentals={'yes' if fundamentals and 'error' not in fundamentals else 'no'}"
        )

        prompt = STRATEGY_PROMPT_TEMPLATE.format(
            ticker=ticker,
            name=name,
            ohlcv_json=json.dumps(ohlcv, indent=2),
            technicals_text=tech_text,
            quote_json=json.dumps(quote, indent=2),
            news_json=json.dumps(slimmed_news, indent=2),
            sentiment_text=sent_text,
            fundamentals_text=fund_text,
        )

        raw = await self._call_llm(prompt, label=f"Strategy [{ticker}]", think=True, use_tools=True)
        if not raw:
            return None

        parsed = self._parse_json(raw, expected="dict")
        if not isinstance(parsed, dict):
            logger.warning(f"[LLM] [{ticker}] Output was not a JSON dict: {raw[:300]}")
            return None

        # Normalise field aliases before Pydantic sees them
        parsed = self._normalise_signal_dict(parsed, ticker)

        try:
            signal = TradingSignal(**parsed)
            logger.info(
                f"[LLM] [{ticker}] Signal parsed OK: action={signal.action} "
                f"| entry={signal.entry_price} | target={signal.target_price} "
                f"| stop={signal.stop_loss} | confidence={signal.confidence:.0%} "
                f"| strategy={signal.strategy}"
            )
            logger.info(f"[LLM] [{ticker}] Reasoning: {signal.reasoning}")
            return signal
        except Exception as exc:
            logger.warning(f"[LLM] [{ticker}] TradingSignal validation error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Core LLM invocation with model fallback
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        prompt: str,
        label: str = "",
        think: bool = True,
        use_tools: bool = True,
    ) -> str:
        """
        Agentic LLM call with optional chain-of-thought and web-search tool support.

        Args:
            prompt:    User-facing prompt to send to the model.
            label:     Short label used in log messages (e.g. "Stock Selection").
            think:     True  → enable chain-of-thought (qwen3 /think mode).
                       False → disable thinking; forces a faster direct answer.
                       Use False for structured-output calls (Call A / stock selection)
                       where the model tends to overthink and produce empty content.
            use_tools: True  → attach web_search / news_search tools to the call.
                       False → no tools (reduces prompt complexity; use for Call A).

        Flow:
          1. Send prompt to Ollama (with or without tools / thinking).
          2. If the model calls a tool  → execute it, inject the result, repeat.
          3. Once the model stops calling tools → return the final text content.

        The loop is capped at MAX_TOOL_ROUNDS to prevent runaway chains.
        Falls back to next model in the list on any hard failure.
        """
        import asyncio
        import time

        MAX_TOOL_ROUNDS = 5

        if not OLLAMA_AVAILABLE:
            logger.error("[LLM] Ollama not available — install with: pip install ollama")
            return ""

        models_to_try = [self.model] + self.fallback_models
        if self._active_model and self._active_model != self.model:
            models_to_try = [self._active_model] + [
                m for m in models_to_try if m != self._active_model
            ]

        tag = f"[LLM] [{label}]" if label else "[LLM]"
        sep = "─" * 60

        # Resolve tools for this call
        active_tools: list = (ALL_TOOLS if DDG_AVAILABLE else []) if use_tools else []
        think_label  = "ENABLED (chain-of-thought)" if think else "DISABLED (fast direct answer)"
        tool_status  = (
            f"web search ENABLED ({len(active_tools)} tools)" if active_tools
            else ("web search DISABLED (use_tools=False)" if not use_tools
                  else "web search DISABLED (install duckduckgo-search)")
        )

        for model in models_to_try:
            logger.info(f"{tag} {sep}")
            logger.info(f"{tag} Model     : {model}")
            logger.info(f"{tag} Thinking  : {think_label}")
            logger.info(f"{tag} Tools     : {tool_status}")
            logger.info(f"{tag} Prompt    : {len(prompt)} chars | temp=0.1")
            logger.info(f"{tag} SYSTEM PROMPT ↓")
            logger.info(f"{SYSTEM_PROMPT}")
            logger.info(f"{tag} USER PROMPT ↓")
            logger.info(f"{prompt}")
            logger.info(f"{tag} {sep}")

            # Build the initial message history
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]

            t0 = time.monotonic()
            round_num = 0

            # Defined once per model attempt; restarted each tool round.
            async def _heartbeat() -> None:
                tick = 0
                while True:
                    await asyncio.sleep(5)
                    tick += 5
                    logger.info(f"{tag} ... still waiting ({tick}s elapsed) ...")

            try:
                while round_num <= MAX_TOOL_ROUNDS:
                    round_num += 1
                    logger.info(f"{tag} Sending to Ollama (round {round_num}/{MAX_TOOL_ROUNDS}) ...")

                    heartbeat = asyncio.create_task(_heartbeat())
                    try:
                        response = await asyncio.get_running_loop().run_in_executor(
                            None,
                            lambda m=model, msgs=messages, tls=active_tools, tk=think: _ollama.chat(
                                model=m,
                                messages=msgs,
                                tools=tls,
                                options={"temperature": 0.1},
                                think=tk,
                            ),
                        )
                    finally:
                        heartbeat.cancel()

                    elapsed = time.monotonic() - t0

                    # ── Unpack the SDK response object ──────────────────
                    # ollama SDK returns a ChatResponse with a Message object,
                    # NOT a plain dict — use attribute access throughout.
                    msg = response.message  # Message object

                    # thinking is a separate attribute (qwen3 chain-of-thought)
                    thinking: str = getattr(msg, "thinking", "") or ""
                    content:  str = getattr(msg, "content",  "") or ""
                    # Fallback: strip any leaked <think> tags from content
                    content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()

                    # tool_calls is a list of ToolCall objects (or None)
                    raw_tool_calls = getattr(msg, "tool_calls", None) or []

                    if thinking:
                        logger.info(f"{tag} 🧠 THINKING ({len(thinking)} chars) ↓")
                        logger.info(f"{thinking}")
                        logger.info(f"{tag} 🧠 END THINKING")

                    # ── Tool call branch ────────────────────────────────
                    if raw_tool_calls:
                        logger.info(f"{tag} {sep}")
                        logger.info(f"{tag} Model requested {len(raw_tool_calls)} tool call(s) at {elapsed:.1f}s:")
                        messages.append({"role": "assistant", "content": content})

                        for tc in raw_tool_calls:
                            # ToolCall object: tc.function.name / tc.function.arguments
                            fn_obj = getattr(tc, "function", tc)
                            name   = getattr(fn_obj, "name", "") or (tc.get("name", "") if isinstance(tc, dict) else "")
                            args   = getattr(fn_obj, "arguments", {}) or (tc.get("arguments", {}) if isinstance(tc, dict) else {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {}

                            logger.info(f"{tag}   → Tool: {name}({args})")
                            result = await execute_tool(name, args)
                            logger.info(f"{tag}   ← Result ({len(result)} chars):")
                            logger.info(f"{result[:600]}{'...' if len(result) > 600 else ''}")

                            messages.append({"role": "tool", "name": name, "content": result})

                        logger.info(f"{tag} Continuing agentic loop with tool results ...")
                        continue

                    # ── Final answer branch ─────────────────────────────
                    self._active_model = model
                    logger.info(f"{tag} {sep}")
                    if content:
                        logger.info(f"{tag} OLLAMA FINAL RESPONSE ↓  (elapsed={elapsed:.1f}s | {len(content)} chars | {round_num} round(s))")
                        logger.info(f"{content}")
                    else:
                        logger.warning(f"{tag} OLLAMA returned empty content after {elapsed:.1f}s — thinking={len(thinking)} chars")
                    logger.info(f"{tag} {sep}")
                    return content

                logger.warning(f"{tag} Reached max tool rounds ({MAX_TOOL_ROUNDS}) — returning last content")
                return content

            except Exception as exc:
                elapsed = time.monotonic() - t0
                logger.warning(f"{tag} Model '{model}' failed after {elapsed:.1f}s: {exc}")
                if model != models_to_try[-1]:
                    logger.info(f"{tag} Trying next fallback model...")

        logger.error(f"{tag} All models exhausted — no LLM response")
        return ""

    # ------------------------------------------------------------------
    # Signal normalisation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_signal_dict(d: dict[str, Any], expected_ticker: str) -> dict[str, Any]:
        """
        Remap common LLM field-name variants to the canonical TradingSignal keys
        before Pydantic validation.

        LLMs frequently use:
          stock / stock_code / symbol  → ticker
          company / company_name       → (drop; not in TradingSignal)
          recommendation / signal / trade_action / direction → action
          entry / entry_point / buy_price / price → entry_price
          target / take_profit / tp    → target_price
          sl / stop / stoploss         → stop_loss
          rationale / explanation / analysis → reasoning
          type / trade_type / setup    → strategy
          sources / references         → news_sources
        """
        # Work on a shallow copy so we don't mutate the original
        out = dict(d)

        def _move(src_keys: list[str], dst: str) -> None:
            """Copy first matching src key into dst if dst not already set."""
            if dst not in out or out[dst] is None:
                for k in src_keys:
                    if k in out and out[k] is not None:
                        out[dst] = out[k]
                        break

        _move(["stock", "stock_code", "symbol", "code"], "ticker")
        _move(
            ["recommendation", "signal", "trade_action", "direction",
             "action_type", "trade_signal", "trade"],
            "action",
        )
        _move(["entry", "entry_point", "buy_price", "price", "current_price"], "entry_price")
        _move(["target", "take_profit", "tp", "profit_target", "price_target"], "target_price")
        _move(["sl", "stop", "stoploss", "stop_price", "stop_level"], "stop_loss")
        _move(["rationale", "explanation", "analysis", "justification", "notes"], "reasoning")
        _move(["type", "trade_type", "setup", "signal_type", "pattern"], "strategy")
        _move(["sources", "references", "news_references"], "news_sources")

        # Always override ticker with the one we fetched data for
        out["ticker"] = expected_ticker

        return out

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str, expected: str = "any") -> Any:
        """
        Parse LLM output as JSON.

        Args:
            raw:      Raw string from the LLM (may contain prose, fences, etc.)
            expected: "dict"  — expect a JSON object  { ... }  (Call B strategy)
                      "list"  — expect a JSON array   [ ... ]  (Call A selection)
                      "any"   — try dict first, then list

        Handles:
        - Plain JSON
        - Markdown code fences  ```json ... ```
        - Trailing commas (common LLM mistake)
        - Arrays embedded inside objects (tries the correct outer bracket first)

        Root cause of the previous bug:
          The old code always tried "[" before "{". A strategy dict like
            { "ticker":"X", "news_sources": ["SGX"] }
          has its first "[" inside the object. The extractor grabbed
          ["SGX"] as the candidate, parsed it as a list, and then
          isinstance(parsed, dict) failed — even though the full JSON was valid.
        """
        text = raw.strip()

        # Strip markdown fences
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        # Determine bracket search order based on what we expect
        if expected == "dict":
            pairs = [("{", "}"), ("[", "]")]
        elif expected == "list":
            pairs = [("[", "]"), ("{", "}")]
        else:
            # "any" — try whichever outer bracket appears first in the text
            first_brace  = text.find("{")
            first_bracket = text.find("[")
            if first_brace == -1:
                pairs = [("[", "]"), ("{", "}")]
            elif first_bracket == -1:
                pairs = [("{", "}"), ("[", "]")]
            elif first_brace < first_bracket:
                pairs = [("{", "}"), ("[", "]")]
            else:
                pairs = [("[", "]"), ("{", "}")]

        for start_char, end_char in pairs:
            start = text.find(start_char)
            end   = text.rfind(end_char)
            if start != -1 and end > start:
                candidate = text[start : end + 1]
                # Remove trailing commas before closing brackets/braces
                candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    logger.debug(f"JSON parse attempt ({start_char}…{end_char}) failed: {exc}")

        logger.warning(f"Could not parse JSON from LLM output: {raw[:300]}")
        return None

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_technicals(
        ohlcv: list[dict[str, Any]],
        quote: dict[str, Any],
    ) -> str:
        """
        Pre-compute technical indicators from raw OHLCV + quote data and
        format them as a compact, labelled text block for the LLM prompt.

        Computed indicators:
          SMA5, SMA20  — simple moving averages of closing prices
          Volume ratio — today's volume vs 5-day average
          5-day return — price change over the available window
          5-day range position — where current price sits in the 5d high-low range
          Intraday range position — (current − day_low) / (day_high − day_low)
        """
        if not ohlcv:
            quote_src = quote.get("source", "?")
            return f"  No OHLCV data available — relying on intraday quote only (source: {quote_src})."

        closes  = [float(b["close"])  for b in ohlcv if b.get("close")]
        volumes = [float(b["volume"]) for b in ohlcv if b.get("volume")]
        highs   = [float(b["high"])   for b in ohlcv if b.get("high")]
        lows    = [float(b["low"])    for b in ohlcv if b.get("low")]

        current_price = (
            float(quote.get("last_price") or 0)
            or (closes[-1] if closes else 0)
        )

        def _sma(prices: list[float], n: int) -> Optional[float]:
            if len(prices) < n:
                return None
            return round(sum(prices[-n:]) / n, 4)

        def _vs_sma(price: float, sma: Optional[float]) -> str:
            if sma is None or price == 0:
                return "N/A"
            diff_pct = (price - sma) / sma * 100
            direction = "ABOVE" if diff_pct >= 0 else "BELOW"
            return f"{sma:.3f} — price is {direction} by {abs(diff_pct):.1f}%"

        sma5  = _sma(closes, 5)
        sma20 = _sma(closes, 20)

        # Volume ratio: today's vol vs 5d average
        vol_ratio_str = "N/A"
        if len(volumes) >= 2:
            prev_vols = volumes[:-1][-4:]   # up to 4 prior days (excludes today)
            avg_vol5  = sum(prev_vols) / len(prev_vols) if prev_vols else None
            today_vol = volumes[-1]
            if avg_vol5 and avg_vol5 > 0:
                ratio = today_vol / avg_vol5
                label = (
                    "HIGH — strong conviction"  if ratio >= 2.0 else
                    "above average"             if ratio >= 1.3 else
                    "normal"                    if ratio >= 0.7 else
                    "LOW — weak conviction"
                )
                vol_ratio_str = f"{ratio:.1f}× 5d avg ({label})"

        # 5-day return
        ret_str = "N/A"
        if len(closes) >= 2 and closes[0] > 0:
            ret_pct = (closes[-1] - closes[0]) / closes[0] * 100
            ret_str = f"{ret_pct:+.1f}%"

        # 5-day range position
        range5_str = "N/A"
        if highs and lows and current_price:
            hi5, lo5 = max(highs), min(lows)
            if hi5 > lo5:
                pos = (current_price - lo5) / (hi5 - lo5) * 100
                label = (
                    "near 5d HIGH — momentum zone"    if pos >= 80 else
                    "near 5d LOW — potential support"  if pos <= 20 else
                    f"{pos:.0f}% of 5d range"
                )
                range5_str = f"S${lo5:.3f} – S${hi5:.3f} | position: {label}"

        # Intraday range position
        intraday_str = "N/A"
        day_high = float(quote.get("high") or 0)
        day_low  = float(quote.get("low")  or 0)
        if day_high > day_low > 0 and current_price:
            ipos = (current_price - day_low) / (day_high - day_low) * 100
            intraday_str = (
                f"S${day_low:.3f} – S${day_high:.3f} | "
                f"current at {ipos:.0f}% of today's range"
            )

        quote_src  = quote.get("source", "?")
        quote_live = "LIVE" if "finnhub" in quote_src.lower() else "DELAYED/CACHED"

        lines = [
            f"  Source        : {quote_src} ({quote_live})",
            f"  SMA 5-day     : {_vs_sma(current_price, sma5)}",
            f"  SMA 20-day    : {_vs_sma(current_price, sma20)}"
            + ("  ← only N/A if <20 bars available" if sma20 is None else ""),
            f"  Volume Ratio  : {vol_ratio_str}",
            f"  5d Return     : {ret_str}",
            f"  5d Price Range: {range5_str}",
            f"  Intraday Range: {intraday_str}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_sentiment_summary(
        news: list[dict[str, Any]],
        stockgeist: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Aggregate sentiment scores from all news articles (Marketaux, EODHD
        provide per-article sentiment floats −1.0 → +1.0) and format alongside
        the StockGeist real-time score and 24-hour trend.
        """
        lines: list[str] = []

        # ── StockGeist real-time + 24h trend ───────────────────────────────
        sg = stockgeist or {}
        now = sg.get("now", {})
        if now:
            score = now.get("score", 0)
            label = (
                "BULLISH"  if score >= 0.3  else
                "BEARISH"  if score <= -0.3 else
                "NEUTRAL"
            )
            trend = sg.get("trend_24h")
            trend_str = ""
            if trend is not None:
                direction = "↑ rising" if trend > 0.05 else "↓ falling" if trend < -0.05 else "→ flat"
                trend_str = f" | 24h trend: {direction} (Δ{trend:+.3f})"
            avg = sg.get("series_avg")
            avg_str = f" | 24h avg score: {avg:.3f}" if avg is not None else ""
            lines.append(
                f"  StockGeist Now : score={score:.3f} ({label})"
                f" | pos={now.get('pos_count',0)} neg={now.get('neg_count',0)} mentions"
                f"{trend_str}{avg_str}"
            )
        else:
            lines.append("  StockGeist     : Not enabled / no data")

        # ── Aggregate news sentiment (Marketaux + EODHD scored articles) ──
        scored = [(n.get("source", "?"), float(n["sentiment"]))
                  for n in news if n.get("sentiment") is not None]

        if scored:
            scores  = [s for _, s in scored]
            avg_s   = sum(scores) / len(scores)
            pos_n   = sum(1 for s in scores if s >= 0.1)
            neg_n   = sum(1 for s in scores if s <= -0.1)
            neu_n   = len(scores) - pos_n - neg_n
            agg_lbl = (
                "BULLISH"  if avg_s >= 0.2  else
                "BEARISH"  if avg_s <= -0.2 else
                "NEUTRAL"
            )
            # Top source by absolute score
            most_src, most_score = max(scored, key=lambda x: abs(x[1]))
            direction = "bullish" if most_score > 0 else "bearish"
            lines.append(
                f"  News Sentiment : avg={avg_s:+.3f} ({agg_lbl}) from {len(scores)} scored articles"
            )
            lines.append(
                f"  Distribution   : {pos_n} positive | {neu_n} neutral | {neg_n} negative"
            )
            lines.append(
                f"  Strongest signal: {most_src} score={most_score:+.3f} ({direction})"
            )
        else:
            lines.append("  News Sentiment : No scored articles (free scrapers only, no sentiment scores)")
            lines.append("  Tip: Enable Marketaux or EODHD in config for per-article sentiment data")

        return "\n".join(lines)

    @staticmethod
    def _format_fundamentals(fund: Optional[dict[str, Any]]) -> str:
        """
        Convert a FundamentalAnalyzer result dict into a compact, readable text
        block for injection into the LLM prompt.

        Keeps the total token count low while preserving every metric that
        matters for signal validation.
        """
        if not fund or fund.get("error"):
            return "Not available — fundamentals data could not be fetched."

        def v(val: Any, suffix: str = "", na: str = "N/A") -> str:
            """Render a value with an optional suffix, or na if None."""
            if val is None:
                return na
            return f"{val}{suffix}"

        val  = fund.get("valuation",  {})
        inc  = fund.get("income",     {})
        bal  = fund.get("balance",    {})
        div  = fund.get("dividends",  {})
        tech = fund.get("technicals", {})
        qs   = fund.get("quality_score")

        # 52-week position label
        pos = tech.get("wk52_position_pct")
        if pos is not None:
            if pos >= 85:
                pos_label = f"{pos}% (near 52W HIGH — momentum zone)"
            elif pos <= 15:
                pos_label = f"{pos}% (near 52W LOW — potential reversal zone)"
            else:
                pos_label = f"{pos}% of 52W range"
        else:
            pos_label = "N/A"

        # Quality score label
        if qs is not None:
            if qs >= 60:
                qs_label = f"{qs}/100 (STRONG — supports higher confidence)"
            elif qs >= 35:
                qs_label = f"{qs}/100 (MODERATE)"
            else:
                qs_label = f"{qs}/100 (WEAK — reduce BUY confidence)"
        else:
            qs_label = "N/A"

        # Dividend flag
        ex_div = div.get("ex_dividend_date")
        div_flag = f"  Ex-Div Date   : {ex_div} ← potential dividend catalyst\n" if ex_div else ""

        lines = [
            f"  Quality Score : {qs_label}",
            f"  Sector        : {fund.get('sector', 'N/A')}",
            "",
            "  [ Valuation ]",
            f"  Market Cap    : {v(val.get('market_cap_fmt'))}    "
            f"P/E (ttm)  : {v(val.get('trailing_pe'))}    "
            f"P/E (fwd)  : {v(val.get('forward_pe'))}    "
            f"P/B        : {v(val.get('price_to_book'))}",
            f"  EV/EBITDA     : {v(val.get('ev_to_ebitda'))}",
            "",
            "  [ Income — Trailing 12 Months ]",
            f"  Revenue       : {v(inc.get('revenue_fmt'))}    "
            f"Net Income : {v(inc.get('net_income_fmt'))}    "
            f"EPS (ttm)  : {v(inc.get('eps_trailing'))}",
            f"  Net Margin    : {v(inc.get('net_margin_pct'))}    "
            f"Op Margin  : {v(inc.get('operating_margin_pct'))}    "
            f"Rev Growth : {v(inc.get('revenue_growth_pct'))}    "
            f"Earn Growth: {v(inc.get('earnings_growth_pct'))}",
            "",
            "  [ Balance Sheet ]",
            f"  Debt/Equity   : {v(bal.get('debt_to_equity'))}    "
            f"Current Ratio: {v(bal.get('current_ratio'))}    "
            f"Book Val/Sh: {v(bal.get('book_value_per_share'))}",
            f"  ROE           : {v(bal.get('roe_pct'))}    "
            f"ROA        : {v(bal.get('roa_pct'))}    "
            f"Free CF    : {v(bal.get('free_cashflow_fmt'))}",
            "",
            "  [ Dividends ]",
            f"  Div Yield     : {v(div.get('yield_pct'))}    "
            f"Annual DPS : {v(div.get('annual_dps'))}    "
            f"Payout     : {v(div.get('payout_ratio_pct'))}",
            div_flag.rstrip() if div_flag else "  Ex-Div Date   : N/A",
            "",
            "  [ Technicals ]",
            f"  52W Range     : {v(tech.get('wk52_low'))} – {v(tech.get('wk52_high'))}    "
            f"Position   : {pos_label}",
            f"  Beta          : {v(tech.get('beta'))}    "
            f"Avg Vol(10d): {v(tech.get('avg_volume_10d'), na='N/A')}",
        ]

        return "\n".join(lines)

    @staticmethod
    def _build_stock_csv(stock_list: list[dict[str, Any]], max_rows: int = 500) -> str:
        """Build a compact CSV representation of the stock list for the prompt."""
        lines = ["ticker,name,sector,market_cap_M,last_price,change_pct,volume,price_date,stale"]
        for s in stock_list[:max_rows]:
            mc = s.get("market_cap")
            mc_str = f"{mc / 1_000_000:.0f}" if mc else "N/A"
            chg = s.get("change_pct")
            chg_str = f"{chg:+.2f}%" if chg is not None else "N/A"
            stale = "Y" if s.get("price_stale") else "N"
            lines.append(
                ",".join(
                    [
                        str(s.get("ticker", "")),
                        str(s.get("name", "")).replace(",", " "),
                        str(s.get("sector", "")).replace(",", " "),
                        mc_str,
                        str(s.get("last_price") or "N/A"),
                        chg_str,
                        str(s.get("volume") or "N/A"),
                        str(s.get("price_date") or "unknown"),
                        stale,
                    ]
                )
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    async def _main() -> None:
        analyst = LLMAnalyst()
        sample_stocks = [
            {"ticker": "D05", "name": "DBS Group", "sector": "Finance", "market_cap": 80e9, "last_price": 38.5, "volume": 5000000},
            {"ticker": "O39", "name": "OCBC Bank", "sector": "Finance", "market_cap": 55e9, "last_price": 14.2, "volume": 3000000},
        ]
        selected = await analyst.select_stocks(sample_stocks, "US markets up overnight, risk-on sentiment")
        print(f"Selected: {selected}")

    asyncio.run(_main())
