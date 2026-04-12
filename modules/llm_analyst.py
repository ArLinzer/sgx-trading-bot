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


STRATEGY_PROMPT_TEMPLATE = """You are a day trading analyst. Analyse the following data for {ticker} ({name}) and generate a trading signal.

PRICE & VOLUME DATA (last 5 days OHLCV, newest last):
{ohlcv_json}

INTRADAY QUOTE:
{quote_json}

ORDER BOOK SNAPSHOT (top 5 levels):
{orderbook_json}

NEWS & ANNOUNCEMENTS (last 24 hours, newest first):
{news_json}

Trading constraints:
- Day trading only — all positions MUST be closed before market close (17:00 SGT)
- SGX market hours: 09:00–17:00 SGT (lunch break 12:00–14:00)
- Singapore dollars (SGD)

Task: Generate ONE trading signal.
Return ONLY a JSON object with exactly these keys:
  "ticker"         : string  — SGX stock code
  "action"         : string  — one of "BUY", "SELL", "HOLD", "WATCH"
  "entry_price"    : number  — suggested entry price (null if HOLD/WATCH)
  "target_price"   : number  — profit target price (null if HOLD/WATCH)
  "stop_loss"      : number  — stop loss price (null if HOLD/WATCH)
  "confidence"     : number  — confidence score between 0.0 and 1.0
  "strategy"       : string  — one of "momentum", "news_catalyst", "breakout", "mean_reversion", "range_trade"
  "reasoning"      : string  — 2-3 sentence explanation of the signal
  "time_horizon"   : string  — always "intraday"
  "exit_before_eod": boolean — always true
  "news_sources"   : array   — list of source names that influenced the signal

Example:
{{
  "ticker": "D05",
  "action": "BUY",
  "entry_price": 38.50,
  "target_price": 39.10,
  "stop_loss": 38.20,
  "confidence": 0.78,
  "strategy": "news_catalyst",
  "reasoning": "DBS reported Q3 earnings beat (+12% NII). Price broke above 20-day SMA on 2x avg volume. Order book shows strong bid support at 38.40.",
  "time_horizon": "intraday",
  "exit_before_eod": true,
  "news_sources": ["Business Times", "SGX Announcements"]
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
    ) -> Optional[TradingSignal]:
        """
        Generate a single trading signal for a stock using all available data.

        Returns:
            Validated TradingSignal, or None on error.
        """
        # Slim down news to avoid exceeding context window
        slimmed_news = [
            {
                "source": n.get("source", ""),
                "headline": n.get("headline", ""),
                "summary": (n.get("summary") or "")[:200],
                "published_at": n.get("published_at", ""),
            }
            for n in news[:15]
        ]

        logger.info(
            f"[LLM] Call B — Strategy | ticker={ticker} ({name}) | model={self.model} "
            f"| ohlcv_bars={len(ohlcv)} | news_items={len(slimmed_news)} "
            f"| order_book={'yes' if order_book else 'no'} | quote={'yes' if quote else 'no'}"
        )

        prompt = STRATEGY_PROMPT_TEMPLATE.format(
            ticker=ticker,
            name=name,
            ohlcv_json=json.dumps(ohlcv, indent=2),
            quote_json=json.dumps(quote, indent=2),
            orderbook_json=json.dumps(
                {"bids": (order_book.get("bids") or [])[:5],
                 "asks": (order_book.get("asks") or [])[:5]},
                indent=2,
            ),
            news_json=json.dumps(slimmed_news, indent=2),
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

            try:
                while round_num <= MAX_TOOL_ROUNDS:
                    round_num += 1
                    logger.info(f"{tag} Sending to Ollama (round {round_num}/{MAX_TOOL_ROUNDS}) ...")

                    async def _heartbeat(t=tag) -> None:
                        tick = 0
                        while True:
                            await asyncio.sleep(5)
                            tick += 5
                            logger.info(f"{t} ... still waiting ({tick}s elapsed) ...")

                    heartbeat = asyncio.create_task(_heartbeat())
                    try:
                        response = await asyncio.get_event_loop().run_in_executor(
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
