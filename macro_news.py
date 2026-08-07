#!/usr/bin/env python3
"""
macro_news.py
=============

Geopolitical and macro risk layer for the intraday momentum screener.

--------------------------------------------------------------------------------
DESIGN RATIONALE — READ THIS, IT EXPLAINS WHY THE CODE LOOKS LIKE IT DOES
--------------------------------------------------------------------------------

The obvious way to build "geopolitical news awareness" is: scrape headlines, run
sentiment on them, add the sentiment to your trading score. That approach is
close to worthless, for four reasons:

  1. LATENCY. By the time a headline reaches a public RSS feed, institutional
     order flow has already responded. Machine-readable news alpha decays in
     seconds; RSS is minutes-to-hours behind. You are trading a stale signal.

  2. THE PRICED-IN PROBLEM. This is the fundamental one. The tradeable question
     is never "is this news bad" — it is "is this news WORSE THAN WHAT WAS
     ALREADY EXPECTED." A headline scraper has no access to expectations. Markets
     routinely rally on catastrophic headlines that came in less bad than feared.

  3. NEGATION AND CONTEXT. Keyword sentiment cannot reliably distinguish
     "sanctions imposed" from "sanctions lifted", "ceasefire collapses" from
     "ceasefire holds", or a retrospective from a breaking event. Sophisticated
     NLP helps but does not solve (2).

  4. SELECTION NOISE. Yahoo's per-ticker feed is heavily polluted with syndicated
     filler. Measured live on this machine: of NVDA's six most recent "news"
     items, ZERO mentioned NVDA. Unfiltered, you are scoring noise.

So this module does something different, on two tracks:

  TRACK A — QUANTITATIVE REGIME (scored, drives position sizing)
     Instead of guessing at geopolitics from text, read what the market is
     ALREADY PRICING. Gold, crude, the dollar, bond volatility, equity
     volatility and defense-sector relative strength are continuous, real-time,
     unambiguous measures of geopolitical stress, produced by participants with
     far better information than an RSS feed. A gold+oil+VIX+defense bid IS the
     geopolitical signal. This is measurable, so this is what gets scored.

  TRACK B — HEADLINES AS A RISK VETO (flagged, NOT scored as direction)
     The presence of a fresh, ticker-relevant catalyst is objectively measurable
     even when its direction is not. And catalyst presence has a concrete,
     actionable implication: NEWS-DRIVEN MOVES GAP THROUGH STOPS. Your technical
     stop assumes continuous price. An event stock can trade straight through it.
     So headlines are used to WIDEN required reward-to-risk and CUT position
     size — never to predict direction.

Direction-of-sentiment is computed for display only and is explicitly excluded
from every score. That is a deliberate choice, not an oversight.

--------------------------------------------------------------------------------
DEPENDENCIES: yfinance, pandas, numpy, feedparser (optional but recommended)
--------------------------------------------------------------------------------
Copyright note: this module handles only headline TITLES plus source name and
link — the standard syndication use RSS exists for. It deliberately never stores
or prints article bodies or feed `summary` fields.
"""

from __future__ import annotations

import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import feedparser  # optional
    _HAS_FEEDPARSER = True
except ImportError:  # pragma: no cover
    _HAS_FEEDPARSER = False


# ==============================================================================
# SECTION 1 — MACRO INSTRUMENT UNIVERSE
# ==============================================================================

@dataclass
class MacroInstrument:
    """One market-priced gauge of macro/geopolitical stress.

    `stress_sign` = +1 means RISING price indicates RISING stress (gold, oil,
    VIX). -1 means rising price indicates FALLING stress (SPX, high-yield).
    """

    symbol: str
    label: str
    stress_sign: int
    weight: float
    note: str


# Chosen so that no single narrative dominates: equity vol, bond vol, energy
# supply risk, monetary metal, reserve currency, and defense equities. A genuine
# geopolitical event moves several of these together; an idiosyncratic equity
# selloff moves only VIX.
MACRO_INSTRUMENTS: List[MacroInstrument] = [
    MacroInstrument("^VIX", "Equity vol (VIX)", +1, 0.20,
                    "front-month S&P implied vol; the headline fear gauge"),
    MacroInstrument("^MOVE", "Bond vol (MOVE)", +1, 0.15,
                    "Treasury implied vol; often leads equity vol on macro shocks"),
    MacroInstrument("GC=F", "Gold", +1, 0.18,
                    "monetary/geopolitical hedge; bid on conflict + debasement fear"),
    MacroInstrument("CL=F", "Crude (WTI)", +1, 0.17,
                    "energy supply risk; the cleanest conflict transmission channel"),
    MacroInstrument("DX-Y.NYB", "US Dollar (DXY)", +1, 0.10,
                    "reserve-currency flight-to-safety bid"),
    MacroInstrument("ITA", "Defense sector", +1, 0.12,
                    "defense equities vs market; a direct conflict-expectation read"),
    MacroInstrument("^TNX", "US 10Y yield", 0, 0.00,
                    "context only: direction is regime-dependent, not signed"),
    MacroInstrument("TLT", "Long Treasuries", -1, 0.08,
                    "duration bid on flight to quality"),
]

BENCHMARK = "^GSPC"  # for computing defense-sector RELATIVE strength

# Categories for classifying broad headlines. Keyword matching is crude; these
# are used for LABELLING and grouping only, never for directional inference.
GEO_CATEGORIES: Dict[str, List[str]] = {
    "conflict": [
        "war", "strike", "missile", "invasion", "troops", "ceasefire", "military",
        "airstrike", "offensive", "drone attack", "shelling", "escalation",
        "casualties", "combat", "insurgen", "militant",
    ],
    "sanctions_trade": [
        "sanction", "tariff", "embargo", "export control", "trade war", "quota",
        "blacklist", "entity list", "customs", "levy", "protectionis", "wto",
    ],
    "energy": [
        "opec", "oil supply", "pipeline", "refinery", "lng", "gas supply",
        "crude output", "energy crisis", "strait of hormuz", "production cut",
    ],
    "monetary": [
        "federal reserve", "fed ", "interest rate", "rate cut", "rate hike",
        "inflation", "cpi", "central bank", "ecb", "boj", "yield curve",
        "quantitative", "fomc",
    ],
    "politics": [
        "election", "parliament", "coup", "protest", "referendum", "impeach",
        "shutdown", "summit", "treaty", "diplomat", "resign",
    ],
    "supply_chain": [
        "shipping", "port", "container", "chip shortage", "semiconductor export",
        "rare earth", "supply chain", "red sea", "canal", "logistics",
    ],
}

# Broad feeds verified reachable. Each entry: (name, url, weight).
# Reuters, AP and FT return HTTP 403 to unauthenticated clients and are omitted.
GEO_FEEDS: List[Tuple[str, str, float]] = [
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml", 1.0),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml", 0.9),
    ("CNBC World", "https://search.cnbc.com/rs/search/combinedcms/view.xml"
                   "?partnerId=wrss01&id=100727362", 1.0),
    ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml"
                     "?partnerId=wrss01&id=20910258", 1.0),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories", 0.9),
]


# ==============================================================================
# SECTION 2 — QUANTITATIVE MACRO REGIME
# ==============================================================================

@dataclass
class InstrumentReading:
    symbol: str
    label: str
    last: float = float("nan")
    change_pct: float = float("nan")     # session change
    zscore: float = float("nan")         # of daily change, vs trailing window
    percentile: float = float("nan")     # of LEVEL, vs trailing window
    stress_contribution: float = 0.0     # signed, weighted
    available: bool = False
    note: str = ""


@dataclass
class MacroRegime:
    """Market-priced geopolitical / macro stress snapshot."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    readings: Dict[str, InstrumentReading] = field(default_factory=dict)
    stress_index: float = float("nan")   # 0-100
    stress_label: str = "unknown"
    breadth: int = 0                     # how many gauges confirm stress
    size_multiplier: float = 1.0         # position-size scaler
    notes: List[str] = field(default_factory=list)


def _zscore_and_pct(series: pd.Series, window: int = 60) -> Tuple[float, float]:
    """Z-score of the latest daily CHANGE and percentile of the latest LEVEL.

    Both matter and they say different things. A one-day VIX spike is a shock
    (z-score). A VIX that has sat at 30 for a month is a regime (percentile).
    """
    s = series.dropna()
    if len(s) < 12:
        return (float("nan"), float("nan"))
    chg = s.pct_change().dropna()
    tail = chg.tail(window)
    mu, sd = float(tail.mean()), float(tail.std(ddof=1))
    z = (float(chg.iloc[-1]) - mu) / sd if sd and math.isfinite(sd) and sd > 0 else float("nan")
    lvl = s.tail(window)
    pct = float((lvl < s.iloc[-1]).mean() * 100.0) if len(lvl) else float("nan")
    return (z, pct)


def fetch_macro_regime(
    period: str = "6mo", verbose: bool = True, timeout_note: bool = True
) -> MacroRegime:
    """Download macro instruments and compute a composite stress index.

    The index is a weighted, sign-corrected blend of (a) how unusual today's move
    is in each gauge and (b) where each gauge's level sits in its recent range.
    Output is 0-100 where ~50 is unremarkable.

    IMPORTANT: this measures stress that is ALREADY PRICED. It is a risk-control
    input, not a forecast. High readings do not mean "sell"; they mean
    "uncertainty is elevated, so size down and widen your R:R requirement."
    """
    import yfinance as yf

    reg = MacroRegime()
    symbols = [m.symbol for m in MACRO_INSTRUMENTS] + [BENCHMARK]

    try:
        raw = yf.download(symbols, period=period, interval="1d", progress=False,
                          auto_adjust=False, group_by="ticker", threads=True)
    except Exception as exc:  # noqa: BLE001
        reg.notes.append(f"macro download failed: {type(exc).__name__}")
        return reg

    if raw is None or raw.empty:
        reg.notes.append("macro download returned no data")
        return reg

    def close_of(sym: str) -> Optional[pd.Series]:
        try:
            return raw[sym]["Close"].dropna()
        except Exception:  # noqa: BLE001
            return None

    bench = close_of(BENCHMARK)

    weighted_sum, weight_used, confirming = 0.0, 0.0, 0
    for m in MACRO_INSTRUMENTS:
        r = InstrumentReading(symbol=m.symbol, label=m.label, note=m.note)
        s = close_of(m.symbol)
        if s is None or len(s) < 12:
            reg.readings[m.symbol] = r
            continue

        # Defense sector is only informative RELATIVE to the market. ITA rising
        # because everything is rising says nothing about geopolitics.
        if m.symbol == "ITA" and bench is not None and len(bench) > 12:
            aligned = pd.concat([s, bench], axis=1).dropna()
            if len(aligned) > 12:
                s = aligned.iloc[:, 0] / aligned.iloc[:, 1]
                r.note = "defense sector RELATIVE to S&P 500 (ratio)"

        r.available = True
        r.last = float(s.iloc[-1])
        if len(s) >= 2:
            r.change_pct = (float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100.0
        r.zscore, r.percentile = _zscore_and_pct(s)

        if m.stress_sign == 0 or m.weight <= 0:
            reg.readings[m.symbol] = r          # context only, unscored
            continue

        # Blend shock (z) and regime (percentile), each mapped to roughly 0-1.
        z_component = _logistic(m.stress_sign * (r.zscore if math.isfinite(r.zscore) else 0.0))
        p = r.percentile if math.isfinite(r.percentile) else 50.0
        p_component = (p if m.stress_sign > 0 else 100.0 - p) / 100.0
        blended = 0.55 * z_component + 0.45 * p_component

        r.stress_contribution = blended * m.weight
        weighted_sum += r.stress_contribution
        weight_used += m.weight
        if blended > 0.65:
            confirming += 1
        reg.readings[m.symbol] = r

    if weight_used <= 0:
        reg.notes.append("no macro instruments available")
        return reg

    reg.stress_index = 100.0 * weighted_sum / weight_used
    reg.breadth = confirming
    reg.stress_label, reg.size_multiplier = _classify_stress(reg.stress_index, confirming)

    missing = [m.symbol for m in MACRO_INSTRUMENTS if not reg.readings[m.symbol].available]
    if missing:
        reg.notes.append(f"unavailable: {', '.join(missing)}")
    return reg


def _logistic(x: float, k: float = 1.1) -> float:
    """Squash a z-score to (0, 1). Bounded so one wild print cannot dominate.

    The input is clamped before exponentiating for two reasons. First,
    robustness: math.exp overflows above ~709, so an extreme z-score from a
    corrupt tick would raise OverflowError and take down the whole macro panel.
    Second, correctness of the composite: saturating to exactly 0.0 or 1.0 makes
    every extreme value indistinguishable, which throws away the difference
    between "unusual" and "absurd, probably a data error". Clamping at ±30 keeps
    the output in the open interval while flattening the tail.
    """
    if not math.isfinite(x):
        return 0.5
    z = max(-30.0, min(30.0, k * x))
    return 1.0 / (1.0 + math.exp(-z))


def _classify_stress(index: float, breadth: int) -> Tuple[str, float]:
    """Map the stress index to a label and a position-size multiplier.

    Breadth is a required confirmation. A high index driven by ONE gauge is
    usually an idiosyncratic move (an oil inventory print, a vol-market
    technical), not a geopolitical regime. Genuine geopolitical stress shows up
    in several unrelated markets at once. So the aggressive size cut requires
    both a high reading AND confirmation across gauges.
    """
    if not math.isfinite(index):
        return ("unknown", 1.0)
    if index >= 70 and breadth >= 3:
        return ("ELEVATED — broad-based", 0.50)
    if index >= 70:
        return ("elevated — narrow (few gauges confirm)", 0.75)
    if index >= 58:
        return ("moderately elevated", 0.80)
    if index >= 42:
        return ("normal", 1.00)
    if index >= 30:
        return ("subdued", 1.00)
    return ("very subdued (complacency risk)", 1.00)


# ==============================================================================
# SECTION 3 — HEADLINE INGESTION
# ==============================================================================

@dataclass
class Headline:
    title: str
    source: str
    url: str = ""
    published: Optional[datetime] = None
    age_hours: float = float("nan")
    categories: List[str] = field(default_factory=list)
    ticker_relevant: bool = False


def _parse_time(value) -> Optional[datetime]:
    """Best-effort timestamp parsing across the several shapes feeds return."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, ValueError, OverflowError):
            return None
    if isinstance(value, str):
        v = value.strip().replace("Z", "+00:00")
        for parser in (
            lambda x: datetime.fromisoformat(x),
            lambda x: datetime.strptime(x, "%a, %d %b %Y %H:%M:%S %z"),
            lambda x: datetime.strptime(x, "%a, %d %b %Y %H:%M:%S %Z"),
        ):
            try:
                dt = parser(v)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
    if isinstance(value, time.struct_time):
        try:
            return datetime.fromtimestamp(time.mktime(value), tz=timezone.utc)
        except (OSError, ValueError, OverflowError):
            return None
    return None


def _compile_category_patterns() -> Dict[str, re.Pattern]:
    """Pre-compile word-boundary regexes for each category.

    WHY NOT `keyword in title.lower()`: naive substring matching is badly broken
    for short keywords. Measured live before this fix, the `supply_chain`
    category matched "Cassidy supports Todd Blanche" and "the July jobs report"
    — because "port" is a substring of "supports" and "report". Nearly every
    headline containing the word "report" was being tagged a supply-chain event.

    `\b` boundaries fix it. Multi-word phrases keep internal whitespace flexible
    so "supply chain" also matches "supply-chain" and "supply  chain".
    """
    compiled: Dict[str, re.Pattern] = {}
    for cat, kws in GEO_CATEGORIES.items():
        parts = []
        for kw in kws:
            kw = kw.strip()
            if not kw:
                continue
            # Allow trailing-stem matches for keywords deliberately left
            # truncated in the list (e.g. "sanction" -> sanctions/sanctioned).
            escaped = r"[\s\-]+".join(re.escape(w) for w in kw.split())
            parts.append(rf"\b{escaped}\w*")
        if parts:
            compiled[cat] = re.compile("|".join(parts), re.IGNORECASE)
    return compiled


_CATEGORY_PATTERNS: Dict[str, re.Pattern] = _compile_category_patterns()


def classify_headline(title: str) -> List[str]:
    """Tag a headline with geopolitical categories via word-boundary matching.

    Still crude by construction: no negation handling ("sanctions LIFTED" tags
    the same as "sanctions IMPOSED"), no entity resolution, no disambiguation of
    metaphorical usage ("price war", "battleground state"). Used for GROUPING
    and DISPLAY only — never for directional inference.
    """
    if not title:
        return []
    return [cat for cat, pat in _CATEGORY_PATTERNS.items() if pat.search(title)]


def fetch_ticker_headlines(
    ticker: str, company_name: str = "", max_age_hours: float = 48.0,
    limit: int = 8
) -> List[Headline]:
    """Per-ticker headlines, filtered for actual relevance to that ticker.

    THE RELEVANCE FILTER IS NOT OPTIONAL. Yahoo's per-ticker feed is padded with
    syndicated market-wide filler. Measured live: zero of NVDA's six most recent
    items mentioned NVDA. Without filtering, every ticker appears to have news
    and the catalyst flag becomes meaningless.

    We check the symbol and, when known, the company name — including a stripped
    form, so "DraftKings Inc." also matches "DraftKings".
    """
    import yfinance as yf

    out: List[Headline] = []
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:  # noqa: BLE001
        return out

    needles = {ticker.lower()}
    if company_name:
        base = re.sub(
            r"\b(inc|corp|corporation|co|ltd|plc|holdings|group|the|company|sa|nv)\b\.?",
            "", company_name.lower(),
        ).strip(" ,.-")
        if len(base) >= 3:
            needles.add(base)
            first = base.split()[0]
            if len(first) >= 4:
                needles.add(first)

    now = datetime.now(timezone.utc)
    for it in items:
        c = it.get("content", it) if isinstance(it, dict) else {}
        title = (c.get("title") or "").strip()
        if not title:
            continue
        prov = c.get("provider") or {}
        source = prov.get("displayName") if isinstance(prov, dict) else str(prov)
        published = _parse_time(c.get("pubDate") or c.get("displayTime")
                                or c.get("providerPublishTime"))
        age = ((now - published).total_seconds() / 3600.0
               if published else float("nan"))
        if math.isfinite(age) and age > max_age_hours:
            continue

        # Deliberately NOT reading c["summary"] — titles + link only.
        url = ""
        for key in ("canonicalUrl", "clickThroughUrl", "previewUrl"):
            v = c.get(key)
            if isinstance(v, dict) and v.get("url"):
                url = v["url"]
                break
            if isinstance(v, str) and v:
                url = v
                break

        low = title.lower()
        h = Headline(
            title=title,
            source=source or "unknown",
            url=url,
            published=published,
            age_hours=age,
            categories=classify_headline(title),
            ticker_relevant=any(n in low for n in needles),
        )
        out.append(h)

    out.sort(key=lambda h: (h.age_hours if math.isfinite(h.age_hours) else 1e9))
    return out[:limit]


def fetch_geopolitical_headlines(
    max_age_hours: float = 24.0, limit_per_feed: int = 25, verbose: bool = False
) -> List[Headline]:
    """Broad macro/geopolitical headlines from public feeds.

    Degrades gracefully: a dead or blocked feed is skipped, not fatal. Returns
    only headlines that matched at least one geopolitical category, so ordinary
    market chatter is excluded.
    """
    if not _HAS_FEEDPARSER:
        return []

    now = datetime.now(timezone.utc)
    seen: set = set()
    out: List[Headline] = []

    for name, url, _w in GEO_FEEDS:
        try:
            d = feedparser.parse(url)
        except Exception:  # noqa: BLE001
            continue
        for e in getattr(d, "entries", [])[:limit_per_feed]:
            title = (getattr(e, "title", "") or "").strip()
            if not title:
                continue
            key = title.lower()[:90]
            if key in seen:
                continue
            published = _parse_time(
                getattr(e, "published_parsed", None)
                or getattr(e, "updated_parsed", None)
                or getattr(e, "published", None)
            )
            age = ((now - published).total_seconds() / 3600.0
                   if published else float("nan"))
            if math.isfinite(age) and age > max_age_hours:
                continue
            cats = classify_headline(title)
            if not cats:
                continue  # not geopolitical; skip
            seen.add(key)
            out.append(Headline(
                title=title, source=name, url=getattr(e, "link", "") or "",
                published=published, age_hours=age, categories=cats,
            ))

    out.sort(key=lambda h: (h.age_hours if math.isfinite(h.age_hours) else 1e9))
    return out


# ==============================================================================
# SECTION 4 — CATALYST RISK (headlines -> risk adjustment, never direction)
# ==============================================================================

@dataclass
class CatalystFlag:
    """Per-ticker news-risk assessment.

    NOTE WHAT IS ABSENT: there is no `direction` field and no `sentiment_score`
    that feeds anything. Presence and freshness of a catalyst are measurable;
    its tradeable direction is not. Encoding only what is measurable is the
    entire point.
    """

    ticker: str
    has_fresh_news: bool = False
    n_relevant: int = 0
    newest_age_hours: float = float("nan")
    categories: List[str] = field(default_factory=list)
    headlines: List[Headline] = field(default_factory=list)
    risk_level: str = "none"          # none | moderate | high
    size_multiplier: float = 1.0
    rr_requirement_multiplier: float = 1.0
    warnings: List[str] = field(default_factory=list)


def assess_catalyst_risk(
    ticker: str,
    headlines: Sequence[Headline],
    fresh_hours: float = 12.0,
    rvol: Optional[float] = None,
) -> CatalystFlag:
    """Convert headlines into concrete risk adjustments.

    The logic rests on one mechanical fact: a technical stop assumes the price
    path is continuous. News-driven names violate that — they gap. A stop at
    -1% on an event stock can fill at -6%. Therefore fresh relevant news implies
    smaller size and a higher required reward-to-risk, regardless of whether the
    news reads bullish or bearish.

    RVOL CROSS-CHECK (`rvol`): headline presence alone over-fires on mega-caps,
    which generate a constant stream of reaction and commentary pieces. Measured
    live, NVDA flagged "high risk" off a single tangential article while trading
    at 0.8x relative volume — i.e. the market was ignoring it. The discriminator
    is whether the news is actually MOVING the stock. Volume is that evidence:

        fresh news + elevated RVOL  -> real catalyst, gap risk is live
        fresh news + normal RVOL    -> commentary; the tape has not reacted

    This is why the flag takes RVOL. Passing it is optional but strongly
    recommended; without it the function falls back to headline-only logic and
    will over-flag.
    """
    flag = CatalystFlag(ticker=ticker)
    relevant = [h for h in headlines if h.ticker_relevant]
    flag.n_relevant = len(relevant)
    flag.headlines = relevant[:5]

    if not relevant:
        return flag

    ages = [h.age_hours for h in relevant if math.isfinite(h.age_hours)]
    flag.newest_age_hours = min(ages) if ages else float("nan")
    cats: List[str] = []
    for h in relevant:
        for c in h.categories:
            if c not in cats:
                cats.append(c)
    flag.categories = cats

    fresh = math.isfinite(flag.newest_age_hours) and flag.newest_age_hours <= fresh_hours
    flag.has_fresh_news = fresh
    very_fresh = math.isfinite(flag.newest_age_hours) and flag.newest_age_hours <= 3.0

    # Is the tape actually reacting? Unknown RVOL is treated as neutral.
    tape_reacting = (rvol is None) or (math.isfinite(rvol) and rvol >= 1.5)
    tape_ignoring = (
        rvol is not None and math.isfinite(rvol) and rvol < 1.2
    )

    if very_fresh and tape_reacting and (flag.n_relevant >= 2 or rvol is None
                                         or (math.isfinite(rvol) and rvol >= 2.0)):
        flag.risk_level = "high"
        flag.size_multiplier = 0.50
        flag.rr_requirement_multiplier = 1.5
        flag.warnings.append(
            f"breaking news {flag.newest_age_hours:.1f}h old with volume "
            f"confirmation — price may gap through any stop; technical levels "
            f"are unreliable"
        )
    elif fresh and tape_reacting:
        flag.risk_level = "moderate"
        flag.size_multiplier = 0.75
        flag.rr_requirement_multiplier = 1.25
        flag.warnings.append(
            f"fresh catalyst {flag.newest_age_hours:.1f}h old — elevated gap risk"
        )
    elif fresh and tape_ignoring:
        flag.risk_level = "none"
        flag.warnings.append(
            f"recent headlines ({flag.newest_age_hours:.1f}h) but RVOL "
            f"{rvol:.2f}x — tape is not reacting; likely commentary, not a catalyst"
        )

    if any(c in ("sanctions_trade", "conflict") for c in cats):
        flag.warnings.append(
            "geopolitical exposure in headlines — headline risk can reverse "
            "intraday trends without warning"
        )
    return flag


def summarize_geo_categories(headlines: Sequence[Headline]) -> Dict[str, int]:
    """Count headlines per geopolitical category, for the dashboard panel."""
    tally: Dict[str, int] = {}
    for h in headlines:
        for c in h.categories:
            tally[c] = tally.get(c, 0) + 1
    return dict(sorted(tally.items(), key=lambda kv: -kv[1]))


def get_company_names(tickers: Sequence[str], verbose: bool = False) -> Dict[str, str]:
    """Resolve display names, used by the relevance filter.

    Wrapped defensively: yfinance's info endpoint is slow and fails often. A
    missing name degrades the relevance filter to symbol-matching, which still
    works; it must never abort the screen.
    """
    import yfinance as yf

    names: Dict[str, str] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
            nm = info.get("shortName") or info.get("longName") or ""
            if nm:
                names[t] = nm
        except Exception:  # noqa: BLE001
            continue
    return names


# ==============================================================================
# SECTION 5 — SELF TEST (offline; no network)
# ==============================================================================

def run_selftest() -> int:
    """Verify classification and risk logic without touching the network."""
    fails: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            fails.append(name)

    print("\n  MACRO/NEWS SELF TEST (offline)\n  " + "-" * 60)

    # --- REGRESSION: substring matching produced garbage tags ---------------
    # Before word boundaries, "port" matched inside "supports" and "report",
    # so most headlines containing "report" were tagged supply-chain events.
    for bad in ["Cassidy supports Todd Blanche",
                "Takeaways from the disappointing July jobs report",
                "Applied Materials is about to report earnings",
                "Company reports record quarterly profit"]:
        check(f"no false supply_chain tag: {bad[:34]!r}",
              "supply_chain" not in classify_headline(bad),
              f"got {classify_headline(bad)}")

    # --- True positives still fire ------------------------------------------
    cases = [
        ("Sanctions imposed on Russian oil exports", "sanctions_trade"),
        ("New tariffs announced on imported chips", "sanctions_trade"),
        ("Red Sea shipping disrupted by attacks", "supply_chain"),
        ("supply-chain snarls at the port of Los Angeles", "supply_chain"),
        ("Missile strike reported near the capital", "conflict"),
        ("OPEC agrees production cut", "energy"),
        ("Fed signals another rate cut", "monetary"),
    ]
    for title, expect in cases:
        check(f"tags {expect}: {title[:32]!r}", expect in classify_headline(title),
              f"got {classify_headline(title)}")

    check("plural/stem forms match",
          "sanctions_trade" in classify_headline("Sanctioned entities added"))
    check("empty title is safe", classify_headline("") == [])

    # --- Relevance filter ----------------------------------------------------
    now = datetime.now(timezone.utc)
    def h(title, hours, relevant):
        return Headline(title=title, source="test", published=now,
                        age_hours=hours, ticker_relevant=relevant,
                        categories=classify_headline(title))

    # Irrelevant headlines must never create a catalyst flag.
    only_noise = [h("Call options explained", 0.2, False),
                  h("Motley Fool picks 3 stocks", 0.3, False)]
    f0 = assess_catalyst_risk("NVDA", only_noise, rvol=3.0)
    check("irrelevant headlines produce no flag",
          f0.n_relevant == 0 and f0.risk_level == "none")

    # --- RVOL cross-check: the core discriminator ---------------------------
    fresh = [h("DraftKings misses Q2 estimates", 1.0, True),
             h("DraftKings earnings call highlights", 2.0, True)]

    hot = assess_catalyst_risk("DKNG", fresh, rvol=4.0)
    check("fresh news + high RVOL => high risk", hot.risk_level == "high",
          hot.risk_level)
    check("high risk halves size", abs(hot.size_multiplier - 0.5) < 1e-9)
    check("high risk raises R:R bar", hot.rr_requirement_multiplier > 1.0)

    quiet = assess_catalyst_risk("DKNG", fresh, rvol=0.8)
    check("fresh news + LOW RVOL => not escalated", quiet.risk_level == "none",
          quiet.risk_level)
    check("quiet tape leaves size untouched",
          abs(quiet.size_multiplier - 1.0) < 1e-9)
    check("quiet tape explains itself",
          any("not reacting" in w for w in quiet.warnings))

    stale = [h("DraftKings misses Q2 estimates", 40.0, True)]
    check("stale news does not flag",
          assess_catalyst_risk("DKNG", stale, rvol=4.0).risk_level == "none")

    # --- No directional inference anywhere ----------------------------------
    check("CatalystFlag exposes no direction/sentiment field",
          not any(k in CatalystFlag.__dataclass_fields__
                  for k in ("direction", "sentiment", "sentiment_score")))

    # --- Stress classification ----------------------------------------------
    lbl, mult = _classify_stress(75.0, 4)
    check("broad high stress cuts size hard", mult == 0.50, f"{lbl} {mult}")
    lbl2, mult2 = _classify_stress(75.0, 1)
    check("narrow high stress cuts less (breadth matters)", mult2 == 0.75,
          f"{lbl2} {mult2}")
    check("normal regime leaves size alone", _classify_stress(50.0, 0)[1] == 1.0)
    check("unknown index is safe", _classify_stress(float('nan'), 0)[1] == 1.0)
    check("logistic is strictly bounded, never saturates to 0/1",
          0.0 < _logistic(-99) < _logistic(99) < 1.0)
    check("logistic survives extreme input without OverflowError",
          math.isfinite(_logistic(-1e6)) and math.isfinite(_logistic(1e6)))
    check("logistic handles NaN", _logistic(float('nan')) == 0.5)
    check("logistic is monotonic", _logistic(-2) < _logistic(0) < _logistic(2))
    check("logistic is centred at 0.5", abs(_logistic(0) - 0.5) < 1e-12)

    # --- Time parsing --------------------------------------------------------
    check("ISO-Z timestamp parses",
          _parse_time("2026-08-07T16:56:04Z") is not None)
    check("RFC-822 timestamp parses",
          _parse_time("Fri, 07 Aug 2026 16:56:04 +0000") is not None)
    check("garbage timestamp returns None", _parse_time("not a date") is None)
    check("None timestamp returns None", _parse_time(None) is None)

    print("  " + "-" * 60)
    if fails:
        print(f"  {len(fails)} FAILURE(S): {', '.join(fails)}\n")
        return 1
    print("  All checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_selftest())
