"""Point-in-time SEC filing text signals.

This module deliberately keeps the text layer deterministic and auditable. It
uses SEC filing timestamps, extracts a bounded set of event and sentiment
signals, and emits evidence excerpts. Optional transformer models can be added
later, but the default path has no large model dependency and is safe for
historical replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests

from .public_training import SecCompanyFactsClient

SEC_HEADERS = {
    "User-Agent": os.environ.get("SEC_USER_AGENT", "TradingSystem research contact research@example.com")
}
DEFAULT_FORMS = ("8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A")

POSITIVE_TERMS = {
    "raised guidance": 2.5,
    "reaffirmed guidance": 1.0,
    "record revenue": 1.5,
    "strong demand": 1.2,
    "new contract": 1.0,
    "backlog": 0.6,
    "share repurchase": 0.6,
    "dividend increase": 0.8,
    "improved margin": 0.8,
}
NEGATIVE_TERMS = {
    "lowered guidance": 3.0,
    "reduced guidance": 3.0,
    "weak demand": 1.8,
    "material weakness": 3.0,
    "going concern": 3.5,
    "restatement": 3.5,
    "subpoena": 2.0,
    "investigation": 1.8,
    "litigation": 1.2,
    "layoff": 1.0,
    "workforce reduction": 1.0,
    "restructuring charge": 1.2,
    "impairment": 1.3,
    "cyberattack": 1.6,
    "data breach": 1.8,
    "recall": 1.5,
    "delayed launch": 1.1,
    "decline in revenue": 1.2,
    "operating loss": 1.0,
}


class _VisibleTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self.parts.append(data)


def html_to_text(payload: str) -> str:
    parser = _VisibleTextParser()
    try:
        parser.feed(payload)
        text = " ".join(parser.parts)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", payload)
    return re.sub(r"\s+", " ", text).strip()


def _timestamp(value, fallback_date=None) -> pd.Timestamp:
    if value:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(parsed):
            return parsed.tz_convert(None)
    fallback = pd.Timestamp(fallback_date) if fallback_date is not None else pd.Timestamp.utcnow()
    return fallback.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)


@dataclass
class FilingRecord:
    symbol: str
    cik: str
    form: str
    accession: str
    primary_document: str
    filed_at: pd.Timestamp
    period_end: Optional[pd.Timestamp]
    url: str
    items: str = ""


class SecFilingTextClient:
    """Fetch and cache SEC submissions and filing documents."""

    def __init__(self, cache_dir: str = "research/sec_text_cache", sleep_seconds: float = 0.2):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_seconds = sleep_seconds
        self._ticker_map = None
        self._facts_client = SecCompanyFactsClient(cache_dir=str(self.cache_dir))

    def ticker_map(self) -> Dict[str, str]:
        if self._ticker_map is None:
            self._ticker_map = self._facts_client.ticker_map()
        return self._ticker_map

    def _request_json(self, url: str) -> Dict:
        response = requests.get(url, headers=SEC_HEADERS, timeout=60)
        response.raise_for_status()
        time.sleep(self.sleep_seconds)
        return response.json()

    def submissions(self, symbol: str) -> Dict:
        cik = self.ticker_map().get(symbol.upper())
        if cik is None:
            raise KeyError(f"SEC CIK not found for {symbol}")
        path = self.cache_dir / f"submissions_{cik}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        payload = self._request_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def filings(self, symbol: str, forms: Sequence[str] = DEFAULT_FORMS, limit: int = 200) -> List[FilingRecord]:
        payload = self.submissions(symbol)
        cik = str(payload.get("cik", "")).strip() or self.ticker_map()[symbol.upper()]
        recent = payload.get("filings", {}).get("recent", {})
        rows = []
        for index, form in enumerate(recent.get("form", [])):
            if form not in set(forms):
                continue
            accession = recent.get("accessionNumber", [""])[index]
            primary = recent.get("primaryDocument", [""])[index]
            filed = recent.get("filingDate", [None])[index]
            acceptance = recent.get("acceptanceDateTime", [None])[index]
            report_date = recent.get("reportDate", [None])[index]
            items = recent.get("items", [""])[index]
            accession_path = accession.replace("-", "")
            cik_path = str(int(cik))
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_path}/{accession_path}/{primary}"
            rows.append(
                FilingRecord(
                    symbol=symbol.upper(),
                    cik=cik,
                    form=form,
                    accession=accession,
                    primary_document=primary,
                    filed_at=_timestamp(acceptance, filed),
                    period_end=pd.to_datetime(report_date, errors="coerce") if report_date else None,
                    url=url,
                    items=items or "",
                )
            )
            if len(rows) >= limit:
                break
        return rows

    def text(self, filing: FilingRecord) -> str:
        digest = hashlib.sha256(filing.url.encode("utf-8")).hexdigest()[:24]
        path = self.cache_dir / f"filing_{digest}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        response = requests.get(filing.url, headers=SEC_HEADERS, timeout=60)
        response.raise_for_status()
        text = html_to_text(response.text)
        path.write_text(text, encoding="utf-8")
        time.sleep(self.sleep_seconds)
        return text


def _contextual_hits(text: str, terms: Dict[str, float], form: str) -> Dict[str, float]:
    lower = text.lower()
    generic_terms = {
        "litigation", "investigation", "impairment", "cyberattack",
        "data breach", "operating loss", "recall",
    }
    event_context = (
        "we have identified", "we identified", "management identified",
        "we received", "received a subpoena", "was served", "has been recorded",
        "incurred", "announced", "entered into", "determined that",
        "filed a complaint", "filed for", "reported",
    )
    hits = {}
    for term, weight in terms.items():
        positions = [match.start() for match in re.finditer(re.escape(term), lower)]
        if not positions:
            continue
        if form in {"10-K", "10-K/A", "10-Q", "10-Q/A"} and term in generic_terms:
            contextual = 0
            for position in positions:
                window = lower[max(0, position - 220): position + len(term) + 220]
                contextual += int(any(phrase in window for phrase in event_context))
            if contextual == 0:
                continue
            count = contextual
        else:
            count = len(positions)
        hits[term] = float(weight) * min(count, 3)
    return hits


def _explicit_hard_risk(text: str) -> bool:
    lower = text.lower()
    patterns = [
        r"(?:we|management).{0,120}(?:identified|determined|concluded).{0,120}material weakness",
        r"(?:restated|restatement).{0,120}(?:financial statements|financial results)",
        r"going concern",
    ]
    return any(re.search(pattern, lower, flags=re.DOTALL) for pattern in patterns)


def score_filing(filing: FilingRecord, text: str) -> Dict[str, object]:
    positive = _contextual_hits(text, POSITIVE_TERMS, filing.form)
    negative = _contextual_hits(text, NEGATIVE_TERMS, filing.form)
    positive_value = float(sum(positive.values()))
    negative_value = float(sum(negative.values()))
    hard_risk = _explicit_hard_risk(text)
    # 10-K/10-Q contain boilerplate risk factors. Downweight ordinary lexical
    # hits unless an explicit event sentence is present. 8-K is more event-like.
    filing_factor = 1.0 if filing.form in {"8-K", "8-K/A"} else 0.35
    adjusted_negative = negative_value * filing_factor
    score = float(np.tanh((positive_value * filing_factor - adjusted_negative) / 4.0))
    event_risk = float(1.0 - np.exp(-adjusted_negative / 3.0))
    if hard_risk:
        event_risk = 1.0
        score = min(score, -0.75)
    matched = list(positive) + list(negative)
    excerpt = ""
    if matched:
        term = matched[0]
        match = re.search(re.escape(term), text, flags=re.IGNORECASE)
        if match:
            excerpt = text[max(0, match.start() - 180): min(len(text), match.end() + 260)]
    return {
        "symbol": filing.symbol,
        "score": score,
        "event_risk": event_risk,
        "hard_risk": hard_risk,
        "available_at": filing.filed_at,
        "period_end": filing.period_end,
        "form": filing.form,
        "accession": filing.accession,
        "source": filing.url,
        "evidence_id": hashlib.sha256(filing.url.encode("utf-8")).hexdigest()[:16],
        "positive_terms": ",".join(positive),
        "negative_terms": ",".join(negative),
        "evidence_excerpt": excerpt[:600],
    }


def build_text_signals(
    symbols: Iterable[str],
    output: str,
    cache_dir: str = "research/sec_text_cache",
    forms: Sequence[str] = DEFAULT_FORMS,
    limit_per_symbol: int = 100,
) -> pd.DataFrame:
    symbols = list(symbols)
    client = SecFilingTextClient(cache_dir=cache_dir)
    records = []
    failures = {}
    for symbol in symbols:
        try:
            for filing in client.filings(symbol, forms=forms, limit=limit_per_symbol):
                try:
                    records.append(score_filing(filing, client.text(filing)))
                except Exception as exc:
                    failures[f"{symbol}:{filing.accession}"] = str(exc)
        except Exception as exc:
            failures[symbol] = str(exc)
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=["symbol", "score", "event_risk", "available_at"])
    else:
        frame["available_at"] = pd.to_datetime(frame["available_at"])
        frame = frame.sort_values(["available_at", "symbol"]).reset_index(drop=True)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    summary = {
        "symbols_requested": len(symbols),
        "filing_rows": int(len(frame)),
        "symbol_count": int(frame["symbol"].nunique()) if not frame.empty else 0,
        "failures": failures,
        "forms": list(forms),
    }
    destination.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    frame.attrs["failures"] = failures
    return frame
