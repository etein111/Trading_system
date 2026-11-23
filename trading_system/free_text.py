"""Free RSS/Atom press-release ingestion with point-in-time text scores.

The feed list is user supplied, so the module works with company IR feeds,
PR Newswire, GlobeNewswire, and other public RSS/Atom sources without an API
key. It preserves publication time and source URL for historical replay.
"""

from __future__ import annotations

import email.utils
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
import yaml

from .text_signal import FilingRecord, html_to_text, score_filing


@dataclass
class FeedSpec:
    symbol: str
    url: str
    source: str = "rss"


def load_feed_specs(path: str) -> List[FeedSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = raw.get("feeds", raw if isinstance(raw, list) else [])
    specs = []
    for item in entries:
        if not isinstance(item, dict) or not item.get("symbol") or not item.get("url"):
            continue
        specs.append(FeedSpec(str(item["symbol"]).upper(), str(item["url"]), str(item.get("source", "rss"))))
    if not specs:
        raise ValueError("feed file must contain feeds with symbol and url")
    return specs


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(node: ET.Element, names: Iterable[str]) -> str:
    wanted = {name.lower() for name in names}
    for child in list(node):
        if _local_name(child.tag) in wanted and child.text:
            return child.text.strip()
    return ""


def _parse_date(value: str) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        return pd.Timestamp(parsed).tz_convert(None) if parsed.tzinfo else pd.Timestamp(parsed)
    except Exception:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        return parsed.tz_convert(None) if pd.notna(parsed) else None


def parse_feed(payload: str, spec: FeedSpec) -> List[Dict[str, object]]:
    root = ET.fromstring(payload)
    rows = []
    for node in root.iter():
        name = _local_name(node.tag)
        if name not in {"item", "entry"}:
            continue
        title = _child_text(node, ["title"])
        link = _child_text(node, ["link", "guid", "id"])
        if not link:
            for child in list(node):
                if _local_name(child.tag) == "link" and child.attrib.get("href"):
                    link = child.attrib["href"]
                    break
        published = _child_text(node, ["pubdate", "published", "updated", "date"])
        available_at = _parse_date(published)
        if not title or not link or available_at is None:
            continue
        summary = _child_text(node, ["description", "summary", "content", "encoded"])
        rows.append({
            "symbol": spec.symbol,
            "title": html_to_text(title),
            "summary": html_to_text(summary),
            "source": spec.source,
            "url": link,
            "available_at": available_at,
        })
    return rows


def build_rss_signals(
    feed_specs: List[FeedSpec],
    output: str,
    asof=None,
    lookback_days: Optional[int] = None,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    rows = []
    failures = {}
    cutoff = pd.Timestamp(asof) if asof else None
    if cutoff is not None and cutoff.hour == 0 and cutoff.minute == 0 and cutoff.second == 0:
        cutoff = cutoff.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    if lookback_days and cutoff is not None:
        cutoff_start = cutoff - pd.Timedelta(days=int(lookback_days))
    else:
        cutoff_start = None
    for spec in feed_specs:
        try:
            response = requests.get(spec.url, timeout=30, headers={"User-Agent": "TradingSystem RSS research"})
            response.raise_for_status()
            for item in parse_feed(response.text, spec):
                item_date = pd.Timestamp(item["available_at"])
                if cutoff is not None and item_date > cutoff:
                    continue
                if cutoff_start is not None and item_date < cutoff_start:
                    continue
                filing = FilingRecord(
                    symbol=item["symbol"],
                    cik="",
                    form="RSS",
                    accession=item["url"],
                    primary_document=item["title"],
                    filed_at=item_date,
                    period_end=None,
                    url=item["url"],
                )
                scored = score_filing(filing, item["title"] + " " + item["summary"])
                scored["source_type"] = item["source"]
                scored["title"] = item["title"]
                rows.append(scored)
            time.sleep(sleep_seconds)
        except Exception as exc:
            failures[spec.url] = str(exc)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["symbol", "score", "event_risk", "available_at"])
    else:
        frame = frame.drop_duplicates(subset=["symbol", "source", "accession"], keep="last")
        frame = frame.sort_values(["available_at", "symbol"]).reset_index(drop=True)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    destination.with_suffix(".summary.json").write_text(
        json.dumps({"feed_count": len(feed_specs), "rows": len(frame), "failures": failures}, indent=2),
        encoding="utf-8",
    )
    frame.attrs["failures"] = failures
    return frame
