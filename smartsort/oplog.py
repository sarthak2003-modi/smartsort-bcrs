"""Operator log — append-only JSONL / CSV of every trial check."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from uuid import uuid4

from smartsort.pipeline import Decision

PathLike = Union[str, Path]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_log_entry(decision: Decision, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    entry = {
        "id": f"chk-{uuid4()}",
        "at": _utc_now(),
        "action": decision.action,
        "eligible": decision.eligible,
        "refund_cents": decision.refund_cents,
        "category": decision.category,
        "label": decision.label,
        "cv_confidence": decision.cv_confidence,
        "fusion_score": decision.fusion_score,
        "weight_g": decision.weight_g,
        "weight_score": decision.weight_score,
        "reasons": list(decision.reasons),
        "latency_ms": decision.latency_ms,
        "material": decision.material,
    }
    if extra:
        entry.update(extra)
    return entry


class OperatorLog:
    """Local operator log stored as JSONL (primary) with optional CSV mirror."""

    def __init__(self, jsonl_path: PathLike, csv_path: Optional[PathLike] = None) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path) if csv_path else None
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, decision: Decision, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        entry = make_log_entry(decision, extra)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if self.csv_path:
            self._append_csv(entry)
        return entry

    def _append_csv(self, entry: Dict[str, Any]) -> None:
        assert self.csv_path is not None
        flat = {**entry, "reasons": "|".join(entry.get("reasons") or [])}
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(flat.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(flat)

    def read_all(self) -> List[Dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
