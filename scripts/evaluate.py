#!/usr/bin/env python3
"""Evaluate labelled trial cases against SmartSort design targets (no fabricated results)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smartsort.metrics import TrialCase, evaluate_trials


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="JSON list of {y_true_eligible, y_pred_eligible, latency_s?, category_true?, category_pred?}",
    )
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = [TrialCase(**row) for row in raw]
    report = evaluate_trials(cases)
    text = json.dumps(report.to_dict(), indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
