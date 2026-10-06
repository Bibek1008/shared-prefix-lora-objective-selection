#!/usr/bin/env python3
"""Evaluate the one frozen gap-crossover rule from a compact CSV."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def predict(row: dict[str, str]) -> tuple[str, int]:
    gaps = [
        float(row[f"e{label}"]) - float(row[f"t{label}"])
        for label in (96, 192, 288)
    ]
    if any(gap == 0.0 for gap in gaps):
        return "fallback", 576
    if (gaps[1] - gaps[0]) * (gaps[2] - gaps[1]) < 0:
        return "example", 288
    return ("token" if float(row["t288"]) < float(row["e288"]) else "example"), 288


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: evaluate_trajectory.py data/historical_curves.csv")
    path = Path(sys.argv[1])
    rows = list(csv.DictReader(path.open()))
    evidence = []
    for row in rows:
        prediction, label = predict(row)
        updates = 1152 if label == 576 else 576 + label
        evidence.append({
            "seed": int(row["seed"]),
            "split": row["split"],
            "prediction": prediction,
            "decision_label": label,
            "winner": row["final_winner"],
            "test_nll_winner": row["test_nll_winner"],
            "test_exact_winner": row["test_exact_winner"],
            "updates": updates,
            "savings_fraction": (1152 - updates) / 1152,
        })
    summary = {}
    for split in ("discovery", "stress"):
        selected = [row for row in evidence if row["split"] == split]
        summary[split] = {
            "n": len(selected),
            "final_agreement": sum(r["prediction"] == r["winner"] for r in selected),
            "test_nll_agreement": sum(r["prediction"] == r["test_nll_winner"] for r in selected),
            "test_exact_agreement": sum(r["prediction"] == r["test_exact_winner"] for r in selected),
            "material_every_seed": all(r["savings_fraction"] >= 0.25 for r in selected),
        }
    classification = (
        "SUPPORTED" if summary["discovery"]["final_agreement"] == 5
        and summary["stress"]["final_agreement"] == 3
        and summary["stress"]["test_nll_agreement"] == 3
        and summary["stress"]["test_exact_agreement"] == 3
        and summary["stress"]["material_every_seed"]
        else "BOUNDED_NEGATIVE"
    )
    result = {"classification": classification, "summary": summary, "rows": evidence}
    assert summary["discovery"]["final_agreement"] == 5, result
    assert summary["stress"]["final_agreement"] == 2, result
    assert classification == "BOUNDED_NEGATIVE", result
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
