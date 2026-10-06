#!/usr/bin/env python3
"""Reproduce the discovery-only worst-case-envelope freeze-thaw audit."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


EXPECTED_SEEDS = (20260812, 20260813, 20260814, 20260815, 20260816)
RUNGS = (192, 288, 384, 480)
FINAL = 576
FULL_PAIR_COST = 2 * FINAL
TARGET_MEAN_COST = 864


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if tuple(int(row["seed"]) for row in rows) != EXPECTED_SEEDS:
        raise ValueError("expected exactly ordered discovery seeds 20260812-20260816")
    if any(row["split"] != "discovery" for row in rows):
        raise ValueError("non-discovery row refused")
    for row in rows:
        if row["final_winner"] not in {"token", "example"}:
            raise ValueError("invalid final winner")
        for label in RUNGS + (FINAL,):
            if not all(math.isfinite(float(row[f"{arm}{label}"])) for arm in ("t", "e")):
                raise ValueError("non-finite validation loss")
    return rows


def gap(row: dict[str, str], label: int) -> float:
    return float(row[f"e{label}"]) - float(row[f"t{label}"])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: evaluate_freeze_thaw.py DISCOVERY_CSV")
    rows = load(Path(sys.argv[1]))
    folds = []
    for holdout in rows:
        calibration = [row for row in rows if row is not holdout]
        prediction = None
        decision_label = None
        for label in RUNGS:
            radius = max(abs(gap(row, FINAL) - gap(row, label)) for row in calibration)
            observed = gap(holdout, label)
            if observed - radius > 0:
                prediction, decision_label = "token", label
                break
            if observed + radius < 0:
                prediction, decision_label = "example", label
                break
        cost = FULL_PAIR_COST if decision_label is None else FINAL + decision_label
        folds.append(
            {
                "seed": int(holdout["seed"]),
                "final_winner": holdout["final_winner"],
                "prediction": prediction,
                "decision_label": decision_label,
                "updates": cost,
                "correct": prediction is not None and prediction == holdout["final_winner"],
            }
        )

    total = sum(fold["updates"] for fold in folds)
    selected_correct = sum(fold["correct"] for fold in folds)
    result = {
        "classification": "BOUNDED_NEGATIVE",
        "method": "worst_case_envelope_freeze_thaw",
        "folds": folds,
        "selected_correct": selected_correct,
        "abstentions": sum(fold["prediction"] is None for fold in folds),
        "cohort_updates": total,
        "mean_updates": total / len(folds),
        "savings_fraction": 1.0 - total / (FULL_PAIR_COST * len(folds)),
        "budget_gate_passed": total <= TARGET_MEAN_COST * len(folds),
        "stress_rows_read": False,
    }
    assert [(fold["prediction"], fold["decision_label"]) for fold in folds] == [
        ("token", 480),
        (None, None),
        (None, None),
        ("token", 480),
        ("token", 480),
    ]
    assert total == 5472 and selected_correct == 3
    assert abs(result["savings_fraction"] - 0.05) < 1e-12
    assert result["budget_gate_passed"] is False
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
