#!/usr/bin/env python3
"""Describe the frozen rule's failure without fitting another selector."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


LABELS = (96, 192, 288, 384, 480, 576)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: diagnose_failure.py data/historical_curves.csv")

    rows = list(csv.DictReader(Path(sys.argv[1]).open()))
    evidence = []
    for row in rows:
        gaps = {
            label: float(row[f"e{label}"]) - float(row[f"t{label}"])
            for label in LABELS
        }
        first_change = gaps[192] - gaps[96]
        second_change = gaps[288] - gaps[192]
        trigger = first_change * second_change < 0
        crossed_after_288 = gaps[288] * gaps[576] < 0
        endpoint_concordant = (
            row["final_winner"]
            == row["test_nll_winner"]
            == row["test_exact_winner"]
        )
        evidence.append(
            {
                "seed": int(row["seed"]),
                "split": row["split"],
                "gaps": {str(label): gaps[label] for label in LABELS},
                "direction_reversal_trigger": trigger,
                "crossed_after_288": crossed_after_288,
                "final_winner": row["final_winner"],
                "endpoint_winners_concordant": endpoint_concordant,
            }
        )

    triggered = [row for row in evidence if row["direction_reversal_trigger"]]
    late_example_wins = [row for row in evidence if row["final_winner"] == "example"]
    false_triggers = [
        row["seed"]
        for row in triggered
        if row["final_winner"] != "example"
    ]
    result = {
        "status": "POST_HOC_DESCRIPTIVE_NOT_A_SELECTOR",
        "all_token_lower_at_288": all(row["gaps"]["288"] > 0 for row in evidence),
        "all_endpoint_winners_concordant": all(
            row["endpoint_winners_concordant"] for row in evidence
        ),
        "eventual_example_wins": len(late_example_wins),
        "eventual_example_wins_triggered": sum(
            row["direction_reversal_trigger"] for row in late_example_wins
        ),
        "triggers": len(triggered),
        "false_trigger_seeds": false_triggers,
        "seed_20260819_gap_path": next(
            row["gaps"] for row in evidence if row["seed"] == 20260819
        ),
        "one_of_three_fallback_mean_updates": (864 + 864 + 1152) / 3,
        "one_of_three_fallback_savings_fraction": 1 - ((864 + 864 + 1152) / 3) / 1152,
        "rows": evidence,
    }

    assert len(evidence) == 8, result
    assert result["all_token_lower_at_288"], result
    assert result["all_endpoint_winners_concordant"], result
    assert result["eventual_example_wins"] == 3, result
    assert result["eventual_example_wins_triggered"] == 3, result
    assert result["triggers"] == 4, result
    assert result["false_trigger_seeds"] == [20260819], result
    assert result["seed_20260819_gap_path"] == {
        "96": 0.135,
        "192": 0.269,
        "288": 0.16300000000000003,
        "384": 0.21100000000000002,
        "480": 0.098,
        "576": 0.173,
    }, result
    assert result["one_of_three_fallback_mean_updates"] == 960, result
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
