from __future__ import annotations

import pandas as pd

from src.category_values import add_batter_category_values, add_pitcher_category_values


def test_add_batter_category_values_rewards_multi_category_help() -> None:
    df = pd.DataFrame(
        [
            {"Name": "Power Speed", "R": 8, "HR": 3, "RBI": 9, "SB": 2, "H": 10, "AB": 25},
            {"Name": "Quiet Week", "R": 1, "HR": 0, "RBI": 1, "SB": 0, "H": 3, "AB": 24},
        ]
    )

    scored = add_batter_category_values(df)

    power_speed = scored[scored["name"].eq("Power Speed")].iloc[0]
    quiet_week = scored[scored["name"].eq("Quiet Week")].iloc[0]
    assert power_speed["category_value"] > quiet_week["category_value"]
    assert "avg_impact_value" in scored.columns


def test_add_pitcher_category_values_rewards_rate_and_counting_stats() -> None:
    df = pd.DataFrame(
        [
            {"Name": "Ace", "W": 1, "SV": 0, "SO": 12, "ERA": 1.50, "WHIP": 0.80, "IP": 6},
            {"Name": "Rough", "W": 0, "SV": 0, "SO": 2, "ERA": 9.00, "WHIP": 2.00, "IP": 4},
        ]
    )

    scored = add_pitcher_category_values(df)

    ace = scored[scored["name"].eq("Ace")].iloc[0]
    rough = scored[scored["name"].eq("Rough")].iloc[0]
    assert ace["category_value"] > rough["category_value"]
    assert "era_impact_value" in scored.columns
