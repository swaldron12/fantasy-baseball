from __future__ import annotations

import pandas as pd

from src.modeling import (
    _numeric_feature_columns,
    train_weekly_over_under_model,
    chronological_train_validation_test_split,
)


def test_chronological_split_keeps_future_weeks_separate() -> None:
    df = pd.DataFrame(
        {
            "week_start": pd.date_range("2026-04-01", periods=10, freq="W"),
            "feature": range(10),
            "weekly_fantasy_points": range(10),
        }
    )

    train, validation, test = chronological_train_validation_test_split(df)

    assert train["week_start"].max() < validation["week_start"].min()
    assert validation["week_start"].max() < test["week_start"].min()
    assert len(train) == 7
    assert len(validation) == 1
    assert len(test) == 2


def test_numeric_features_exclude_actual_outcome_columns() -> None:
    df = pd.DataFrame(
        {
            "category_value": [1.0, 2.0],
            "home_runs": [1, 0],
            "rbi": [3, 1],
            "woba_prev_4w": [0.350, 0.280],
            "player_baseline_value": [0.2, -0.1],
        }
    )

    features = _numeric_feature_columns(df, "category_value", "week_start")

    assert "woba_prev_4w" in features
    assert "home_runs" not in features
    assert "rbi" not in features


def test_over_under_model_trains_on_pre_week_features(tmp_path) -> None:
    rows = []
    for week in pd.date_range("2026-04-06", periods=8, freq="W-MON"):
        for player_id in range(1, 6):
            signal = player_id / 10
            rows.append(
                {
                    "mlbamid": player_id,
                    "week_start": week,
                    "name": f"Player {player_id}",
                    "category_value": signal,
                    "home_runs": 99,
                    "woba_prev_4w": signal,
                    "xwoba_prev_4w": signal,
                }
            )
    path = tmp_path / "training.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    result = train_weekly_over_under_model(path, model_output_path=tmp_path / "model.pkl")

    assert result["metrics"]["feature_count"] == 3
    assert "home_runs" not in result["feature_columns"]
    assert (tmp_path / "model.pkl").exists()
