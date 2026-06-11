"""Minimal weekly model training helpers.

The real model needs historical weekly examples:
features known before a week starts, plus the target outcome from that week.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

NON_FEATURE_COLUMNS = {
    "week_start",
    "week_end",
    "player_type",
    "player_key",
    "playerid",
    "mlbamid",
    "name",
    "team",
    "opponent",
    "opponents",
    "target",
}
TARGET_COMPONENT_COLUMNS = {
    "category_value",
    "runs",
    "home_runs",
    "rbi",
    "stolen_bases",
    "avg_impact",
    "wins",
    "saves",
    "strikeouts",
    "era_impact",
    "whip_impact",
}


def chronological_train_validation_test_split(
    df: pd.DataFrame,
    date_column: str = "week_start",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by week so validation and test are future periods."""
    if date_column not in df.columns:
        raise ValueError(f"Missing date column: {date_column}")

    work = df.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    dates = sorted(work[date_column].dropna().unique())

    if len(dates) < 3:
        raise ValueError(
            "Need at least three distinct weeks for train/validation/test splitting."
        )

    train_count = max(1, int(len(dates) * train_fraction))
    validation_count = max(1, int(len(dates) * validation_fraction))
    if train_count + validation_count >= len(dates):
        train_count = max(1, len(dates) - 2)
        validation_count = 1

    train_dates = dates[:train_count]
    validation_dates = dates[train_count : train_count + validation_count]
    test_dates = dates[train_count + validation_count :]

    train = work[work[date_column].isin(train_dates)].copy()
    validation = work[work[date_column].isin(validation_dates)].copy()
    test = work[work[date_column].isin(test_dates)].copy()
    return train, validation, test


def train_weekly_regression_model(
    training_data_path: Path,
    target_column: str,
    date_column: str = "week_start",
    model_output_path: Path | None = None,
) -> dict[str, Any]:
    """Train a simple tree-based regression model on weekly historical rows."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    df = _load_table(training_data_path)
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    train, validation, test = chronological_train_validation_test_split(
        df,
        date_column=date_column,
    )
    feature_columns = _numeric_feature_columns(train, target_column, date_column)
    if not feature_columns:
        raise ValueError("No numeric feature columns found for model training.")

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        random_state=42,
    )
    model.fit(train[feature_columns], train[target_column])

    metrics = {
        "feature_count": len(feature_columns),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "validation_mae": _mae(model, validation, feature_columns, target_column),
        "test_mae": _mae(model, test, feature_columns, target_column),
        "validation_r2": _r2(model, validation, feature_columns, target_column),
        "test_r2": _r2(model, test, feature_columns, target_column),
    }

    if model_output_path is not None:
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        with model_output_path.open("wb") as model_file:
            pickle.dump(
                {
                    "model": model,
                    "feature_columns": feature_columns,
                    "target_column": target_column,
                    "date_column": date_column,
                    "metrics": metrics,
                },
                model_file,
            )

    return {"model": model, "feature_columns": feature_columns, "metrics": metrics}


def train_weekly_over_under_model(
    training_data_path: Path,
    target_column: str = "category_value",
    date_column: str = "week_start",
    model_output_path: Path | None = None,
    report_output_path: Path | None = None,
) -> dict[str, Any]:
    """Train a Savant-feature model to predict above/below player baseline value."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    df = _load_table(training_data_path)
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    prepared = _add_player_baseline_features(df, target_column, date_column)
    prepared["over_under_target"] = (
        pd.to_numeric(prepared[target_column], errors="coerce")
        - pd.to_numeric(prepared["player_baseline_value"], errors="coerce")
    )
    prepared = prepared[prepared[target_column].notna()].copy()

    train, validation, test = chronological_train_validation_test_split(
        prepared,
        date_column=date_column,
    )
    feature_columns = _over_under_feature_columns(train)
    if not feature_columns:
        raise ValueError("No pre-week feature columns found for model training.")

    best_model = None
    best_params = None
    best_validation_mae = None
    tuning_rows = []
    for params in _small_tuning_grid():
        model = HistGradientBoostingRegressor(random_state=42, **params)
        model.fit(train[feature_columns], train["over_under_target"])
        validation_prediction = _final_over_under_prediction(
            model,
            validation,
            feature_columns,
        )
        validation_mae = float(
            mean_absolute_error(validation[target_column], validation_prediction)
        )
        tuning_rows.append({**params, "validation_mae": validation_mae})
        if best_validation_mae is None or validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_model = model
            best_params = params

    assert best_model is not None
    validation_prediction = _final_over_under_prediction(
        best_model,
        validation,
        feature_columns,
    )
    test_prediction = _final_over_under_prediction(best_model, test, feature_columns)
    validation_baseline = pd.to_numeric(
        validation["player_baseline_value"],
        errors="coerce",
    )
    test_baseline = pd.to_numeric(test["player_baseline_value"], errors="coerce")

    metrics = {
        "feature_count": len(feature_columns),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_weeks": int(pd.to_datetime(train[date_column]).nunique()),
        "validation_weeks": int(pd.to_datetime(validation[date_column]).nunique()),
        "test_weeks": int(pd.to_datetime(test[date_column]).nunique()),
        "target_mean": float(pd.to_numeric(prepared[target_column], errors="coerce").mean()),
        "target_std": float(pd.to_numeric(prepared[target_column], errors="coerce").std()),
        "baseline_validation_mae": float(
            mean_absolute_error(validation[target_column], validation_baseline)
        ),
        "model_validation_mae": float(
            mean_absolute_error(validation[target_column], validation_prediction)
        ),
        "baseline_test_mae": float(mean_absolute_error(test[target_column], test_baseline)),
        "model_test_mae": float(mean_absolute_error(test[target_column], test_prediction)),
        "validation_r2": float(r2_score(validation[target_column], validation_prediction)),
        "test_r2": float(r2_score(test[target_column], test_prediction)),
        "validation_correlation": _correlation(validation[target_column], validation_prediction),
        "test_correlation": _correlation(test[target_column], test_prediction),
        "validation_direction_accuracy": _direction_accuracy(
            validation[target_column],
            validation_baseline,
            validation_prediction,
        ),
        "test_direction_accuracy": _direction_accuracy(
            test[target_column],
            test_baseline,
            test_prediction,
        ),
        "best_params": best_params,
        "tuning_results": tuning_rows,
    }
    metrics["validation_mae_improvement_pct"] = _improvement_pct(
        metrics["baseline_validation_mae"],
        metrics["model_validation_mae"],
    )
    metrics["test_mae_improvement_pct"] = _improvement_pct(
        metrics["baseline_test_mae"],
        metrics["model_test_mae"],
    )

    payload = {
        "model": best_model,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "date_column": date_column,
        "mode": "over_under_player_baseline",
        "metrics": metrics,
    }
    if model_output_path is not None:
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        with model_output_path.open("wb") as model_file:
            pickle.dump(payload, model_file)
    if report_output_path is not None:
        report_output_path.parent.mkdir(parents=True, exist_ok=True)
        report_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return payload


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Training data must be a .csv or .parquet file.")


def _numeric_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    date_column: str,
) -> list[str]:
    excluded = NON_FEATURE_COLUMNS | TARGET_COMPONENT_COLUMNS | {target_column, date_column}
    numeric_columns = df.select_dtypes(include="number").columns
    return [column for column in numeric_columns if column not in excluded]


def _over_under_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [
        column
        for column in df.columns
        if column.endswith("_prev_1w")
        or column.endswith("_prev_2w")
        or column.endswith("_prev_4w")
    ]
    feature_columns.append("player_baseline_value")
    return [
        column
        for column in feature_columns
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]


def _add_player_baseline_features(
    df: pd.DataFrame,
    target_column: str,
    date_column: str,
) -> pd.DataFrame:
    work = df.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    work = work.sort_values(["mlbamid", date_column]).reset_index(drop=True)
    expanding = work.groupby("mlbamid", dropna=False)[target_column].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )
    global_by_week = (
        work.sort_values(date_column)
        .groupby(date_column, dropna=False)[target_column]
        .mean()
        .shift(1)
        .expanding(min_periods=1)
        .mean()
    )
    global_baseline = work[date_column].map(global_by_week).fillna(
        work[target_column].mean()
    )
    work["player_baseline_value"] = expanding.fillna(global_baseline).fillna(0)
    return work


def _small_tuning_grid() -> list[dict[str, Any]]:
    return [
        {"learning_rate": learning_rate, "max_iter": max_iter, "max_leaf_nodes": leaves}
        for learning_rate in [0.03, 0.06, 0.10]
        for max_iter in [100, 200]
        for leaves in [15, 31]
    ]


def _final_over_under_prediction(
    model: Any,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    residual = model.predict(df[feature_columns])
    baseline = pd.to_numeric(df["player_baseline_value"], errors="coerce").fillna(0)
    return baseline + residual


def _correlation(actual: pd.Series, predicted: pd.Series) -> float:
    combined = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if combined.empty:
        return 0.0
    corr = combined["actual"].corr(combined["predicted"])
    return 0.0 if pd.isna(corr) else float(corr)


def _direction_accuracy(
    actual: pd.Series,
    baseline: pd.Series,
    predicted: pd.Series,
) -> float:
    actual_direction = pd.to_numeric(actual, errors="coerce") - pd.to_numeric(
        baseline,
        errors="coerce",
    )
    predicted_direction = pd.to_numeric(predicted, errors="coerce") - pd.to_numeric(
        baseline,
        errors="coerce",
    )
    valid = actual_direction.notna() & predicted_direction.notna()
    if not valid.any():
        return 0.0
    return float((actual_direction.loc[valid].ge(0) == predicted_direction.loc[valid].ge(0)).mean())


def _improvement_pct(baseline_mae: float, model_mae: float) -> float:
    if baseline_mae == 0:
        return 0.0
    return float((baseline_mae - model_mae) / baseline_mae * 100)


def _mae(
    model: Any,
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> float:
    from sklearn.metrics import mean_absolute_error

    predictions = model.predict(df[feature_columns])
    return float(mean_absolute_error(df[target_column], predictions))


def _r2(
    model: Any,
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> float:
    from sklearn.metrics import r2_score

    predictions = model.predict(df[feature_columns])
    return float(r2_score(df[target_column], predictions))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a weekly regression model on historical training rows."
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--date-column", default="week_start")
    parser.add_argument("--model-output", type=Path, default=MODELS_DIR / "weekly_model.pkl")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument(
        "--over-under",
        action="store_true",
        help="Train a Savant-feature model for above/below player baseline value.",
    )
    args = parser.parse_args()

    if args.over_under:
        result = train_weekly_over_under_model(
            training_data_path=args.training_data,
            target_column=args.target,
            date_column=args.date_column,
            model_output_path=args.model_output,
            report_output_path=args.report_output,
        )
    else:
        result = train_weekly_regression_model(
            training_data_path=args.training_data,
            target_column=args.target,
            date_column=args.date_column,
            model_output_path=args.model_output,
        )

    print("Model training complete.")
    print(f"Features used: {result['metrics']['feature_count']}")
    if args.over_under:
        print(f"Baseline validation MAE: {result['metrics']['baseline_validation_mae']:.3f}")
        print(f"Model validation MAE: {result['metrics']['model_validation_mae']:.3f}")
        print(f"Baseline test MAE: {result['metrics']['baseline_test_mae']:.3f}")
        print(f"Model test MAE: {result['metrics']['model_test_mae']:.3f}")
        print(f"Test correlation: {result['metrics']['test_correlation']:.3f}")
        print(f"Test direction accuracy: {result['metrics']['test_direction_accuracy']:.3f}")
    else:
        print(f"Validation MAE: {result['metrics']['validation_mae']:.3f}")
        print(f"Test MAE: {result['metrics']['test_mae']:.3f}")
    print(f"Model output: {args.model_output}")
    if args.report_output:
        print(f"Report output: {args.report_output}")


if __name__ == "__main__":
    main()
