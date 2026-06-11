"""Create category-value targets from weekly fantasy baseball outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.clean import standardize_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_DIR = PROJECT_ROOT / "data" / "modeling"

BATTER_COUNTING_CATEGORIES = {
    "runs": ["r", "runs"],
    "home_runs": ["hr", "home_runs"],
    "rbi": ["rbi"],
    "stolen_bases": ["sb", "stolen_bases"],
}

PITCHER_COUNTING_CATEGORIES = {
    "wins": ["w", "wins"],
    "saves": ["sv", "saves"],
    "strikeouts": ["so", "k", "strikeouts"],
}


def add_batter_category_values(df: pd.DataFrame) -> pd.DataFrame:
    """Add hitter category component scores and one total category value."""
    work = _standardize_frame(df)
    work["player_type"] = "batter"

    for target, aliases in BATTER_COUNTING_CATEGORIES.items():
        work[target] = _first_numeric(work, aliases)

    work["at_bats"] = _first_numeric(work, ["ab", "at_bats"])
    work["hits"] = _first_numeric(work, ["h", "hits"])
    work["batting_average"] = _first_numeric(work, ["avg", "batting_average"])
    missing_avg = work["batting_average"].isna()
    work.loc[missing_avg, "batting_average"] = _safe_divide(
        work.loc[missing_avg, "hits"],
        work.loc[missing_avg, "at_bats"],
    )

    league_avg = _safe_divide(work["hits"].sum(), work["at_bats"].sum())
    work["avg_impact"] = (
        (work["batting_average"] - league_avg) * work["at_bats"].fillna(0)
    )

    component_columns = [
        "runs",
        "home_runs",
        "rbi",
        "stolen_bases",
        "avg_impact",
    ]
    return _add_value_score(work, component_columns)


def add_pitcher_category_values(df: pd.DataFrame) -> pd.DataFrame:
    """Add pitcher category component scores and one total category value."""
    work = _standardize_frame(df)
    work["player_type"] = "pitcher"

    for target, aliases in PITCHER_COUNTING_CATEGORIES.items():
        work[target] = _first_numeric(work, aliases)

    work["innings_pitched"] = _first_numeric(work, ["ip", "innings_pitched"])
    work["era"] = _first_numeric(work, ["era"])
    work["whip"] = _first_numeric(work, ["whip"])

    league_era = _weighted_average(work["era"], work["innings_pitched"])
    league_whip = _weighted_average(work["whip"], work["innings_pitched"])

    work["era_impact"] = (
        (league_era - work["era"]) * work["innings_pitched"].fillna(0)
    )
    work["whip_impact"] = (
        (league_whip - work["whip"]) * work["innings_pitched"].fillna(0)
    )

    component_columns = [
        "wins",
        "saves",
        "strikeouts",
        "era_impact",
        "whip_impact",
    ]
    return _add_value_score(work, component_columns)


def build_category_value_targets(
    batter_weekly_path: Path,
    pitcher_weekly_path: Path,
    output_dir: Path = MODELING_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and save weekly category-value targets for batters and pitchers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    batters = add_batter_category_values(_load_table(batter_weekly_path))
    pitchers = add_pitcher_category_values(_load_table(pitcher_weekly_path))

    batter_output = output_dir / "weekly_batter_category_targets.parquet"
    pitcher_output = output_dir / "weekly_pitcher_category_targets.parquet"
    combined_output = output_dir / "weekly_category_targets.parquet"

    batters.to_parquet(batter_output, index=False, engine="pyarrow")
    pitchers.to_parquet(pitcher_output, index=False, engine="pyarrow")
    pd.concat([batters, pitchers], ignore_index=True).to_parquet(
        combined_output,
        index=False,
        engine="pyarrow",
    )

    print(f"Batter category targets: {batter_output}")
    print(f"Pitcher category targets: {pitcher_output}")
    print(f"Combined category targets: {combined_output}")
    return batters, pitchers


def _add_value_score(df: pd.DataFrame, component_columns: list[str]) -> pd.DataFrame:
    scored = df.copy()
    score_columns = []

    for column in component_columns:
        value_column = f"{column}_value"
        scored[value_column] = _z_score(scored[column])
        score_columns.append(value_column)

    scored["category_value"] = scored[score_columns].sum(axis=1).round(3)
    return scored


def _standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work.columns = standardize_columns(work.columns)
    return work


def _first_numeric(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    values = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for alias in aliases:
        if alias in df.columns:
            values = values.fillna(pd.to_numeric(df[alias], errors="coerce"))
    return values


def _z_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (numeric - numeric.mean()) / std


def _safe_divide(numerator: object, denominator: object) -> object:
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")
    if isinstance(denominator, pd.Series):
        return numerator / denominator.mask(denominator == 0)
    if pd.isna(denominator) or denominator == 0:
        return pd.NA
    return numerator / denominator


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    valid = numeric_values.notna() & numeric_weights.gt(0)
    if not valid.any():
        return float(numeric_values.mean(skipna=True))
    return float(
        (numeric_values.loc[valid] * numeric_weights.loc[valid]).sum()
        / numeric_weights.loc[valid].sum()
    )


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input table must be a .csv or .parquet file.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build category-value targets from weekly outcome tables."
    )
    parser.add_argument("--batters", type=Path, required=True)
    parser.add_argument("--pitchers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=MODELING_DIR)
    args = parser.parse_args()

    build_category_value_targets(
        batter_weekly_path=args.batters,
        pitcher_weekly_path=args.pitchers,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
