"""Create first-pass weekly fantasy baseball projection tables.

These are baseline projection indexes, not a trained ML model yet. The goal is
to combine current player features with the upcoming weekly schedule in a way
that is easy to inspect and explain.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.clean import standardize_columns
from src.stat_projections import add_hybrid_stat_projections


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SCHEDULE_DIR = PROJECT_ROOT / "data" / "raw" / "schedules"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
DEFAULT_SCHEDULE_PATH = SCHEDULE_DIR / "probable_pitchers_week.csv"
SCHEDULE_TEMPLATE_PATH = SCHEDULE_DIR / "probable_pitchers_template.csv"

SCHEDULE_COLUMNS = [
    "week_start",
    "game_date",
    "team",
    "opponent",
    "home_away",
    "opposing_probable_pitcher_name",
    "opposing_probable_pitcher_id",
    "opposing_probable_pitcher_throws",
    "own_probable_pitcher_name",
    "own_probable_pitcher_id",
    "own_probable_pitcher_throws",
]


def create_schedule_template(path: Path = SCHEDULE_TEMPLATE_PATH) -> Path:
    """Create the empty probable-pitcher schedule template."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=SCHEDULE_COLUMNS).to_csv(path, index=False)
    print(f"Schedule template: {path}")
    return path


def run(
    schedule_path: Path = DEFAULT_SCHEDULE_PATH,
    output_dir: Path = PROJECTIONS_DIR,
) -> None:
    """Build and save weekly batter, pitcher, and combined projections."""
    if not schedule_path.exists():
        create_schedule_template()
        print(f"No schedule found at {schedule_path}")
        print("Fill out the template, save it as probable_pitchers_week.csv, then rerun.")
        return

    schedule = load_probable_pitchers(schedule_path)
    batters = pd.read_parquet(FEATURES_DIR / "batters_features.parquet")
    pitchers = pd.read_parquet(FEATURES_DIR / "pitchers_features.parquet")

    batter_projections = build_batter_weekly_projections(batters, schedule)
    pitcher_projections = build_pitcher_weekly_projections(pitchers, schedule)
    batter_projections, pitcher_projections = add_hybrid_stat_projections(
        batter_projections,
        pitcher_projections,
        schedule,
    )
    save_projection_tables(batter_projections, pitcher_projections, output_dir)


def load_probable_pitchers(schedule_path: Path) -> pd.DataFrame:
    """Load a team-game schedule file and standardize core fields."""
    schedule = pd.read_csv(schedule_path, dtype="string")
    schedule.columns = standardize_columns(schedule.columns)

    missing_columns = [
        column for column in SCHEDULE_COLUMNS if column not in schedule.columns
    ]
    if missing_columns:
        raise ValueError(
            "Probable pitcher schedule is missing columns: "
            + ", ".join(missing_columns)
        )

    schedule = schedule.dropna(how="all").copy()
    if schedule.empty:
        raise ValueError(f"Probable pitcher schedule has no rows: {schedule_path}")

    for column in ["team", "opponent"]:
        schedule[column] = schedule[column].astype("string").str.strip().str.upper()

    schedule["home_away"] = (
        schedule["home_away"].astype("string").str.strip().str.lower()
    )
    for column in ["opposing_probable_pitcher_throws", "own_probable_pitcher_throws"]:
        schedule[column] = _standardize_handedness(schedule[column])

    return schedule


def build_batter_weekly_projections(
    batters: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Project hitter weekly value using skill features and scheduled games."""
    projected = batters.copy()
    projected["team"] = projected["team"].astype("string").str.strip().str.upper()
    has_current_batting_volume = pd.to_numeric(
        projected.get("pa"),
        errors="coerce",
    ).fillna(0).gt(0)
    has_hitter_projection = _first_available(
        projected,
        ["zips_woba", "zips_hr_rate"],
    ).notna()
    projected = projected[has_current_batting_volume | has_hitter_projection].copy()

    schedule_summary = _team_schedule_summary(schedule)
    projected = projected.merge(schedule_summary, on="team", how="left")
    for column in ["projected_games", "games_vs_lhp", "games_vs_rhp"]:
        projected[column] = pd.to_numeric(projected[column], errors="coerce").fillna(0)

    projected["overall_woba_signal"] = _first_available(
        projected,
        ["zips_woba", "woba", "xwoba"],
    )
    projected["matchup_woba_signal"] = _weighted_average(
        projected["woba_vs_lhp"],
        projected["woba_vs_rhp"],
        projected["games_vs_lhp"],
        projected["games_vs_rhp"],
        projected["overall_woba_signal"],
    )

    projected["skill_score"] = _weighted_percentile_score(
        projected,
        [
            ("overall_woba_signal", "higher", 0.25),
            ("xwoba", "higher", 0.20),
            ("matchup_woba_signal", "higher", 0.20),
            ("iso", "higher", 0.15),
            ("zips_hr_rate", "higher", 0.10),
            ("bb_rate", "higher", 0.05),
            ("k_rate", "lower", 0.05),
        ],
    )

    max_games = max(float(projected["projected_games"].max()), 1.0)
    projected["schedule_multiplier"] = projected["projected_games"] / max_games
    projected["weekly_projection_index"] = (
        projected["skill_score"] * projected["schedule_multiplier"]
    ).round(1)
    projected["player_type"] = "batter"

    output_columns = [
        "player_type",
        "player_key",
        "playerid",
        "mlbamid",
        "name",
        "team",
        "projected_games",
        "games_vs_lhp",
        "games_vs_rhp",
        "skill_score",
        "weekly_projection_index",
        "overall_woba_signal",
        "matchup_woba_signal",
        "zips_hr_rate",
    ]
    return _rank_projection_table(projected[output_columns])


def build_pitcher_weekly_projections(
    pitchers: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Project scheduled pitcher value using skill features and opponent splits."""
    starts = _scheduled_pitcher_starts(schedule)
    if starts.empty:
        return _empty_pitcher_projection_table()

    projected = _merge_pitcher_features(starts, pitchers)
    projected = _add_pitcher_matchup_scores(projected)

    projected["skill_score"] = _weighted_percentile_score(
        projected,
        [
            ("k_rate", "higher", 0.22),
            ("zips_k_rate", "higher", 0.16),
            ("bb_rate", "lower", 0.14),
            ("xfip", "lower", 0.16),
            ("siera", "lower", 0.16),
            ("zips_era", "lower", 0.16),
        ],
    )
    projected["matchup_score"] = (
        pd.to_numeric(projected["matchup_score"], errors="coerce").fillna(50)
    )
    projected["start_projection_index"] = (
        0.75 * projected["skill_score"] + 0.25 * projected["matchup_score"]
    )

    projected["player_key"] = projected["player_key"].fillna(
        _normalize_name_series(projected["own_probable_pitcher_name"])
    )
    projected["name"] = projected["name"].fillna(projected["own_probable_pitcher_name"])
    projected["team"] = projected["team"].fillna(projected["scheduled_team"])
    projected["mlbamid"] = projected["mlbamid"].fillna(
        projected["own_probable_pitcher_id"]
    )
    projected["player_type"] = "pitcher"

    grouped = (
        projected.groupby(
            ["player_type", "player_key", "playerid", "mlbamid", "name", "team"],
            dropna=False,
        )
        .agg(
            projected_starts=("start_projection_index", "size"),
            opponents=("opponent", _join_unique),
            skill_score=("skill_score", "max"),
            matchup_score=("matchup_score", "mean"),
            weekly_projection_index=("start_projection_index", "sum"),
        )
        .reset_index()
    )
    grouped["weekly_projection_index"] = grouped["weekly_projection_index"].round(1)
    grouped["matchup_score"] = grouped["matchup_score"].round(1)
    return _rank_projection_table(grouped)


def save_projection_tables(
    batter_projections: pd.DataFrame,
    pitcher_projections: pd.DataFrame,
    output_dir: Path = PROJECTIONS_DIR,
) -> None:
    """Save projections in both CSV and Parquet formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    batter_csv = output_dir / "weekly_batter_projections.csv"
    pitcher_csv = output_dir / "weekly_pitcher_projections.csv"
    combined_csv = output_dir / "weekly_projections.csv"

    batter_projections.to_csv(batter_csv, index=False)
    pitcher_projections.to_csv(pitcher_csv, index=False)

    batter_projections.to_parquet(
        output_dir / "weekly_batter_projections.parquet",
        index=False,
        engine="pyarrow",
    )
    pitcher_projections.to_parquet(
        output_dir / "weekly_pitcher_projections.parquet",
        index=False,
        engine="pyarrow",
    )

    combined = pd.concat(
        [
            _combined_projection_view(batter_projections),
            _combined_projection_view(pitcher_projections),
        ],
        ignore_index=True,
    )
    combined = _rank_projection_table(combined)
    combined.to_csv(combined_csv, index=False)
    combined.to_parquet(
        output_dir / "weekly_projections.parquet",
        index=False,
        engine="pyarrow",
    )

    print(f"Batter projections: {batter_csv}")
    print(f"Pitcher projections: {pitcher_csv}")
    print(f"Combined projections: {combined_csv}")


def _team_schedule_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    work = schedule.copy()
    work["is_lhp"] = work["opposing_probable_pitcher_throws"].eq("L")
    work["is_rhp"] = work["opposing_probable_pitcher_throws"].eq("R")
    return (
        work.groupby("team", dropna=False)
        .agg(
            projected_games=("game_date", "size"),
            games_vs_lhp=("is_lhp", "sum"),
            games_vs_rhp=("is_rhp", "sum"),
        )
        .reset_index()
    )


def _scheduled_pitcher_starts(schedule: pd.DataFrame) -> pd.DataFrame:
    starts = schedule.copy()
    has_pitcher_name = starts["own_probable_pitcher_name"].notna()
    has_pitcher_id = starts["own_probable_pitcher_id"].notna()
    starts = starts[has_pitcher_name | has_pitcher_id].copy()
    starts = starts.rename(columns={"team": "scheduled_team"})
    starts["scheduled_team"] = starts["scheduled_team"].astype("string")
    return starts


def _merge_pitcher_features(
    starts: pd.DataFrame,
    pitchers: pd.DataFrame,
) -> pd.DataFrame:
    pitcher_features = pitchers.copy()
    for column in ["playerid", "mlbamid"]:
        pitcher_features[column] = _string_column(pitcher_features, column)

    starts = starts.copy()
    starts["own_probable_pitcher_id"] = _string_column(
        starts,
        "own_probable_pitcher_id",
    )

    by_mlbam = pitcher_features.dropna(subset=["mlbamid"]).drop_duplicates("mlbamid")
    merged = starts.merge(
        by_mlbam,
        left_on="own_probable_pitcher_id",
        right_on="mlbamid",
        how="left",
    )

    missing_match = merged["player_key"].isna()
    if missing_match.any():
        by_name = pitcher_features.copy()
        by_name["pitcher_name_key"] = _normalize_name_series(by_name["name"])
        by_name = by_name.drop_duplicates("pitcher_name_key")

        name_matches = starts.loc[missing_match].copy()
        name_matches["pitcher_name_key"] = _normalize_name_series(
            name_matches["own_probable_pitcher_name"]
        )
        name_matches = name_matches.merge(by_name, on="pitcher_name_key", how="left")

        feature_columns = list(pitcher_features.columns)
        merged.loc[missing_match, feature_columns] = name_matches[
            feature_columns
        ].to_numpy()

    return merged


def _add_pitcher_matchup_scores(projected: pd.DataFrame) -> pd.DataFrame:
    matchup_scores = _load_team_matchup_scores()
    if matchup_scores.empty:
        projected["matchup_score"] = 50
        return projected

    return projected.merge(
        matchup_scores,
        left_on=["opponent", "own_probable_pitcher_throws"],
        right_on=["team", "pitcher_throws"],
        how="left",
        suffixes=("", "_opponent"),
    )


def _load_team_matchup_scores() -> pd.DataFrame:
    pieces = []
    for pitcher_throws, filename in [
        ("L", "team_batting_lhp_05_06.parquet"),
        ("R", "team_batting_rhp_05_06.parquet"),
    ]:
        path = PROCESSED_DIR / filename
        if not path.exists():
            continue

        split = pd.read_parquet(path)
        split.columns = standardize_columns(split.columns)
        split["team"] = split["team"].astype("string").str.strip().str.upper()
        split["pitcher_throws"] = pitcher_throws
        split["opponent_woba"] = pd.to_numeric(split.get("woba"), errors="coerce")
        split["opponent_k_rate"] = pd.to_numeric(split.get("k_pct"), errors="coerce")
        split["matchup_score"] = 100 * (
            0.65 * (1 - split["opponent_woba"].rank(pct=True))
            + 0.35 * split["opponent_k_rate"].rank(pct=True)
        )
        pieces.append(
            split[
                [
                    "team",
                    "pitcher_throws",
                    "opponent_woba",
                    "opponent_k_rate",
                    "matchup_score",
                ]
            ]
        )

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _rank_projection_table(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["weekly_projection_index"] = pd.to_numeric(
        ranked["weekly_projection_index"],
        errors="coerce",
    ).fillna(0)
    ranked = ranked.sort_values(
        ["weekly_projection_index", "name"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["projection_rank"] = ranked.index + 1
    return ranked


def _combined_projection_view(df: pd.DataFrame) -> pd.DataFrame:
    common_columns = [
        "player_type",
        "player_key",
        "playerid",
        "mlbamid",
        "name",
        "team",
        "weekly_projection_index",
    ]
    view = df.copy()
    for column in common_columns:
        if column not in view.columns:
            view[column] = pd.NA
    return view[common_columns]


def _weighted_percentile_score(
    df: pd.DataFrame,
    metric_rules: Iterable[tuple[str, str, float]],
) -> pd.Series:
    weighted_scores = []
    total_weight = 0.0

    for column, direction, weight in metric_rules:
        values = pd.to_numeric(df[column], errors="coerce")
        score = values.rank(pct=True)
        if direction == "lower":
            score = 1 - score
        weighted_scores.append(score.fillna(0.5) * weight)
        total_weight += weight

    if total_weight == 0:
        return pd.Series(50, index=df.index)
    return (sum(weighted_scores) / total_weight * 100).round(1)


def _weighted_average(
    left_values: pd.Series,
    right_values: pd.Series,
    left_weights: pd.Series,
    right_weights: pd.Series,
    fallback: pd.Series,
) -> pd.Series:
    left = pd.to_numeric(left_values, errors="coerce").fillna(fallback)
    right = pd.to_numeric(right_values, errors="coerce").fillna(fallback)
    left_weight = pd.to_numeric(left_weights, errors="coerce").fillna(0)
    right_weight = pd.to_numeric(right_weights, errors="coerce").fillna(0)

    total_weight = left_weight + right_weight
    weighted = (left * left_weight + right * right_weight) / total_weight.mask(
        total_weight == 0,
    )
    fallback = pd.to_numeric(fallback, errors="coerce")
    return pd.to_numeric(weighted, errors="coerce").fillna(fallback)


def _first_available(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for column in columns:
        if column in df.columns:
            values = values.fillna(pd.to_numeric(df[column], errors="coerce"))
    return values


def _string_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    return df[column].astype("string").str.strip()


def _standardize_handedness(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().str[:1]


def _normalize_name_series(series: pd.Series) -> pd.Series:
    return series.astype("string").map(_normalize_name_value)


def _normalize_name_value(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_name.lower()).strip()


def _join_unique(values: pd.Series) -> str:
    unique_values = [str(value) for value in values.dropna().unique()]
    return ", ".join(unique_values)


def _empty_pitcher_projection_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "player_type",
            "player_key",
            "playerid",
            "mlbamid",
            "name",
            "team",
            "projected_starts",
            "opponents",
            "skill_score",
            "matchup_score",
            "weekly_projection_index",
            "projection_rank",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build weekly baseline projections from features and schedule."
    )
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECTIONS_DIR)
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create data/raw/schedules/probable_pitchers_template.csv and exit.",
    )
    args = parser.parse_args()

    if args.create_template:
        create_schedule_template()
        return

    run(schedule_path=args.schedule, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
