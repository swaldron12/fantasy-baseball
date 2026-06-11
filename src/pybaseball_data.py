"""Pull pybaseball data and build weekly historical tables.

This module keeps the model-training data local and explainable:
- FanGraphs leaderboards give season-level standard/advanced context.
- Baseball Savant Statcast events give rolling skill features and splits.
- Weekly range stats give category outcomes for the ML target.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.category_values import add_batter_category_values, add_pitcher_category_values
from src.clean import clean_dataset, standardize_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELING_DIR = PROJECT_ROOT / "data" / "modeling"
STATCAST_RAW_DIR = RAW_DIR / "statcast"
WEEKLY_OUTCOMES_DIR = RAW_DIR / "weekly_outcomes"

DEFAULT_START_DATE = date(2023, 3, 30)
DEFAULT_END_DATE = date.today()
ROLLING_WINDOWS = (1, 2, 4)

HIT_EVENTS = {"single", "double", "triple", "home_run"}
WALK_EVENTS = {"walk", "intent_walk"}
HBP_EVENTS = {"hit_by_pitch"}
SAC_EVENTS = {"sac_fly", "sac_bunt"}
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
BBE_EVENTS = {
    "single",
    "double",
    "triple",
    "home_run",
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "field_error",
    "fielders_choice",
    "double_play",
}
AT_BAT_EXCLUDED_EVENTS = WALK_EVENTS | HBP_EVENTS | SAC_EVENTS | {"catcher_interf", "truncated_pa"}
OUTS_BY_EVENT = {
    "field_out": 1,
    "force_out": 1,
    "fielders_choice_out": 1,
    "strikeout": 1,
    "sac_fly": 1,
    "sac_bunt": 1,
    "double_play": 2,
    "grounded_into_double_play": 2,
    "strikeout_double_play": 2,
    "sac_fly_double_play": 2,
}


def download_fangraphs_season_stats(
    season: int,
    output_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download season-to-date FanGraphs batting and pitching leaderboards."""
    _prepare_pybaseball_import()
    from pybaseball import batting_stats, pitching_stats

    batters = _call_fangraphs_leaderboard(batting_stats, season)
    pitchers = _call_fangraphs_leaderboard(pitching_stats, season)

    batters = clean_dataset(batters)
    pitchers = clean_dataset(pitchers)

    label = f"{season}_ytd"
    output_dir.mkdir(parents=True, exist_ok=True)
    batter_path = output_dir / f"batters_fangraphs_{label}.parquet"
    pitcher_path = output_dir / f"pitchers_fangraphs_{label}.parquet"
    batters.to_parquet(batter_path, index=False, engine="pyarrow")
    pitchers.to_parquet(pitcher_path, index=False, engine="pyarrow")

    print(f"FanGraphs batters: {batter_path}")
    print(f"FanGraphs pitchers: {pitcher_path}")
    return batters, pitchers


def download_weekly_outcomes(
    start_date: date = DEFAULT_START_DATE,
    end_date: date = DEFAULT_END_DATE,
    output_dir: Path = WEEKLY_OUTCOMES_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download weekly batting/pitching outcomes for category-value targets."""
    _prepare_pybaseball_import()
    from pybaseball import batting_stats_range, pitching_stats_range

    output_dir.mkdir(parents=True, exist_ok=True)
    batter_parts = []
    pitcher_parts = []

    for week_start, week_end in iter_weekly_windows(start_date, end_date):
        start_text = week_start.isoformat()
        end_text = week_end.isoformat()
        print(f"Pulling weekly outcomes: {start_text} to {end_text}")

        batters = _read_or_download_weekly_outcome(
            output_dir / f"batting_outcomes_{start_text}_{end_text}.parquet",
            batting_stats_range,
            start_text,
            end_text,
            label="batting",
        )
        pitchers = _read_or_download_weekly_outcome(
            output_dir / f"pitching_outcomes_{start_text}_{end_text}.parquet",
            pitching_stats_range,
            start_text,
            end_text,
            label="pitching",
        )

        if not batters.empty:
            batter_parts.append(batters)
        if not pitchers.empty:
            pitcher_parts.append(pitchers)

    batter_outcomes = (
        pd.concat(batter_parts, ignore_index=True) if batter_parts else pd.DataFrame()
    )
    pitcher_outcomes = (
        pd.concat(pitcher_parts, ignore_index=True) if pitcher_parts else pd.DataFrame()
    )

    batter_targets = (
        add_batter_category_values(batter_outcomes)
        if not batter_outcomes.empty
        else pd.DataFrame()
    )
    pitcher_targets = (
        add_pitcher_category_values(pitcher_outcomes)
        if not pitcher_outcomes.empty
        else pd.DataFrame()
    )

    MODELING_DIR.mkdir(parents=True, exist_ok=True)
    batter_targets.to_parquet(
        MODELING_DIR / "weekly_batter_category_targets.parquet",
        index=False,
        engine="pyarrow",
    )
    pitcher_targets.to_parquet(
        MODELING_DIR / "weekly_pitcher_category_targets.parquet",
        index=False,
        engine="pyarrow",
    )
    return batter_targets, pitcher_targets


def _read_or_download_weekly_outcome(
    output_path: Path,
    download_function: object,
    start_text: str,
    end_text: str,
    label: str,
) -> pd.DataFrame:
    week_start_label = _monday_week_start(pd.Timestamp(start_text))

    if output_path.exists():
        print(f"Using cached {label} outcomes: {output_path.name}")
        table = pd.read_parquet(output_path)
        if not table.empty:
            table["week_start"] = week_start_label
            table["week_end"] = end_text
        return table

    try:
        table = clean_dataset(download_function(start_text, end_text))
    except Exception as exc:
        print(
            f"Skipping {label} outcomes {start_text} to {end_text}: "
            f"{type(exc).__name__}: {exc}"
        )
        return pd.DataFrame()

    if table.empty:
        print(f"Skipping empty {label} outcomes: {start_text} to {end_text}")
        return table

    table["week_start"] = week_start_label
    table["week_end"] = end_text
    table.to_parquet(output_path, index=False, engine="pyarrow")
    return table


def build_cached_weekly_outcome_targets(
    outcomes_dir: Path = WEEKLY_OUTCOMES_DIR,
    modeling_dir: Path = MODELING_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build category-value targets from every cached weekly outcome parquet."""
    batter_outcomes = _load_cached_weekly_outcomes(outcomes_dir, "batting")
    pitcher_outcomes = _load_cached_weekly_outcomes(outcomes_dir, "pitching")

    batter_targets = (
        add_batter_category_values(batter_outcomes)
        if not batter_outcomes.empty
        else pd.DataFrame()
    )
    pitcher_targets = (
        add_pitcher_category_values(pitcher_outcomes)
        if not pitcher_outcomes.empty
        else pd.DataFrame()
    )

    modeling_dir.mkdir(parents=True, exist_ok=True)
    batter_targets.to_parquet(
        modeling_dir / "weekly_batter_category_targets.parquet",
        index=False,
        engine="pyarrow",
    )
    pitcher_targets.to_parquet(
        modeling_dir / "weekly_pitcher_category_targets.parquet",
        index=False,
        engine="pyarrow",
    )
    print(f"Cached batter targets rows: {len(batter_targets)}")
    print(f"Cached pitcher targets rows: {len(pitcher_targets)}")
    return batter_targets, pitcher_targets


def build_statcast_weekly_outcome_targets(
    statcast_dir: Path = STATCAST_RAW_DIR,
    modeling_dir: Path = MODELING_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build weekly category-value targets directly from cached Statcast events."""
    statcast = _load_statcast_files(statcast_dir)
    if statcast.empty:
        raise ValueError(f"No Statcast parquet files found in {statcast_dir}")

    statcast.columns = standardize_columns(statcast.columns)
    statcast["game_date"] = pd.to_datetime(statcast["game_date"], errors="coerce")
    statcast["week_start"] = statcast["game_date"].map(_monday_week_start)
    plate_appearances = statcast[statcast["events"].notna()].copy()
    if plate_appearances.empty:
        raise ValueError("Cached Statcast files contain no completed plate appearances.")

    batter_outcomes = _build_statcast_batter_outcomes(plate_appearances)
    pitcher_outcomes = _build_statcast_pitcher_outcomes(plate_appearances)
    batter_targets = add_batter_category_values(batter_outcomes)
    pitcher_targets = add_pitcher_category_values(pitcher_outcomes)

    modeling_dir.mkdir(parents=True, exist_ok=True)
    batter_targets.to_parquet(
        modeling_dir / "weekly_batter_category_targets.parquet",
        index=False,
        engine="pyarrow",
    )
    pitcher_targets.to_parquet(
        modeling_dir / "weekly_pitcher_category_targets.parquet",
        index=False,
        engine="pyarrow",
    )
    print(f"Statcast batter targets rows: {len(batter_targets)}")
    print(f"Statcast pitcher targets rows: {len(pitcher_targets)}")
    return batter_targets, pitcher_targets


def _build_statcast_batter_outcomes(pa: pd.DataFrame) -> pd.DataFrame:
    work = pa.copy()
    work["mlbid"] = pd.to_numeric(work["batter"], errors="coerce")
    work["tm"] = _batting_team(work)
    work["events"] = work["events"].astype("string")
    work["is_ab"] = ~work["events"].isin(AT_BAT_EXCLUDED_EVENTS)
    work["is_hit"] = work["events"].isin(HIT_EVENTS)
    work["is_single"] = work["events"].eq("single")
    work["is_double"] = work["events"].eq("double")
    work["is_triple"] = work["events"].eq("triple")
    work["is_hr"] = work["events"].eq("home_run")
    work["is_walk"] = work["events"].isin(WALK_EVENTS)
    work["is_hbp"] = work["events"].isin(HBP_EVENTS)
    work["is_sac_fly"] = work["events"].eq("sac_fly")
    work["is_sb"] = False
    work["rbi"] = _score_delta(work)

    grouped = (
        work.dropna(subset=["mlbid", "week_start"])
        .groupby(["mlbid", "week_start"], dropna=False)
        .agg(
            tm=("tm", _mode_text),
            g=("game_pk", "nunique"),
            pa=("events", "size"),
            ab=("is_ab", "sum"),
            h=("is_hit", "sum"),
            **{
                "1b": ("is_single", "sum"),
                "2b": ("is_double", "sum"),
                "3b": ("is_triple", "sum"),
            },
            hr=("is_hr", "sum"),
            rbi=("rbi", "sum"),
            bb=("is_walk", "sum"),
            hbp=("is_hbp", "sum"),
            sf=("is_sac_fly", "sum"),
            sb=("is_sb", "sum"),
        )
        .reset_index()
    )
    grouped["name"] = grouped["mlbid"].map(lambda value: f"Batter {int(value)}")
    grouped["r"] = pd.NA
    grouped["ba"] = _safe_divide(grouped["h"], grouped["ab"])
    return grouped


def _build_statcast_pitcher_outcomes(pa: pd.DataFrame) -> pd.DataFrame:
    work = pa.copy()
    work["mlbid"] = pd.to_numeric(work["pitcher"], errors="coerce")
    work["tm"] = _pitching_team(work)
    work["events"] = work["events"].astype("string")
    work["is_hit_allowed"] = work["events"].isin(HIT_EVENTS)
    work["is_walk"] = work["events"].isin(WALK_EVENTS)
    work["is_strikeout"] = work["events"].isin(STRIKEOUT_EVENTS)
    work["outs_on_play"] = work["events"].map(OUTS_BY_EVENT).fillna(0)
    work["runs_allowed"] = _score_delta(work)

    grouped = (
        work.dropna(subset=["mlbid", "week_start"])
        .groupby(["mlbid", "week_start"], dropna=False)
        .agg(
            name=("player_name", _mode_text),
            tm=("tm", _mode_text),
            g=("game_pk", "nunique"),
            ip_outs=("outs_on_play", "sum"),
            h=("is_hit_allowed", "sum"),
            er=("runs_allowed", "sum"),
            bb=("is_walk", "sum"),
            so=("is_strikeout", "sum"),
        )
        .reset_index()
    )
    grouped["ip"] = grouped["ip_outs"] / 3
    grouped["w"] = 0
    grouped["sv"] = 0
    grouped["era"] = _safe_divide(grouped["er"] * 9, grouped["ip"])
    grouped["whip"] = _safe_divide(grouped["h"] + grouped["bb"], grouped["ip"])
    return grouped


def _batting_team(df: pd.DataFrame) -> pd.Series:
    top = df.get("inning_topbot").astype("string").str.lower().eq("top")
    return df.get("away_team").where(top, df.get("home_team"))


def _pitching_team(df: pd.DataFrame) -> pd.Series:
    top = df.get("inning_topbot").astype("string").str.lower().eq("top")
    return df.get("home_team").where(top, df.get("away_team"))


def _score_delta(df: pd.DataFrame) -> pd.Series:
    before = pd.to_numeric(df.get("bat_score"), errors="coerce")
    after = pd.to_numeric(df.get("post_bat_score"), errors="coerce")
    return (after - before).clip(lower=0).fillna(0)


def _mode_text(values: pd.Series) -> str:
    cleaned = values.dropna().astype(str)
    if cleaned.empty:
        return ""
    return str(cleaned.mode().iloc[0])


def _load_cached_weekly_outcomes(
    outcomes_dir: Path,
    prefix: str,
) -> pd.DataFrame:
    parts = []
    pattern = re.compile(
        rf"{prefix}_outcomes_(?P<start>\d{{4}}-\d{{2}}-\d{{2}})_(?P<end>\d{{4}}-\d{{2}}-\d{{2}})\.parquet$"
    )
    for path in sorted(outcomes_dir.glob(f"{prefix}_outcomes_*.parquet")):
        match = pattern.search(path.name)
        if not match:
            continue
        table = pd.read_parquet(path)
        if table.empty:
            continue
        start_text = match.group("start")
        table["week_start"] = _monday_week_start(pd.Timestamp(start_text))
        table["week_end"] = match.group("end")
        parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def download_weekly_statcast(
    start_date: date = DEFAULT_START_DATE,
    end_date: date = DEFAULT_END_DATE,
    output_dir: Path = STATCAST_RAW_DIR,
) -> pd.DataFrame:
    """Download Statcast data in weekly chunks and save one file per week."""
    _prepare_pybaseball_import()
    from pybaseball import statcast

    output_dir.mkdir(parents=True, exist_ok=True)
    parts = []

    for week_start, week_end in iter_weekly_windows(start_date, end_date):
        start_text = week_start.isoformat()
        end_text = week_end.isoformat()
        output_path = output_dir / f"statcast_{start_text}_{end_text}.parquet"

        if output_path.exists():
            print(f"Using cached Statcast file: {output_path.name}")
            weekly = pd.read_parquet(output_path)
        else:
            print(f"Pulling Statcast: {start_text} to {end_text}")
            weekly = statcast(start_text, end_text, verbose=False, parallel=True)
            weekly.to_parquet(output_path, index=False, engine="pyarrow")

        parts.append(weekly)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_weekly_savant_features(
    statcast_dir: Path = STATCAST_RAW_DIR,
    output_dir: Path = FEATURES_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build rolling weekly features and handedness splits from Statcast files."""
    statcast = _load_statcast_files(statcast_dir)
    if statcast.empty:
        raise ValueError(f"No Statcast parquet files found in {statcast_dir}")

    statcast.columns = standardize_columns(statcast.columns)
    statcast["game_date"] = pd.to_datetime(statcast["game_date"], errors="coerce")
    statcast["week_start"] = statcast["game_date"].map(_monday_week_start)
    plate_appearances = statcast[statcast["events"].notna()].copy()

    batter_weekly = _build_batter_savant_weekly(plate_appearances)
    pitcher_weekly = _build_pitcher_savant_weekly(plate_appearances)

    output_dir.mkdir(parents=True, exist_ok=True)
    batter_path = output_dir / "weekly_batter_savant_features.parquet"
    pitcher_path = output_dir / "weekly_pitcher_savant_features.parquet"
    batter_weekly.to_parquet(batter_path, index=False, engine="pyarrow")
    pitcher_weekly.to_parquet(pitcher_path, index=False, engine="pyarrow")

    print(f"Weekly batter Savant features: {batter_path}")
    print(f"Weekly pitcher Savant features: {pitcher_path}")
    return batter_weekly, pitcher_weekly


def build_weekly_training_tables(
    features_dir: Path = FEATURES_DIR,
    modeling_dir: Path = MODELING_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join pre-week Savant features to weekly category-value targets."""
    batter_features = pd.read_parquet(
        features_dir / "weekly_batter_savant_features.parquet"
    )
    pitcher_features = pd.read_parquet(
        features_dir / "weekly_pitcher_savant_features.parquet"
    )
    batter_targets = pd.read_parquet(
        modeling_dir / "weekly_batter_category_targets.parquet"
    )
    pitcher_targets = pd.read_parquet(
        modeling_dir / "weekly_pitcher_category_targets.parquet"
    )

    batter_training = _join_features_to_targets(
        batter_features,
        batter_targets,
        target_id_column="mlbid",
    )
    pitcher_training = _join_features_to_targets(
        pitcher_features,
        pitcher_targets,
        target_id_column="mlbid",
    )

    modeling_dir.mkdir(parents=True, exist_ok=True)
    batter_output = modeling_dir / "weekly_batter_training.parquet"
    pitcher_output = modeling_dir / "weekly_pitcher_training.parquet"
    batter_training.to_parquet(batter_output, index=False, engine="pyarrow")
    pitcher_training.to_parquet(pitcher_output, index=False, engine="pyarrow")

    print(f"Weekly batter training table: {batter_output}")
    print(f"Weekly pitcher training table: {pitcher_output}")
    return batter_training, pitcher_training


def iter_weekly_windows(
    start_date: date,
    end_date: date,
) -> Iterable[tuple[date, date]]:
    """Yield Monday-start fantasy weeks between two dates."""
    current = start_date
    while current <= end_date:
        days_until_sunday = 6 - current.weekday()
        week_end = min(current + timedelta(days=days_until_sunday), end_date)
        yield current, week_end
        current = week_end + timedelta(days=1)


def _build_batter_savant_weekly(pa: pd.DataFrame) -> pd.DataFrame:
    work = pa.copy()
    work["is_hit"] = work["events"].isin(HIT_EVENTS)
    work["is_hr"] = work["events"].eq("home_run")
    work["is_walk"] = work["events"].isin(WALK_EVENTS)
    work["is_strikeout"] = work["events"].isin(STRIKEOUT_EVENTS)
    work["is_bbe"] = work["events"].isin(BBE_EVENTS)
    work["is_hardhit"] = pd.to_numeric(work.get("launch_speed"), errors="coerce").ge(95)
    work["estimated_woba"] = pd.to_numeric(
        work.get("estimated_woba_using_speedangle"),
        errors="coerce",
    )
    work["woba_value"] = pd.to_numeric(work.get("woba_value"), errors="coerce")

    base = (
        work.groupby(["batter", "week_start"], dropna=False)
        .agg(
            pa=("events", "size"),
            hits=("is_hit", "sum"),
            hr=("is_hr", "sum"),
            walks=("is_walk", "sum"),
            strikeouts=("is_strikeout", "sum"),
            bbe=("is_bbe", "sum"),
            hardhit=("is_hardhit", "sum"),
            xwoba=("estimated_woba", "mean"),
            woba=("woba_value", "mean"),
        )
        .reset_index()
        .rename(columns={"batter": "mlbamid"})
    )
    base["bb_rate"] = _safe_divide(base["walks"], base["pa"])
    base["k_rate"] = _safe_divide(base["strikeouts"], base["pa"])
    base["hardhit_rate"] = _safe_divide(base["hardhit"], base["bbe"])
    base["hr_rate"] = _safe_divide(base["hr"], base["pa"])

    split = _build_split_rates(
        work,
        id_column="batter",
        split_column="p_throws",
        prefixes={"L": "vs_lhp", "R": "vs_rhp"},
    ).rename(columns={"batter": "mlbamid"})

    features = base.merge(split, on=["mlbamid", "week_start"], how="left")
    rolling_columns = [
        "woba",
        "xwoba",
        "bb_rate",
        "k_rate",
        "hardhit_rate",
        "hr_rate",
        "woba_vs_lhp",
        "woba_vs_rhp",
        "k_rate_vs_lhp",
        "k_rate_vs_rhp",
    ]
    return _add_shifted_rolling_features(features, "mlbamid", rolling_columns)


def _build_pitcher_savant_weekly(pa: pd.DataFrame) -> pd.DataFrame:
    work = pa.copy()
    work["is_hit_allowed"] = work["events"].isin(HIT_EVENTS)
    work["is_hr_allowed"] = work["events"].eq("home_run")
    work["is_walk"] = work["events"].isin(WALK_EVENTS)
    work["is_strikeout"] = work["events"].isin(STRIKEOUT_EVENTS)
    work["is_bbe"] = work["events"].isin(BBE_EVENTS)
    work["is_hardhit"] = pd.to_numeric(work.get("launch_speed"), errors="coerce").ge(95)
    work["estimated_woba"] = pd.to_numeric(
        work.get("estimated_woba_using_speedangle"),
        errors="coerce",
    )
    work["woba_value"] = pd.to_numeric(work.get("woba_value"), errors="coerce")

    base = (
        work.groupby(["pitcher", "week_start"], dropna=False)
        .agg(
            batters_faced=("events", "size"),
            hits_allowed=("is_hit_allowed", "sum"),
            hr_allowed=("is_hr_allowed", "sum"),
            walks=("is_walk", "sum"),
            strikeouts=("is_strikeout", "sum"),
            bbe=("is_bbe", "sum"),
            hardhit_allowed=("is_hardhit", "sum"),
            xwoba_allowed=("estimated_woba", "mean"),
            woba_allowed=("woba_value", "mean"),
        )
        .reset_index()
        .rename(columns={"pitcher": "mlbamid"})
    )
    base["bb_rate"] = _safe_divide(base["walks"], base["batters_faced"])
    base["k_rate"] = _safe_divide(base["strikeouts"], base["batters_faced"])
    base["hardhit_rate_allowed"] = _safe_divide(
        base["hardhit_allowed"],
        base["bbe"],
    )
    base["hr_rate_allowed"] = _safe_divide(base["hr_allowed"], base["batters_faced"])

    split = _build_split_rates(
        work,
        id_column="pitcher",
        split_column="stand",
        prefixes={"L": "vs_lhb", "R": "vs_rhb"},
        woba_name="woba_allowed",
    ).rename(columns={"pitcher": "mlbamid"})

    features = base.merge(split, on=["mlbamid", "week_start"], how="left")
    rolling_columns = [
        "woba_allowed",
        "xwoba_allowed",
        "bb_rate",
        "k_rate",
        "hardhit_rate_allowed",
        "hr_rate_allowed",
        "woba_allowed_vs_lhb",
        "woba_allowed_vs_rhb",
        "k_rate_vs_lhb",
        "k_rate_vs_rhb",
    ]
    return _add_shifted_rolling_features(features, "mlbamid", rolling_columns)


def _build_split_rates(
    df: pd.DataFrame,
    id_column: str,
    split_column: str,
    prefixes: dict[str, str],
    woba_name: str = "woba",
) -> pd.DataFrame:
    work = df.copy()
    work["is_strikeout"] = work["events"].isin(STRIKEOUT_EVENTS)
    work["woba_value"] = pd.to_numeric(work.get("woba_value"), errors="coerce")

    grouped = (
        work.groupby([id_column, "week_start", split_column], dropna=False)
        .agg(
            pa=("events", "size"),
            strikeouts=("is_strikeout", "sum"),
            woba=("woba_value", "mean"),
        )
        .reset_index()
    )
    grouped["k_rate"] = _safe_divide(grouped["strikeouts"], grouped["pa"])

    pieces = []
    for split_value, prefix in prefixes.items():
        split = grouped[grouped[split_column].eq(split_value)].copy()
        split = split.rename(
            columns={
                "pa": f"pa_{prefix}",
                "woba": f"{woba_name}_{prefix}",
                "k_rate": f"k_rate_{prefix}",
            }
        )
        pieces.append(
            split[
                [
                    id_column,
                    "week_start",
                    f"pa_{prefix}",
                    f"{woba_name}_{prefix}",
                    f"k_rate_{prefix}",
                ]
            ]
        )

    if not pieces:
        return pd.DataFrame(columns=[id_column, "week_start"])

    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on=[id_column, "week_start"], how="outer")
    return result


def _add_shifted_rolling_features(
    df: pd.DataFrame,
    id_column: str,
    value_columns: list[str],
) -> pd.DataFrame:
    work = df.sort_values([id_column, "week_start"]).reset_index(drop=True).copy()

    for column in value_columns:
        if column not in work.columns:
            continue
        work[column] = pd.to_numeric(work[column], errors="coerce")
        for window in ROLLING_WINDOWS:
            feature_name = f"{column}_prev_{window}w"
            work[feature_name] = work.groupby(id_column, dropna=False)[column].transform(
                lambda series: series.shift(1).rolling(window, min_periods=1).mean()
            )

    return work


def _load_statcast_files(statcast_dir: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(path) for path in sorted(statcast_dir.glob("*.parquet"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _join_features_to_targets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    target_id_column: str,
) -> pd.DataFrame:
    feature_table = features.copy()
    target_table = targets.copy()

    feature_table["mlbamid"] = pd.to_numeric(feature_table["mlbamid"], errors="coerce")
    target_table["mlbamid"] = pd.to_numeric(
        target_table[target_id_column],
        errors="coerce",
    )
    feature_table["week_start"] = pd.to_datetime(
        feature_table["week_start"],
        errors="coerce",
    ).dt.date.astype("string")
    target_table["week_start"] = pd.to_datetime(
        target_table["week_start"],
        errors="coerce",
    ).dt.date.astype("string")

    feature_columns = [
        column
        for column in feature_table.columns
        if column.endswith("_prev_1w")
        or column.endswith("_prev_2w")
        or column.endswith("_prev_4w")
    ]
    id_columns = ["mlbamid", "week_start"]
    selected_features = feature_table[id_columns + feature_columns]

    target_columns = [
        column
        for column in [
            "mlbamid",
            "week_start",
            "name",
            "tm",
            "player_type",
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
        ]
        if column in target_table.columns
    ]
    selected_targets = target_table[target_columns]

    training = selected_targets.merge(
        selected_features,
        on=["mlbamid", "week_start"],
        how="inner",
    )
    training = training.rename(columns={"tm": "team"})
    return training


def _call_fangraphs_leaderboard(function: object, season: int) -> pd.DataFrame:
    try:
        return function(start_season=season, end_season=season, qual=0)
    except TypeError:
        return function(season, qual=0)


def _prepare_pybaseball_import() -> None:
    matplotlib_cache = PROJECT_ROOT / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")
    return numerator / denominator.mask(denominator == 0)


def _monday_week_start(value: pd.Timestamp) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    return (value - pd.Timedelta(days=value.weekday())).date().isoformat()


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull pybaseball data and build weekly historical tables."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    parser.add_argument("--season", type=int, default=DEFAULT_END_DATE.year)
    parser.add_argument("--fangraphs-season", action="store_true")
    parser.add_argument("--weekly-outcomes", action="store_true")
    parser.add_argument("--statcast", action="store_true")
    parser.add_argument("--savant-features", action="store_true")
    parser.add_argument("--cached-outcome-targets", action="store_true")
    parser.add_argument("--statcast-outcome-targets", action="store_true")
    parser.add_argument("--training-tables", action="store_true")
    args = parser.parse_args()

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)

    if args.fangraphs_season:
        download_fangraphs_season_stats(args.season)
    if args.weekly_outcomes:
        download_weekly_outcomes(start_date, end_date)
    if args.statcast:
        download_weekly_statcast(start_date, end_date)
    if args.savant_features:
        build_weekly_savant_features()
    if args.cached_outcome_targets:
        build_cached_weekly_outcome_targets()
    if args.statcast_outcome_targets:
        build_statcast_weekly_outcome_targets()
    if args.training_tables:
        build_weekly_training_tables()

    if not any(
        [
            args.fangraphs_season,
            args.weekly_outcomes,
            args.statcast,
            args.savant_features,
            args.cached_outcome_targets,
            args.statcast_outcome_targets,
            args.training_tables,
        ]
    ):
        parser.print_help()


if __name__ == "__main__":
    main()
