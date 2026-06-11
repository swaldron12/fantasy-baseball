"""Build player-level feature tables from cleaned Fangraphs data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.clean import standardize_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

ID_COLUMNS = ["player_key", "playerid", "mlbamid", "name", "team"]

BATTER_FEATURE_COLUMNS = [
    "woba",
    "xwoba",
    "barrel_rate",
    "hardhit_rate",
    "k_rate",
    "bb_rate",
    "woba_vs_lhp",
    "woba_vs_rhp",
    "k_rate_vs_lhp",
    "k_rate_vs_rhp",
    "iso",
    "pa",
    "zips_woba",
    "zips_hr_rate",
]

PITCHER_FEATURE_COLUMNS = [
    "k_rate",
    "bb_rate",
    "xfip",
    "siera",
    "xwoba_allowed",
    "barrel_rate_allowed",
    "hardhit_rate_allowed",
    "woba_allowed_vs_lhb",
    "woba_allowed_vs_rhb",
    "k_rate_vs_lhb",
    "k_rate_vs_rhb",
    "ip",
    "zips_era",
    "zips_k_rate",
]


def build_batter_features(
    processed_dir: Path = PROCESSED_DIR,
    features_dir: Path = FEATURES_DIR,
) -> pd.DataFrame:
    """Create one player-level feature row per batter."""
    datasets = {
        "standard": _read_dataset(processed_dir / "batters_standard_05_06.parquet"),
        "advanced": _read_dataset(processed_dir / "batters_advanced_05_06.parquet"),
        "statcast": _read_dataset(processed_dir / "batters_statcast_05_06.parquet"),
        "splits_lhp": _read_dataset(processed_dir / "batters_splits_lhp_05_06.parquet"),
        "splits_rhp": _read_dataset(processed_dir / "batters_splits_rhp_05_06.parquet"),
        "zips": _read_dataset(processed_dir / "zips_batters_ros.parquet", required=False),
    }

    features = _build_player_base(datasets.values())
    features = _left_join_features(features, datasets["standard"], {"pa": "pa"})
    features = _left_join_features(
        features,
        datasets["advanced"],
        {
            "woba": "woba",
            "k_pct": "k_rate",
            "bb_pct": "bb_rate",
            "iso": "iso",
        },
    )
    features = _left_join_features(
        features,
        datasets["statcast"],
        {
            "xwoba": "xwoba",
            "barrel_pct": "barrel_rate",
            "hardhit_pct": "hardhit_rate",
        },
    )
    features = _left_join_features(
        features,
        datasets["splits_lhp"],
        {
            "woba": "woba_vs_lhp",
            "k_pct": "k_rate_vs_lhp",
        },
    )
    features = _left_join_features(
        features,
        datasets["splits_rhp"],
        {
            "woba": "woba_vs_rhp",
            "k_pct": "k_rate_vs_rhp",
        },
    )

    zips = datasets["zips"].copy()
    if not zips.empty:
        zips["zips_hr_rate"] = _safe_divide(zips.get("hr"), zips.get("pa"))
    features = _left_join_features(
        features,
        zips,
        {
            "woba": "zips_woba",
            "zips_hr_rate": "zips_hr_rate",
        },
    )

    features = _finalize_features(features, BATTER_FEATURE_COLUMNS)
    output_path = features_dir / "batters_features.parquet"
    _save_feature_table(features, output_path, BATTER_FEATURE_COLUMNS)
    return features


def build_pitcher_features(
    processed_dir: Path = PROCESSED_DIR,
    features_dir: Path = FEATURES_DIR,
) -> pd.DataFrame:
    """Create one player-level feature row per pitcher."""
    datasets = {
        "standard": _read_dataset(processed_dir / "pitchers_standard_05_06.parquet"),
        "advanced": _read_dataset(processed_dir / "pitchers_advanced_05_06.parquet"),
        "statcast": _read_dataset(processed_dir / "pitchers_statcast_05_06.parquet"),
        "splits_lhb": _read_dataset(processed_dir / "pitchers_split_lhh_05_06.parquet"),
        "splits_rhb": _read_dataset(processed_dir / "pitchers_split_rhh_05_06.parquet"),
        "zips": _read_dataset(processed_dir / "zips_pitchers_ros.parquet", required=False),
    }

    features = _build_player_base(datasets.values())
    features = _left_join_features(features, datasets["standard"], {"ip": "ip"})
    features = _left_join_features(
        features,
        datasets["advanced"],
        {
            "k_pct": "k_rate",
            "bb_pct": "bb_rate",
            "xfip": "xfip",
            "siera": "siera",
        },
    )
    features = _left_join_features(
        features,
        datasets["statcast"],
        {
            "xwoba": "xwoba_allowed",
            "barrel_pct": "barrel_rate_allowed",
            "hardhit_pct": "hardhit_rate_allowed",
        },
    )
    features = _left_join_features(
        features,
        datasets["splits_lhb"],
        {
            "woba": "woba_allowed_vs_lhb",
            "k_pct": "k_rate_vs_lhb",
        },
    )
    features = _left_join_features(
        features,
        datasets["splits_rhb"],
        {
            "woba": "woba_allowed_vs_rhb",
            "k_pct": "k_rate_vs_rhb",
        },
    )
    features = _left_join_features(
        features,
        datasets["zips"],
        {
            "era": "zips_era",
            "k_pct": "zips_k_rate",
        },
    )

    features = _finalize_features(features, PITCHER_FEATURE_COLUMNS)
    output_path = features_dir / "pitchers_features.parquet"
    _save_feature_table(features, output_path, PITCHER_FEATURE_COLUMNS)
    return features


def run() -> None:
    """Build all current player-level feature tables."""
    build_batter_features()
    build_pitcher_features()


def _read_dataset(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            print(f"WARNING: missing processed dataset: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df.columns = standardize_columns(df.columns)
    return _add_player_key(df)


def _add_player_key(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()

    playerid = _string_column(keyed, "playerid")
    name = _string_column(keyed, "name")
    playerid_is_valid = playerid.notna() & (playerid.str.strip() != "")

    keyed["player_key"] = playerid.where(playerid_is_valid, name.str.lower())
    return keyed


def _build_player_base(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    parts = []

    for df in datasets:
        if df.empty or "player_key" not in df.columns:
            continue

        part = df.copy()
        for column in ID_COLUMNS:
            if column not in part.columns:
                part[column] = pd.NA
        parts.append(part[ID_COLUMNS])

    if not parts:
        return pd.DataFrame(columns=ID_COLUMNS)

    base = pd.concat(parts, ignore_index=True)
    base = base.dropna(subset=["player_key"])
    base = base.drop_duplicates(subset=["player_key"], keep="first")
    return base.sort_values(["team", "name"], na_position="last").reset_index(drop=True)


def _left_join_features(
    base: pd.DataFrame,
    source: pd.DataFrame,
    column_map: dict[str, str],
) -> pd.DataFrame:
    if source.empty or "player_key" not in source.columns:
        return base

    selected = source[["player_key"]].copy()
    for source_column, feature_column in column_map.items():
        selected[feature_column] = (
            source[source_column] if source_column in source.columns else pd.NA
        )

    selected = selected.drop_duplicates(subset=["player_key"], keep="first")
    return base.merge(selected, on="player_key", how="left")


def _finalize_features(
    features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    finalized = features.copy()

    for column in ID_COLUMNS + feature_columns:
        if column not in finalized.columns:
            finalized[column] = pd.NA

    return finalized[ID_COLUMNS + feature_columns]


def _save_feature_table(
    features: pd.DataFrame,
    output_path: Path,
    feature_columns: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False, engine="pyarrow")

    missing_counts = features[feature_columns].isna().sum()
    print(f"{output_path.name}")
    print(f"  players: {len(features)}")
    print(f"  features: {len(feature_columns)}")
    print(f"  missing values: {int(missing_counts.sum())}")
    print(f"  output path: {output_path}")

    nonzero_missing = missing_counts[missing_counts > 0]
    if not nonzero_missing.empty:
        print("  missing by feature:")
        for column, count in nonzero_missing.items():
            print(f"    {column}: {int(count)}")


def _safe_divide(numerator: pd.Series | None, denominator: pd.Series | None) -> pd.Series:
    if numerator is None or denominator is None:
        return pd.Series(pd.NA)

    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator / denominator.replace(0, pd.NA)


def _string_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    return df[column].astype("string").str.strip()


if __name__ == "__main__":
    run()
