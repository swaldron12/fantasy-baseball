"""Hybrid weekly stat projections from recent form, ZiPS, and matchup context."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from src.clean import standardize_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
STATCAST_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "statcast"

BATTER_STAT_COLUMNS = [
    "ab",
    "h",
    "1b",
    "2b",
    "3b",
    "hr",
    "r",
    "rbi",
    "bb",
    "hbp",
    "sf",
    "sb",
    "total_bases",
    "avg",
    "ops",
]

PITCHER_STAT_COLUMNS = [
    "ip",
    "w",
    "sv",
    "hld",
    "k",
    "er",
    "hits_allowed",
    "bb_allowed",
    "baserunners",
    "era",
    "whip",
]


def add_hybrid_stat_projections(
    batter_projections: pd.DataFrame,
    pitcher_projections: pd.DataFrame,
    schedule: pd.DataFrame,
    processed_dir: Path = PROCESSED_DIR,
    features_dir: Path = FEATURES_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add weekly category stat estimates to projection tables."""
    batters = add_batter_hybrid_stats(
        batter_projections,
        schedule,
        processed_dir=processed_dir,
        features_dir=features_dir,
    )
    pitchers = add_pitcher_hybrid_stats(
        pitcher_projections,
        processed_dir=processed_dir,
        features_dir=features_dir,
    )
    return batters, pitchers


def add_batter_hybrid_stats(
    batter_projections: pd.DataFrame,
    schedule: pd.DataFrame,
    processed_dir: Path = PROCESSED_DIR,
    features_dir: Path = FEATURES_DIR,
) -> pd.DataFrame:
    """Project hitter counting/rate stats from ZiPS, YTD form, and matchups."""
    if batter_projections.empty:
        return batter_projections

    work = batter_projections.copy()
    work["mlbamid"] = pd.to_numeric(work["mlbamid"], errors="coerce")
    work["team"] = work["team"].astype("string").str.upper().str.strip()

    work = _merge_by_mlbamid(work, _read_parquet(processed_dir / "zips_batters_ros.parquet"), "zips")
    work = _merge_by_mlbamid(work, _read_ytd_batters(processed_dir), "ytd")
    work = _merge_latest_weekly_features(
        work,
        _read_parquet(features_dir / "weekly_batter_savant_features.parquet"),
        "recent",
    )
    work = work.merge(_batter_handedness_lookup(), on="mlbamid", how="left")
    work = work.merge(
        _batter_team_context(schedule, processed_dir, features_dir),
        on="team",
        how="left",
    )
    matchup_rows = _schedule_with_pitcher_difficulty(
        schedule,
        processed_dir,
        features_dir,
    )

    games = _num(work.get("projected_games")).fillna(0)
    ab_per_game = _blend(
        work,
        [
            (_ratio(work, "zips_ab", "zips_g"), 0.65),
            (_ratio(work, "ytd_ab", "ytd_g"), 0.35),
        ],
    ).fillna(0)
    ab = games * ab_per_game

    handedness_factor = _ratio(
        work,
        "matchup_woba_signal",
        "overall_woba_signal",
    ).fillna(1).clip(0.88, 1.12)
    pitcher_factor = _num(work.get("opposing_pitcher_factor")).fillna(1).clip(0.88, 1.12)
    team_factor = _num(work.get("team_offense_factor")).fillna(1).clip(0.92, 1.08)
    recent_quality = _recent_batter_quality_factor(work)
    contact_factor = _soften_factor(handedness_factor * pitcher_factor * recent_quality, 0.25)
    run_factor = _soften_factor(team_factor * pitcher_factor, 0.60)

    h_rate = _blend(
        work,
        [
            (_ratio(work, "zips_h", "zips_ab"), 0.80),
            (_ratio(work, "ytd_h", "ytd_ab"), 0.20),
        ],
    ).fillna(0) * contact_factor
    h_rate = _cap_rate_delta(
        h_rate,
        _ratio(work, "zips_h", "zips_ab"),
        max_up=0.045,
        max_down=0.060,
    )
    hr_rate = _blend(
        work,
        [
            (_ratio(work, "zips_hr", "zips_ab"), 0.55),
            (_ratio(work, "ytd_hr", "ytd_ab"), 0.30),
            (_recent_rate_feature(work, "hr_rate"), 0.15),
        ],
    ).fillna(0) * _soften_factor(pitcher_factor * recent_quality, 0.70)
    double_rate = _blend(
        work,
        [
            (_ratio(work, "zips_2b", "zips_ab"), 0.65),
            (_ratio(work, "ytd_2b", "ytd_ab"), 0.35),
        ],
    ).fillna(0) * _soften_factor(pitcher_factor * recent_quality, 0.45)
    triple_rate = _blend(
        work,
        [
            (_ratio(work, "zips_3b", "zips_ab"), 0.65),
            (_ratio(work, "ytd_3b", "ytd_ab"), 0.35),
        ],
    ).fillna(0) * _soften_factor(team_factor * recent_quality, 0.25)
    r_per_game = _blend(
        work,
        [
            (_ratio(work, "zips_r", "zips_g"), 0.60),
            (_ratio(work, "ytd_r", "ytd_g"), 0.40),
        ],
    ).fillna(0) * run_factor
    rbi_per_game = _blend(
        work,
        [
            (_ratio(work, "zips_rbi", "zips_g"), 0.60),
            (_ratio(work, "ytd_rbi", "ytd_g"), 0.40),
        ],
    ).fillna(0) * run_factor
    sb_per_game = _blend(
        work,
        [
            (_ratio(work, "zips_sb", "zips_g"), 0.65),
            (_ratio(work, "ytd_sb", "ytd_g"), 0.35),
        ],
    ).fillna(0) * _soften_factor(team_factor, 0.25)
    bb_rate = _blend(
        work,
        [
            (_ratio(work, "zips_bb", "zips_ab"), 0.60),
            (_ratio(work, "ytd_bb", "ytd_ab"), 0.25),
            (_recent_rate_feature(work, "bb_rate"), 0.15),
        ],
    ).fillna(0)
    hbp_rate = _blend(
        work,
        [
            (_ratio(work, "zips_hbp", "zips_ab"), 0.70),
            (_ratio(work, "ytd_hbp", "ytd_ab"), 0.30),
        ],
    ).fillna(0)
    sf_rate = _blend(
        work,
        [
            (_ratio(work, "zips_sf", "zips_ab"), 0.70),
            (_ratio(work, "ytd_sf", "ytd_ab"), 0.30),
        ],
    ).fillna(0)
    work["ab"] = ab.clip(lower=0)
    work["h"] = (work["ab"] * h_rate).clip(lower=0)
    work["hr"] = (work["ab"] * hr_rate).clip(lower=0)
    work["h"] = work[["h", "ab"]].min(axis=1)
    work["hr"] = work[["hr", "h"]].min(axis=1)
    raw_doubles = (work["ab"] * double_rate).clip(lower=0)
    raw_triples = (work["ab"] * triple_rate).clip(lower=0)
    non_hr_hits = (work["h"] - work["hr"]).clip(lower=0)
    raw_non_hr_extra_bases = raw_doubles + raw_triples
    extra_base_scale = (
        non_hr_hits / raw_non_hr_extra_bases.mask(raw_non_hr_extra_bases == 0)
    ).clip(upper=1).fillna(1)
    work["2b"] = raw_doubles * extra_base_scale
    work["3b"] = raw_triples * extra_base_scale
    work["1b"] = (non_hr_hits - work["2b"] - work["3b"]).clip(lower=0)
    work["h"] = work["1b"] + work["2b"] + work["3b"] + work["hr"]
    work["r"] = (games * r_per_game).clip(lower=0)
    work["rbi"] = (games * rbi_per_game).clip(lower=0)
    work["sb"] = (games * sb_per_game).clip(lower=0)
    work["bb"] = (work["ab"] * bb_rate).clip(lower=0)
    work["hbp"] = (work["ab"] * hbp_rate).clip(lower=0)
    work["sf"] = (work["ab"] * sf_rate).clip(lower=0)
    work["total_bases"] = (
        work["1b"] + 2 * work["2b"] + 3 * work["3b"] + 4 * work["hr"]
    )

    obp_denominator = work["ab"] + work["bb"] + work["hbp"] + work["sf"]
    obp = (work["h"] + work["bb"] + work["hbp"]) / obp_denominator.mask(
        obp_denominator == 0
    )
    slg = work["total_bases"] / work["ab"].mask(work["ab"] == 0)
    work["avg"] = work["h"] / work["ab"].mask(work["ab"] == 0)
    work["ops"] = obp.fillna(0) + slg.fillna(0)
    work["matchup_edge_pct"] = (
        (_ratio(work, "matchup_woba_signal", "overall_woba_signal") - 1) * 100
    ).round(1)
    work = _add_batter_specific_matchup_evidence(work, matchup_rows)
    work["matchup_context"] = work.apply(_format_batter_matchup_context, axis=1)
    work["projection_stat_source"] = "hybrid_zips_ytd_recent_matchup"
    work = _round_projection_columns(
        work,
        [column for column in BATTER_STAT_COLUMNS if column not in {"avg", "ops"}],
    )
    return _derive_batter_rate_stats(work)


def add_pitcher_hybrid_stats(
    pitcher_projections: pd.DataFrame,
    processed_dir: Path = PROCESSED_DIR,
    features_dir: Path = FEATURES_DIR,
) -> pd.DataFrame:
    """Project pitcher counting/rate stats from ZiPS, YTD form, and opponent strength."""
    if pitcher_projections.empty:
        return pitcher_projections

    work = pitcher_projections.copy()
    work["mlbamid"] = pd.to_numeric(work["mlbamid"], errors="coerce")
    work = _merge_by_mlbamid(work, _read_parquet(processed_dir / "zips_pitchers_ros.parquet"), "zips")
    work = _merge_by_mlbamid(work, _read_ytd_pitchers(processed_dir), "ytd")
    work = _merge_latest_weekly_features(
        work,
        _read_parquet(features_dir / "weekly_pitcher_savant_features.parquet"),
        "recent",
    )

    starts = _num(work.get("projected_starts")).fillna(0)
    matchup_score = _num(work.get("matchup_score")).fillna(50).clip(0, 100)
    k_factor = (1 + (matchup_score - 50) / 450).clip(0.85, 1.15)
    run_factor = (1 - (matchup_score - 50) / 500).clip(0.85, 1.15)
    ip_factor = (1 + (matchup_score - 50) / 1200).clip(0.94, 1.06)

    ip_per_start = _blend(
        work,
        [
            (_ratio(work, "zips_ip", "zips_gs"), 0.70),
            (_ratio(work, "ytd_ip", "ytd_gs"), 0.30),
        ],
    ).fillna(_ratio(work, "zips_ip", "zips_g").fillna(1))
    ip = (starts * ip_per_start * ip_factor).clip(lower=0)

    k_per_ip = _blend(
        work,
        [
            (_ratio(work, "zips_so", "zips_ip"), 0.58),
            (_ratio(work, "ytd_so", "ytd_ip"), 0.27),
            (_num(work.get("recent_k_rate")) * 4.25, 0.15),
        ],
    ).fillna(0)
    er_per_ip = _blend(
        work,
        [
            (_ratio(work, "zips_er", "zips_ip"), 0.65),
            (_ratio(work, "ytd_er", "ytd_ip"), 0.35),
        ],
    ).fillna(0)
    h_per_ip = _blend(
        work,
        [
            (_ratio(work, "zips_h", "zips_ip"), 0.60),
            (_ratio(work, "ytd_h", "ytd_ip"), 0.25),
            (_ratio(work, "recent_hits_allowed", "recent_batters_faced") * 4.25, 0.15),
        ],
    ).fillna(0)
    bb_per_ip = _blend(
        work,
        [
            (_ratio(work, "zips_bb", "zips_ip"), 0.60),
            (_ratio(work, "ytd_bb", "ytd_ip"), 0.25),
            (_num(work.get("recent_bb_rate")) * 4.25, 0.15),
        ],
    ).fillna(0)
    w_per_start = _blend(
        work,
        [
            (_ratio(work, "zips_w", "zips_gs"), 0.70),
            (_ratio(work, "ytd_w", "ytd_gs"), 0.30),
        ],
    ).fillna(0)

    work["ip"] = ip
    work["k"] = (ip * k_per_ip * k_factor).clip(lower=0)
    work["er"] = (ip * er_per_ip * run_factor).clip(lower=0)
    work["hits_allowed"] = (ip * h_per_ip * run_factor).clip(lower=0)
    work["bb_allowed"] = (ip * bb_per_ip * run_factor).clip(lower=0)
    work["baserunners"] = work["hits_allowed"] + work["bb_allowed"]
    work["w"] = (starts * w_per_start * k_factor).clip(lower=0)
    work["sv"] = 0.0
    work["hld"] = 0.0
    work["era"] = work["er"] * 9 / work["ip"].mask(work["ip"] == 0)
    work["whip"] = work["baserunners"] / work["ip"].mask(work["ip"] == 0)
    work["projection_stat_source"] = "hybrid_zips_ytd_recent_matchup"
    return _round_projection_columns(work, PITCHER_STAT_COLUMNS)


def _batter_team_context(
    schedule: pd.DataFrame,
    processed_dir: Path,
    features_dir: Path,
) -> pd.DataFrame:
    context_columns = [
        "team",
        "opposing_pitcher_factor",
        "team_offense_factor",
        "opposing_pitcher_difficulty",
        "opposing_pitcher_summary",
        "favorable_opposing_pitchers",
        "tough_opposing_pitchers",
    ]
    if schedule.empty:
        return pd.DataFrame(columns=context_columns)

    work = _schedule_with_pitcher_difficulty(schedule, processed_dir, features_dir)

    rows = []
    for team, team_rows in work.groupby("team", dropna=False):
        rows.append(
            {
                "team": team,
                "opposing_pitcher_difficulty": float(
                    team_rows["pitcher_difficulty"].mean()
                ),
                "opposing_pitcher_summary": _join_pitcher_matchups(team_rows),
                "favorable_opposing_pitchers": _join_pitcher_matchups(
                    team_rows[team_rows["pitcher_difficulty"].le(45)],
                    limit=3,
                ),
                "tough_opposing_pitchers": _join_pitcher_matchups(
                    team_rows[team_rows["pitcher_difficulty"].ge(58)],
                    limit=3,
                ),
            }
        )
    context = pd.DataFrame(rows)
    if context.empty:
        return pd.DataFrame(columns=context_columns)
    context["opposing_pitcher_difficulty"] = pd.to_numeric(
        context["opposing_pitcher_difficulty"],
        errors="coerce",
    ).fillna(50)
    context["opposing_pitcher_factor"] = (
        1 + (50 - context["opposing_pitcher_difficulty"]) / 500
    )
    context["opposing_pitcher_factor"] = context["opposing_pitcher_factor"].clip(0.88, 1.12)
    offense = _team_offense_factor(processed_dir)
    context = context.merge(offense, on="team", how="left")
    context["team_offense_factor"] = _num(context.get("team_offense_factor")).fillna(1)
    for column in context_columns:
        if column not in context.columns:
            context[column] = ""
    return context[context_columns]


def _schedule_with_pitcher_difficulty(
    schedule: pd.DataFrame,
    processed_dir: Path,
    features_dir: Path,
) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame()

    work = schedule.copy()
    work["team"] = work["team"].astype("string").str.upper().str.strip()
    work["opposing_probable_pitcher_name"] = (
        _text_series(work, "opposing_probable_pitcher_name")
        .astype("string")
        .str.strip()
    )
    work["opposing_probable_pitcher_throws"] = (
        _text_series(work, "opposing_probable_pitcher_throws")
        .astype("string")
        .str.strip()
        .str.upper()
    )
    work["opposing_probable_pitcher_id"] = pd.to_numeric(
        work.get("opposing_probable_pitcher_id"),
        errors="coerce",
    )
    difficulty = _opposing_pitcher_difficulty(processed_dir, features_dir)
    work = work.merge(
        difficulty,
        left_on="opposing_probable_pitcher_id",
        right_on="mlbamid",
        how="left",
    )
    evidence_columns = [
        "zips_era",
        "zips_whip",
        "zips_k_pct",
        "recent_woba_allowed_prev_4w",
        "recent_xwoba_allowed_prev_4w",
        "recent_woba_allowed_vs_lhb_prev_4w",
        "recent_woba_allowed_vs_rhb_prev_4w",
        "recent_k_rate_prev_4w",
        "recent_bb_rate_prev_4w",
    ]
    if not difficulty.empty and "name_key" in difficulty.columns:
        difficulty_by_name = difficulty.dropna(subset=["name_key"]).drop_duplicates(
            "name_key",
        )[["name_key", "pitcher_difficulty", *evidence_columns]]
        work["opposing_pitcher_name_key"] = work[
            "opposing_probable_pitcher_name"
        ].map(_normalize_name)
        work = work.merge(
            difficulty_by_name.rename(
                columns={
                    column: f"name_{column}"
                    for column in ["pitcher_difficulty", *evidence_columns]
                }
            ),
            left_on="opposing_pitcher_name_key",
            right_on="name_key",
            how="left",
        )
        work["pitcher_difficulty"] = _num(work.get("pitcher_difficulty")).fillna(
            _num(work.get("name_pitcher_difficulty"))
        )
        for column in evidence_columns:
            if column in work.columns and f"name_{column}" in work.columns:
                work[column] = _num(work.get(column)).fillna(
                    _num(work.get(f"name_{column}"))
                )
    work["pitcher_difficulty"] = _num(work.get("pitcher_difficulty")).fillna(50)
    return work


def _opposing_pitcher_difficulty(
    processed_dir: Path,
    features_dir: Path,
) -> pd.DataFrame:
    output_columns = [
        "mlbamid",
        "name_key",
        "pitcher_difficulty",
        "zips_era",
        "zips_whip",
        "zips_k_pct",
        "recent_woba_allowed_prev_4w",
        "recent_xwoba_allowed_prev_4w",
        "recent_woba_allowed_vs_lhb_prev_4w",
        "recent_woba_allowed_vs_rhb_prev_4w",
        "recent_k_rate_prev_4w",
        "recent_bb_rate_prev_4w",
    ]
    zips = _read_parquet(processed_dir / "zips_pitchers_ros.parquet")
    if zips.empty:
        return pd.DataFrame(columns=output_columns)
    zips = zips.copy()
    zips["mlbamid"] = pd.to_numeric(zips["mlbamid"], errors="coerce")
    zips["name_key"] = zips.get("name", pd.Series("", index=zips.index)).map(
        _normalize_name
    )
    era_score = 1 - _num(zips.get("era")).rank(pct=True)
    whip_score = 1 - _num(zips.get("whip")).rank(pct=True)
    k_score = _num(zips.get("k_pct")).rank(pct=True)
    zips["pitcher_difficulty"] = (
        (0.42 * era_score.fillna(0.5) + 0.33 * k_score.fillna(0.5) + 0.25 * whip_score.fillna(0.5))
        * 100
    )
    zips["zips_era"] = _num(zips.get("era"))
    zips["zips_whip"] = _num(zips.get("whip"))
    zips["zips_k_pct"] = _num(zips.get("k_pct"))

    recent = _read_parquet(features_dir / "weekly_pitcher_savant_features.parquet")
    if not recent.empty and "mlbamid" in recent.columns and "week_start" in recent.columns:
        recent = recent.copy()
        recent["mlbamid"] = pd.to_numeric(recent["mlbamid"], errors="coerce")
        recent["week_start"] = pd.to_datetime(recent["week_start"], errors="coerce")
        recent = recent.sort_values(["mlbamid", "week_start"]).drop_duplicates(
            "mlbamid",
            keep="last",
        )
        recent_columns = [
            "mlbamid",
            "woba_allowed_prev_4w",
            "xwoba_allowed_prev_4w",
            "woba_allowed_vs_lhb_prev_4w",
            "woba_allowed_vs_rhb_prev_4w",
            "k_rate_prev_4w",
            "bb_rate_prev_4w",
        ]
        recent_columns = [column for column in recent_columns if column in recent.columns]
        recent = recent[recent_columns].rename(
            columns={
                column: f"recent_{column}"
                for column in recent_columns
                if column != "mlbamid"
            }
        )
        zips = zips.merge(recent, on="mlbamid", how="left")

    for column in output_columns:
        if column not in zips.columns:
            zips[column] = pd.NA
    return zips[output_columns]


def _batter_handedness_lookup(
    statcast_dir: Path | None = None,
) -> pd.DataFrame:
    statcast_dir = statcast_dir or STATCAST_RAW_DIR
    output_columns = ["mlbamid", "batter_bats", "stand_vs_lhp", "stand_vs_rhp"]
    if not statcast_dir.exists():
        return pd.DataFrame(columns=output_columns)

    parts = []
    for path in sorted(statcast_dir.glob("*.parquet")):
        table = pd.read_parquet(path, columns=["batter", "stand", "p_throws"])
        if table.empty:
            continue
        table.columns = standardize_columns(table.columns)
        parts.append(table)
    if not parts:
        return pd.DataFrame(columns=output_columns)

    work = pd.concat(parts, ignore_index=True)
    work["mlbamid"] = pd.to_numeric(work.get("batter"), errors="coerce")
    work["stand"] = _text_series(work, "stand").astype("string").str.upper().str.strip()
    work["p_throws"] = (
        _text_series(work, "p_throws").astype("string").str.upper().str.strip()
    )
    work = work[
        work["mlbamid"].notna()
        & work["stand"].isin(["L", "R"])
        & work["p_throws"].isin(["L", "R"])
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=output_columns)

    overall_counts = (
        work.groupby(["mlbamid", "stand"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    for stand in ["L", "R"]:
        if stand not in overall_counts.columns:
            overall_counts[stand] = 0
    total = overall_counts["L"] + overall_counts["R"]
    handedness = overall_counts.reset_index()
    handedness["batter_bats"] = "R"
    handedness.loc[overall_counts["L"].to_numpy() > overall_counts["R"].to_numpy(), "batter_bats"] = "L"
    switch_mask = (
        (overall_counts["L"] / total.mask(total == 0)).fillna(0).ge(0.15)
        & (overall_counts["R"] / total.mask(total == 0)).fillna(0).ge(0.15)
    ).to_numpy()
    handedness.loc[switch_mask, "batter_bats"] = "S"

    split_counts = (
        work.groupby(["mlbamid", "p_throws", "stand"], dropna=False)
        .size()
        .reset_index(name="pa")
        .sort_values(["mlbamid", "p_throws", "pa"], ascending=[True, True, False])
        .drop_duplicates(["mlbamid", "p_throws"])
    )
    split = split_counts.pivot(
        index="mlbamid",
        columns="p_throws",
        values="stand",
    ).reset_index()
    split = split.rename(columns={"L": "stand_vs_lhp", "R": "stand_vs_rhp"})
    handedness = handedness[["mlbamid", "batter_bats"]].merge(
        split,
        on="mlbamid",
        how="left",
    )
    for column in output_columns:
        if column not in handedness.columns:
            handedness[column] = pd.NA
    return handedness[output_columns]


def _add_batter_specific_matchup_evidence(
    batters: pd.DataFrame,
    matchup_rows: pd.DataFrame,
) -> pd.DataFrame:
    enriched = batters.copy()
    output_columns = [
        "batter_side_note",
        "relevant_opposing_pitcher_summary",
        "favorable_batter_matchups",
        "tough_batter_matchups",
    ]
    for column in output_columns:
        enriched[column] = ""
    if enriched.empty or matchup_rows.empty:
        return enriched

    matchup_by_team = {
        team: rows.copy()
        for team, rows in matchup_rows.groupby("team", dropna=False)
    }
    side_notes = []
    summaries = []
    favorable_notes = []
    tough_notes = []
    for _, batter in enriched.iterrows():
        team_rows = matchup_by_team.get(str(batter.get("team", "")).upper())
        side_notes.append(_batter_side_note(batter))
        if team_rows is None or team_rows.empty:
            summaries.append("")
            favorable_notes.append("")
            tough_notes.append("")
            continue
        matchup = _batter_specific_matchup_strings(batter, team_rows)
        summaries.append(matchup["summary"])
        favorable_notes.append(matchup["favorable"])
        tough_notes.append(matchup["tough"])

    enriched["batter_side_note"] = side_notes
    enriched["relevant_opposing_pitcher_summary"] = summaries
    enriched["favorable_batter_matchups"] = favorable_notes
    enriched["tough_batter_matchups"] = tough_notes
    return enriched


def _batter_specific_matchup_strings(
    batter: pd.Series,
    team_rows: pd.DataFrame,
) -> dict[str, str]:
    work = team_rows.copy()
    if "game_date" in work.columns:
        work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
        work = work.sort_values("game_date", na_position="last")

    summary_parts = []
    favorable_parts = []
    tough_parts = []
    seen = set()
    for _, pitcher in work.iterrows():
        name = _clean_text(pitcher.get("opposing_probable_pitcher_name"))
        if not name or name.lower() == "nan":
            continue
        throws = _clean_text(pitcher.get("opposing_probable_pitcher_throws")).upper()
        key = (name.lower(), throws)
        if key in seen:
            continue
        seen.add(key)

        batter_side = _expected_batter_side(batter, throws)
        label = _format_batter_specific_pitcher_evidence(pitcher, batter_side)
        if label:
            summary_parts.append(label)
        relevant_woba = _relevant_pitcher_woba_allowed(pitcher, batter_side)
        difficulty = _clean_number(pitcher.get("pitcher_difficulty"))
        if _is_favorable_batter_matchup(relevant_woba, difficulty) and label:
            favorable_parts.append(label)
        elif _is_tough_batter_matchup(relevant_woba, difficulty) and label:
            tough_parts.append(label)

    return {
        "summary": ", ".join(summary_parts[:7]),
        "favorable": ", ".join(favorable_parts[:3]),
        "tough": ", ".join(tough_parts[:3]),
    }


def _expected_batter_side(batter: pd.Series, pitcher_throws: str) -> str:
    pitcher_throws = _clean_text(pitcher_throws).upper()
    if pitcher_throws == "L":
        side = _clean_text(batter.get("stand_vs_lhp")).upper()
    elif pitcher_throws == "R":
        side = _clean_text(batter.get("stand_vs_rhp")).upper()
    else:
        side = ""
    if side in {"L", "R"}:
        return side

    bats = _clean_text(batter.get("batter_bats")).upper()
    if bats in {"L", "R"}:
        return bats
    if bats == "S" and pitcher_throws == "L":
        return "R"
    if bats == "S" and pitcher_throws == "R":
        return "L"
    return ""


def _batter_side_note(batter: pd.Series) -> str:
    bats = _clean_text(batter.get("batter_bats")).upper()
    stand_vs_lhp = _clean_text(batter.get("stand_vs_lhp")).upper()
    stand_vs_rhp = _clean_text(batter.get("stand_vs_rhp")).upper()
    if bats == "S":
        if stand_vs_lhp in {"L", "R"} and stand_vs_rhp in {"L", "R"}:
            return f"bats S; projected {stand_vs_lhp} vs LHP and {stand_vs_rhp} vs RHP"
        return "bats S; projected side depends on pitcher handedness"
    if bats in {"L", "R"}:
        return f"bats {bats}"
    return "batter handedness unknown; relevant split not applied"


def _format_batter_specific_pitcher_evidence(
    pitcher: pd.Series,
    batter_side: str,
) -> str:
    name = _clean_text(pitcher.get("opposing_probable_pitcher_name"))
    if not name:
        return ""
    throws = _clean_text(pitcher.get("opposing_probable_pitcher_throws")).upper()
    label = f"{name} ({throws})" if throws in {"L", "R"} else name
    pieces = []

    relevant_woba = _relevant_pitcher_woba_allowed(pitcher, batter_side)
    if relevant_woba is not None and batter_side in {"L", "R"}:
        pieces.append(
            f"relevant split {_format_rate_stat(relevant_woba)} wOBA allowed vs {batter_side}HB"
        )
    else:
        woba_allowed = _clean_number(pitcher.get("recent_woba_allowed_prev_4w"))
        if woba_allowed is not None:
            pieces.append(f"4w {_format_rate_stat(woba_allowed)} wOBA allowed")

    zips_era = _clean_number(pitcher.get("zips_era"))
    zips_whip = _clean_number(pitcher.get("zips_whip"))
    if zips_era is not None and zips_whip is not None:
        pieces.append(f"ZiPS {zips_era:.2f} ERA/{zips_whip:.2f} WHIP")

    if not pieces and batter_side not in {"L", "R"}:
        pieces.append("batter handedness unknown")
    return f"{label}; " + "; ".join(pieces[:3]) if pieces else label


def _relevant_pitcher_woba_allowed(
    pitcher: pd.Series,
    batter_side: str,
) -> float | None:
    if batter_side == "L":
        return _clean_number(pitcher.get("recent_woba_allowed_vs_lhb_prev_4w"))
    if batter_side == "R":
        return _clean_number(pitcher.get("recent_woba_allowed_vs_rhb_prev_4w"))
    return None


def _is_favorable_batter_matchup(
    relevant_woba: float | None,
    difficulty: float | None,
) -> bool:
    if relevant_woba is not None:
        return relevant_woba >= 0.330
    return difficulty is not None and difficulty <= 45


def _is_tough_batter_matchup(
    relevant_woba: float | None,
    difficulty: float | None,
) -> bool:
    if relevant_woba is not None:
        return relevant_woba <= 0.285
    return difficulty is not None and difficulty >= 58


def _format_batter_matchup_context(row: pd.Series) -> str:
    pieces = []
    projected_games = _clean_count(row.get("projected_games"))
    games_vs_lhp = _clean_count(row.get("games_vs_lhp"))
    games_vs_rhp = _clean_count(row.get("games_vs_rhp"))
    if projected_games:
        pieces.append(
            f"{projected_games} games: {games_vs_lhp} vs LHP, {games_vs_rhp} vs RHP"
        )
    side_note = _clean_text(row.get("batter_side_note"))
    if side_note:
        pieces.append(side_note)

    edge = row.get("matchup_edge_pct")
    if pd.notna(edge):
        edge = float(edge)
        if edge >= 4:
            pieces.append(f"favorable handedness/schedule signal ({edge:+.1f}%)")
        elif edge <= -4:
            pieces.append(f"tough handedness/schedule signal ({edge:+.1f}%)")

    favorable = _clean_text(row.get("favorable_batter_matchups"))
    tough = _clean_text(row.get("tough_batter_matchups"))
    summary = _clean_text(row.get("relevant_opposing_pitcher_summary"))
    if favorable:
        pieces.append(f"favorable relevant matchups: {favorable}")
    elif tough:
        pieces.append(f"tough relevant matchups: {tough}")
    elif summary:
        pieces.append(f"relevant matchup evidence: {summary}")
    return "; ".join(pieces)


def _join_pitcher_matchups(rows: pd.DataFrame, limit: int = 7) -> str:
    if rows.empty:
        return ""
    work = rows.copy()
    if "game_date" in work.columns:
        work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
        work = work.sort_values("game_date", na_position="last")
    parts = []
    seen = set()
    for _, row in work.iterrows():
        name = _clean_text(row.get("opposing_probable_pitcher_name"))
        if not name or name.lower() == "nan":
            continue
        throws = _clean_text(row.get("opposing_probable_pitcher_throws")).upper()
        key = (name.lower(), throws)
        if key in seen:
            continue
        seen.add(key)
        label = name
        if throws in {"L", "R"}:
            label = f"{label} ({throws})"
        evidence = _format_pitcher_matchup_evidence(row)
        if evidence:
            label = f"{label}; {evidence}"
        parts.append(label)
        if len(parts) >= limit:
            break
    return ", ".join(parts)


def _format_pitcher_matchup_evidence(row: pd.Series) -> str:
    pieces = []
    zips_era = _clean_number(row.get("zips_era"))
    zips_whip = _clean_number(row.get("zips_whip"))
    if zips_era is not None and zips_whip is not None:
        pieces.append(f"ZiPS {zips_era:.2f} ERA/{zips_whip:.2f} WHIP")

    woba_allowed = _clean_number(row.get("recent_woba_allowed_prev_4w"))
    if woba_allowed is not None:
        pieces.append(f"4w {_format_rate_stat(woba_allowed)} wOBA allowed")

    woba_lhb = _clean_number(row.get("recent_woba_allowed_vs_lhb_prev_4w"))
    woba_rhb = _clean_number(row.get("recent_woba_allowed_vs_rhb_prev_4w"))
    split_parts = []
    if woba_lhb is not None:
        split_parts.append(f"{_format_rate_stat(woba_lhb)} vs LHB")
    if woba_rhb is not None:
        split_parts.append(f"{_format_rate_stat(woba_rhb)} vs RHB")
    if split_parts:
        pieces.append("4w " + "/".join(split_parts))

    k_rate = _clean_number(row.get("recent_k_rate_prev_4w"))
    if k_rate is not None:
        pieces.append(f"4w {k_rate * 100:.1f}% K")
    return "; ".join(pieces[:3])


def _team_offense_factor(processed_dir: Path) -> pd.DataFrame:
    pieces = []
    for filename in ["team_batting_lhp_05_06.parquet", "team_batting_rhp_05_06.parquet"]:
        table = _read_parquet(processed_dir / filename)
        if table.empty or "team" not in table.columns:
            continue
        table = table.copy()
        table["team"] = table["team"].astype("string").str.upper().str.strip()
        table["team_woba"] = _num(table.get("woba"))
        pieces.append(table[["team", "team_woba"]])
    if not pieces:
        return pd.DataFrame(columns=["team", "team_offense_factor"])
    combined = pd.concat(pieces, ignore_index=True)
    offense = combined.groupby("team", dropna=False)["team_woba"].mean().reset_index()
    offense["team_offense_factor"] = 0.95 + offense["team_woba"].rank(pct=True).fillna(0.5) * 0.10
    return offense[["team", "team_offense_factor"]]


def _read_ytd_batters(processed_dir: Path) -> pd.DataFrame:
    return _read_first_existing(
        [
            processed_dir / "batters_fangraphs_2026_ytd.parquet",
            processed_dir / "batters_standard_05_06.parquet",
        ]
    )


def _read_ytd_pitchers(processed_dir: Path) -> pd.DataFrame:
    return _read_first_existing(
        [
            processed_dir / "pitchers_fangraphs_2026_ytd.parquet",
            processed_dir / "pitchers_standard_05_06.parquet",
        ]
    )


def _read_first_existing(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        table = _read_parquet(path)
        if not table.empty:
            return table
    return pd.DataFrame()


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_parquet(path)
    table.columns = standardize_columns(table.columns)
    return table


def _merge_by_mlbamid(base: pd.DataFrame, table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if table.empty or "mlbamid" not in table.columns:
        return base
    selected = table.copy()
    selected["mlbamid"] = pd.to_numeric(selected["mlbamid"], errors="coerce")
    selected = selected.dropna(subset=["mlbamid"]).drop_duplicates("mlbamid")
    rename_map = {
        column: f"{prefix}_{column}"
        for column in selected.columns
        if column != "mlbamid"
    }
    selected = selected.rename(columns=rename_map)
    return base.merge(selected, on="mlbamid", how="left")


def _merge_latest_weekly_features(base: pd.DataFrame, table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if table.empty or "mlbamid" not in table.columns or "week_start" not in table.columns:
        return base
    selected = table.copy()
    selected["mlbamid"] = pd.to_numeric(selected["mlbamid"], errors="coerce")
    selected["week_start"] = pd.to_datetime(selected["week_start"], errors="coerce")
    selected = selected.sort_values(["mlbamid", "week_start"]).drop_duplicates(
        "mlbamid",
        keep="last",
    )
    rename_map = {
        column: f"{prefix}_{column}"
        for column in selected.columns
        if column != "mlbamid"
    }
    selected = selected.rename(columns=rename_map)
    return base.merge(selected, on="mlbamid", how="left")


def _blend(df: pd.DataFrame, weighted_values: list[tuple[pd.Series, float]]) -> pd.Series:
    numerator = pd.Series(0.0, index=df.index)
    denominator = pd.Series(0.0, index=df.index)
    for values, weight in weighted_values:
        series = pd.to_numeric(values, errors="coerce")
        valid = series.notna()
        numerator = numerator.add(series.fillna(0) * weight, fill_value=0)
        denominator = denominator.add(valid.astype(float) * weight, fill_value=0)
    return numerator / denominator.mask(denominator == 0)


def _ratio(df: pd.DataFrame, numerator_column: str, denominator_column: str | None) -> pd.Series:
    numerator = _num(df.get(numerator_column))
    if denominator_column is None:
        return numerator
    denominator = _num(df.get(denominator_column))
    return numerator / denominator.mask(denominator == 0)


def _num(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce")
    return pd.Series(pd.NA, dtype="Float64")


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series("", index=df.index)


def _normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _clean_count(value: object) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0
    return int(round(float(numeric)))


def _clean_number(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def _format_rate_stat(value: float) -> str:
    formatted = f"{value:.3f}"
    return formatted[1:] if formatted.startswith("0") else formatted


def _recent_batter_quality_factor(df: pd.DataFrame) -> pd.Series:
    recent_quality = _coalesce_numeric_columns(
        df,
        [
            "recent_xwoba_prev_4w",
            "recent_xwoba_prev_2w",
            "recent_xwoba_prev_1w",
            "recent_woba_prev_4w",
            "recent_woba_prev_2w",
            "recent_woba_prev_1w",
            "recent_xwoba",
            "recent_woba",
        ],
    )
    baseline = _num(df.get("zips_woba")).fillna(_num(df.get("overall_woba_signal")))
    factor = recent_quality / baseline.mask(baseline == 0)
    return factor.fillna(1).clip(0.88, 1.12)


def _recent_rate_feature(df: pd.DataFrame, rate_name: str) -> pd.Series:
    return _coalesce_numeric_columns(
        df,
        [
            f"recent_{rate_name}_prev_4w",
            f"recent_{rate_name}_prev_2w",
            f"recent_{rate_name}_prev_1w",
            f"recent_{rate_name}",
        ],
    )


def _coalesce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for column in columns:
        values = values.fillna(_num(df.get(column)))
    return values


def _soften_factor(factor: pd.Series, strength: float) -> pd.Series:
    return (1 + (pd.to_numeric(factor, errors="coerce").fillna(1) - 1) * strength).clip(0.75, 1.25)


def _cap_rate_delta(
    rate: pd.Series,
    baseline: pd.Series,
    max_up: float,
    max_down: float,
) -> pd.Series:
    numeric_rate = pd.to_numeric(rate, errors="coerce")
    numeric_baseline = pd.to_numeric(baseline, errors="coerce")
    lower = numeric_baseline - max_down
    upper = numeric_baseline + max_up
    capped = numeric_rate.clip(lower=lower, upper=upper)
    return capped.where(numeric_baseline.notna(), numeric_rate)


def _round_projection_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rounded = df.copy()
    for column in columns:
        if column in rounded.columns:
            rounded[column] = pd.to_numeric(rounded[column], errors="coerce").round(3)
    return rounded


def _derive_batter_rate_stats(df: pd.DataFrame) -> pd.DataFrame:
    derived = df.copy()
    singles = _num(derived.get("1b")).fillna(0)
    doubles = _num(derived.get("2b")).fillna(0)
    triples = _num(derived.get("3b")).fillna(0)
    homers = _num(derived.get("hr")).fillna(0)
    ab = _num(derived.get("ab")).fillna(0)
    h = _num(derived.get("h")).fillna(0)
    bb = _num(derived.get("bb")).fillna(0)
    hbp = _num(derived.get("hbp")).fillna(0)
    sf = _num(derived.get("sf")).fillna(0)

    derived["total_bases"] = (
        singles + 2 * doubles + 3 * triples + 4 * homers
    ).round(3)
    obp_denominator = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_denominator.mask(obp_denominator == 0)
    slg = derived["total_bases"] / ab.mask(ab == 0)
    derived["avg"] = (h / ab.mask(ab == 0)).fillna(0).round(3)
    derived["ops"] = (obp.fillna(0) + slg.fillna(0)).round(3)
    return derived
