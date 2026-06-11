"""Match a user roster to weekly projections and summarize expected value."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

from src.availability import apply_availability_flags


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "config"
SCHEDULE_PATH = PROJECT_ROOT / "data" / "raw" / "schedules" / "probable_pitchers_week.csv"

BATTER_PROJECTIONS_PATH = PROJECTIONS_DIR / "weekly_batter_projections.csv"
PITCHER_PROJECTIONS_PATH = PROJECTIONS_DIR / "weekly_pitcher_projections.csv"
ZIPS_BATTERS_PATH = PROCESSED_DIR / "zips_batters_ros.parquet"
ZIPS_PITCHERS_PATH = PROCESSED_DIR / "zips_pitchers_ros.parquet"
ELIGIBILITY_PATH = CONFIG_DIR / "player_eligibility.csv"
DEFAULT_BEDROCK_MODEL_ID = "amazon.nova-lite-v1:0"

HITTER_SLOTS = ["C", "1B", "2B", "3B", "SS", "OF1", "OF2", "OF3", "UTIL"]
HITTER_SLOT_LABELS = {
    "C": "C",
    "1B": "1B",
    "2B": "2B",
    "3B": "3B",
    "SS": "SS",
    "OF1": "OF",
    "OF2": "OF",
    "OF3": "OF",
    "UTIL": "UTIL",
}
PITCHER_SLOTS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
BENCH_SLOTS = ["BN1", "BN2", "BN3"]
ACTIVE_SLOTS = HITTER_SLOTS + PITCHER_SLOTS
POSITION_ORDER = ["C", "1B", "2B", "3B", "SS", "OF"]
POSITION_ALIASES = {
    "C": ["C"],
    "1B": ["1B"],
    "2B": ["2B"],
    "3B": ["3B"],
    "SS": ["SS"],
    "OF": ["OF"],
    "LF": ["OF"],
    "CF": ["OF"],
    "RF": ["OF"],
    "MI": ["2B", "SS"],
    "CI": ["1B", "3B"],
    "UTIL": [],
    "DH": [],
}

TYPE_WORDS = {
    "batter": "batter",
    "bat": "batter",
    "hitter": "batter",
    "hit": "batter",
    "pitcher": "pitcher",
    "pitch": "pitcher",
    "sp": "pitcher",
    "rp": "pitcher",
    "p": "pitcher",
}

POSITION_WORDS = {
    "c",
    "1b",
    "2b",
    "3b",
    "ss",
    "of",
    "dh",
    "mi",
    "ci",
    "util",
    "bench",
}

THREE_DECIMAL_OUTPUT_COLUMNS = {
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
}


@dataclass
class RosterEntry:
    """One parsed roster input row."""

    raw_text: str
    player_name: str
    requested_type: str | None = None


@dataclass
class LineupSlot:
    """One slot in the front-end lineup builder."""

    slot_id: str
    slot_label: str
    slot_type: str
    player_name: str
    eligible_positions: list[str]


def analyze_roster(
    roster_text: str,
    projections_dir: Path = PROJECTIONS_DIR,
    schedule_path: Path = SCHEDULE_PATH,
) -> dict[str, Any]:
    """Return matched roster projections, unmatched rows, and roster summary."""
    projections = load_projection_pool(projections_dir)
    entries = parse_roster_text(roster_text)

    matched_rows = []
    unmatched_rows = []

    for entry in entries:
        match = match_roster_entry(entry, projections)
        if match is None:
            unmatched_rows.append(
                {
                    "input": entry.raw_text,
                    "player_name": entry.player_name,
                    "suggestions": suggest_player_names(entry.player_name, projections),
                }
            )
            continue

        row = match.to_dict()
        row["input"] = entry.raw_text
        matched_rows.append(_clean_output_row(row))

    summary = summarize_roster(matched_rows, unmatched_rows, projections)
    metadata = projection_metadata(schedule_path, projections)
    return {
        "metadata": metadata,
        "summary": summary,
        "players": matched_rows,
        "unmatched": unmatched_rows,
    }


def analyze_lineup_slots(
    slots: list[dict[str, Any]],
    projections_dir: Path = PROJECTIONS_DIR,
    schedule_path: Path = SCHEDULE_PATH,
    include_analysis: bool = True,
) -> dict[str, Any]:
    """Analyze a structured lineup with active and bench slots."""
    projections = load_projection_pool(projections_dir)
    eligibility_lookup = load_player_eligibility()
    parsed_slots = [
        LineupSlot(
            slot_id=str(slot.get("slot_id", "")).strip(),
            slot_label=str(slot.get("slot_label", "")).strip(),
            slot_type=str(slot.get("slot_type", "")).strip(),
            player_name=str(slot.get("player_name", "")).strip(),
            eligible_positions=parse_position_list(slot.get("eligible_positions", "")),
        )
        for slot in slots
        if str(slot.get("player_name", "")).strip()
    ]

    matched_rows = []
    unmatched_rows = []
    seen_player_keys = set()

    for slot in parsed_slots:
        requested_type = _slot_requested_type(slot)
        entry = RosterEntry(
            raw_text=slot.player_name,
            player_name=slot.player_name,
            requested_type=requested_type,
        )
        match = match_roster_entry(entry, projections)
        if match is None:
            unmatched_rows.append(
                {
                    "slot_id": slot.slot_id,
                    "slot_label": slot.slot_label,
                    "input": slot.player_name,
                    "suggestions": suggest_player_names(slot.player_name, projections),
                }
            )
            continue

        row = match.to_dict()
        unique_key = f"{row.get('player_type')}::{row.get('player_key')}::{row.get('name_key')}"
        if unique_key in seen_player_keys:
            continue
        seen_player_keys.add(unique_key)

        row["slot_id"] = slot.slot_id
        row["slot_label"] = slot.slot_label
        row["slot_type"] = slot.slot_type
        row["input"] = slot.player_name
        row["lineup_status"] = "bench" if slot.slot_type == "bench" else "active"
        row["eligible_positions"] = resolve_player_eligibility(
            row,
            slot,
            eligibility_lookup,
        )
        row["eligible_positions_label"] = format_positions(row["eligible_positions"])
        matched_rows.append(_clean_output_row(row))

    active_rows = [row for row in matched_rows if row.get("lineup_status") == "active"]
    current_summary = summarize_lineup(active_rows, unmatched_rows)
    optimized = optimize_lineup(matched_rows)
    metadata = projection_metadata(schedule_path, projections)
    analysis = (
        generate_lineup_analysis(current_summary, optimized, matched_rows)
        if include_analysis
        else None
    )

    result = {
        "metadata": metadata,
        "current": current_summary,
        "optimized": optimized,
        "players": matched_rows,
        "unmatched": unmatched_rows,
    }
    if analysis is not None:
        result["analysis"] = analysis
    return result


def load_projection_pool(
    projections_dir: Path = PROJECTIONS_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Load batter and pitcher projection tables into one searchable pool."""
    batters = _read_projection_table(projections_dir / "weekly_batter_projections.csv")
    pitchers = _read_projection_table(projections_dir / "weekly_pitcher_projections.csv")

    batters = _add_batter_category_projections(batters, processed_dir)
    pitchers = _add_pitcher_category_projections(pitchers, processed_dir)

    projection_pool = pd.concat([batters, pitchers], ignore_index=True, sort=False).copy()
    if projection_pool.empty:
        return projection_pool

    projection_pool["name_key"] = projection_pool["name"].map(normalize_player_name)
    projection_pool = apply_availability_flags(projection_pool)
    projection_pool["expected_fantasy_value"] = (
        projection_pool.groupby("player_type", group_keys=False)[
            "weekly_projection_index"
        ].transform(_z_score)
    ).round(2)
    projection_pool["value_percentile"] = (
        projection_pool.groupby("player_type", group_keys=False)[
            "weekly_projection_index"
        ].rank(pct=True)
        * 100
    ).round(0)
    return projection_pool


def search_players(
    query: str,
    player_type: str | None = None,
    limit: int = 12,
    projections_dir: Path = PROJECTIONS_DIR,
) -> list[dict[str, Any]]:
    """Return player-search suggestions for the front end."""
    projections = load_projection_pool(projections_dir)
    if projections.empty:
        return []
    eligibility_lookup = load_player_eligibility()

    query_key = normalize_player_name(query)
    candidates = projections
    if player_type in {"batter", "pitcher"}:
        candidates = candidates[candidates["player_type"].eq(player_type)]

    if query_key:
        contains = candidates[candidates["name_key"].str.contains(query_key, na=False)]
        if contains.empty:
            keys = difflib.get_close_matches(
                query_key,
                candidates["name_key"].dropna().unique().tolist(),
                n=limit,
                cutoff=0.45,
            )
            contains = candidates[candidates["name_key"].isin(keys)]
        candidates = contains

    candidates = candidates.sort_values(
        ["expected_fantasy_value", "weekly_projection_index", "name"],
        ascending=[False, False, True],
    ).head(limit)
    return [
        _clean_output_row(
            {
                "name": row["name"],
                "team": row.get("team"),
                "player_type": row.get("player_type"),
                "expected_fantasy_value": row.get("expected_fantasy_value"),
                "weekly_projection_index": row.get("weekly_projection_index"),
                "eligible_positions": format_positions(
                    _configured_eligibility_for_row(row, eligibility_lookup)
                ),
                "is_available": row.get("is_available"),
                "availability_status": row.get("availability_status"),
                "availability_note": row.get("availability_note"),
                "label": f"{row['name']} ({row.get('team', '')}, {row.get('player_type', '')})",
            }
        )
        for _, row in candidates.iterrows()
    ]


def load_player_eligibility(
    path: Path = ELIGIBILITY_PATH,
) -> dict[str, list[str]]:
    """Load optional manual fantasy position eligibility."""
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    df.columns = [str(column).strip().lower() for column in df.columns]
    lookup = {}
    name_column = "player_name" if "player_name" in df.columns else "name"
    if name_column not in df.columns or "eligible_positions" not in df.columns:
        return lookup

    for _, row in df.iterrows():
        name_key = normalize_player_name(row.get(name_column))
        positions = parse_position_list(row.get("eligible_positions"))
        if name_key and positions:
            lookup[name_key] = positions
    return lookup


def parse_position_list(value: object) -> list[str]:
    """Normalize fantasy position text like '2B, OF' into canonical positions."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_tokens = [str(item).upper() for item in value]
    elif pd.isna(value):
        return []
    else:
        raw_tokens = re.split(r"[,/|;\s]+", str(value).upper())

    positions = []
    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue
        positions.extend(POSITION_ALIASES.get(token, []))
    return _dedupe_positions(positions)


def resolve_player_eligibility(
    row: dict[str, Any],
    slot: LineupSlot,
    eligibility_lookup: dict[str, list[str]],
) -> list[str]:
    """Resolve a hitter's known positions from UI, config, then active slot."""
    if row.get("player_type") == "pitcher":
        return ["P"]

    explicit = slot.eligible_positions
    if explicit:
        return explicit

    configured = _configured_eligibility_for_row(row, eligibility_lookup)
    if configured:
        return configured

    slot_position = _slot_base_position(slot.slot_id)
    if slot.slot_type == "hitter" and slot_position:
        return [slot_position]

    return []


def format_positions(positions: object) -> str:
    """Format canonical positions for display."""
    parsed = parse_position_list(positions) if isinstance(positions, str) else positions
    if not parsed:
        return ""
    return ", ".join(_dedupe_positions(parsed))


def parse_roster_text(roster_text: str) -> list[RosterEntry]:
    """Parse one-player-per-line or comma-separated roster text."""
    parts = []
    for line in roster_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line and _looks_like_comma_list(line):
            parts.extend(part.strip() for part in line.split(",") if part.strip())
        else:
            parts.append(line)

    entries = []
    for part in parts:
        entry = _parse_roster_part(part)
        if entry.player_name:
            entries.append(entry)
    return entries


def match_roster_entry(
    entry: RosterEntry,
    projections: pd.DataFrame,
) -> pd.Series | None:
    """Find the best projection row for one roster entry."""
    if projections.empty:
        return None

    candidates = projections
    if entry.requested_type:
        candidates = candidates[candidates["player_type"].eq(entry.requested_type)]

    query_key = normalize_player_name(entry.player_name)
    exact_matches = candidates[candidates["name_key"].eq(query_key)]
    if not exact_matches.empty:
        return _choose_best_match(exact_matches)

    close_keys = difflib.get_close_matches(
        query_key,
        candidates["name_key"].dropna().unique().tolist(),
        n=1,
        cutoff=0.82,
    )
    if not close_keys:
        return None

    close_matches = candidates[candidates["name_key"].eq(close_keys[0])]
    return _choose_best_match(close_matches)


def suggest_player_names(
    player_name: str,
    projections: pd.DataFrame,
    limit: int = 3,
) -> list[str]:
    """Suggest close projection names for an unmatched roster row."""
    query_key = normalize_player_name(player_name)
    keys = projections["name_key"].dropna().unique().tolist()
    close_keys = difflib.get_close_matches(query_key, keys, n=limit, cutoff=0.65)
    suggestions = []
    for key in close_keys:
        match = projections[projections["name_key"].eq(key)].iloc[0]
        suggestions.append(str(match["name"]))
    return suggestions


def summarize_roster(
    matched_rows: list[dict[str, Any]],
    unmatched_rows: list[dict[str, Any]],
    projections: pd.DataFrame,
) -> dict[str, Any]:
    """Create a compact roster-level summary from matched players."""
    matched = pd.DataFrame(matched_rows)
    if matched.empty:
        return {
            "matched_players": 0,
            "unmatched_players": len(unmatched_rows),
            "total_expected_fantasy_value": 0.0,
            "average_expected_fantasy_value": 0.0,
            "total_projection_index": 0.0,
            "best_player": None,
            "needs_review": [],
        }

    expected_value = pd.to_numeric(
        matched["expected_fantasy_value"],
        errors="coerce",
    ).fillna(0)
    projection_index = pd.to_numeric(
        matched["weekly_projection_index"],
        errors="coerce",
    ).fillna(0)
    best_row = matched.sort_values(
        ["expected_fantasy_value", "weekly_projection_index"],
        ascending=[False, False],
    ).iloc[0]
    needs_review = matched[expected_value.lt(0)][
        ["name", "team", "player_type", "expected_fantasy_value"]
    ].head(5)

    return {
        "matched_players": int(len(matched)),
        "unmatched_players": int(len(unmatched_rows)),
        "total_expected_fantasy_value": round(float(expected_value.sum()), 2),
        "average_expected_fantasy_value": round(float(expected_value.mean()), 2),
        "total_projection_index": round(float(projection_index.sum()), 1),
        "best_player": {
            "name": str(best_row["name"]),
            "team": str(best_row["team"]),
            "player_type": str(best_row["player_type"]),
            "expected_fantasy_value": float(best_row["expected_fantasy_value"]),
        },
        "needs_review": needs_review.to_dict(orient="records"),
    }


def summarize_lineup(
    active_rows: list[dict[str, Any]],
    unmatched_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize active lineup value and category totals."""
    unmatched_rows = unmatched_rows or []
    active = pd.DataFrame(active_rows)
    if active.empty:
        return {
            "active_players": 0,
            "unmatched_players": len(unmatched_rows),
            "unavailable_players": 0,
            "total_expected_fantasy_value": 0.0,
            "total_projection_index": 0.0,
            "batting": _empty_batting_totals(),
            "pitching": _empty_pitching_totals(),
        }

    scored_active = active[_availability_mask(active)]
    if scored_active.empty:
        return {
            "active_players": 0,
            "unmatched_players": len(unmatched_rows),
            "unavailable_players": int(len(active)),
            "total_expected_fantasy_value": 0.0,
            "total_projection_index": 0.0,
            "batting": _empty_batting_totals(),
            "pitching": _empty_pitching_totals(),
        }

    expected_value = pd.to_numeric(
        scored_active["expected_fantasy_value"],
        errors="coerce",
    ).fillna(0)
    projection_index = pd.to_numeric(
        scored_active["weekly_projection_index"],
        errors="coerce",
    ).fillna(0)
    batters = scored_active[scored_active["player_type"].eq("batter")]
    pitchers = scored_active[scored_active["player_type"].eq("pitcher")]
    return {
        "active_players": int(len(scored_active)),
        "unmatched_players": len(unmatched_rows),
        "unavailable_players": int(len(active) - len(scored_active)),
        "total_expected_fantasy_value": round(float(expected_value.sum()), 2),
        "total_projection_index": round(float(projection_index.sum()), 1),
        "batting": aggregate_batting_categories(batters.to_dict(orient="records")),
        "pitching": aggregate_pitching_categories(pitchers.to_dict(orient="records")),
    }


def optimize_lineup(matched_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the best active lineup from entered active and bench players."""
    matched = pd.DataFrame(matched_rows)
    if matched.empty:
        return {
            "summary": summarize_lineup([]),
            "lineup": [],
            "bench": [],
            "unavailable": [],
            "changes": [],
            "assumption": _optimizer_assumption(),
        }

    matched["expected_fantasy_value"] = pd.to_numeric(
        matched["expected_fantasy_value"],
        errors="coerce",
    ).fillna(0)
    matched["weekly_projection_index"] = pd.to_numeric(
        matched["weekly_projection_index"],
        errors="coerce",
    ).fillna(0)

    available = matched[_availability_mask(matched)]
    batters = _sort_for_optimizer(available[available["player_type"].eq("batter")])
    pitchers = _sort_for_optimizer(available[available["player_type"].eq("pitcher")])
    active_now = matched[matched["lineup_status"].eq("active")]

    selected_batters = _assign_hitters_to_slots(batters)
    selected_pitchers = _assign_pitchers_to_slots(pitchers, active_now)

    optimized_lineup = []
    for row in selected_batters:
        optimized_lineup.append(_clean_output_row(row))
    for row in selected_pitchers:
        optimized_lineup.append(_clean_output_row(row))

    selected_keys = {
        _lineup_unique_key(row)
        for row in optimized_lineup
    }
    optimized_bench = [
        _clean_output_row(row)
        for _, row in _sort_for_optimizer(matched).iterrows()
        if _lineup_unique_key(row) not in selected_keys
    ]

    summary = summarize_lineup(optimized_lineup)
    return {
        "summary": summary,
        "lineup": optimized_lineup,
        "bench": optimized_bench,
        "unavailable": _availability_warning_rows(matched.to_dict(orient="records")),
        "changes": build_lineup_changes(optimized_lineup, active_now.to_dict(orient="records")),
        "assumption": _optimizer_assumption(),
    }


def build_lineup_changes(
    optimized_lineup: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build true start/sit changes, ignoring active-player slot reshuffling."""
    changes = []
    changes.extend(_build_hitter_set_changes(optimized_lineup, active_rows))
    changes.extend(_build_pitcher_set_changes(optimized_lineup, active_rows))
    return sorted(changes, key=lambda row: (_slot_sort_order(str(row.get("slot_id", ""))), -float(row.get("value_gain") or 0)))


def aggregate_batting_categories(players: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate hitter categories using correct ratio math."""
    if not players:
        return _empty_batting_totals()
    df = pd.DataFrame(players)
    ab = _sum_column(df, "ab")
    h = _sum_column(df, "h")
    bb = _sum_column(df, "bb")
    hbp = _sum_column(df, "hbp")
    sf = _sum_column(df, "sf")
    tb = _sum_column(df, "total_bases")
    obp_denominator = ab + bb + hbp + sf
    avg = h / ab if ab else 0.0
    obp = (h + bb + hbp) / obp_denominator if obp_denominator else 0.0
    slg = tb / ab if ab else 0.0
    return {
        "ab": round(ab, 1),
        "h": round(h, 1),
        "hr": round(_sum_column(df, "hr"), 1),
        "r": round(_sum_column(df, "r"), 1),
        "rbi": round(_sum_column(df, "rbi"), 1),
        "sb": round(_sum_column(df, "sb"), 1),
        "avg": round(avg, 3),
        "ops": round(obp + slg, 3),
    }


def aggregate_pitching_categories(players: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate pitcher categories using correct ratio math."""
    if not players:
        return _empty_pitching_totals()
    df = pd.DataFrame(players)
    ip = _sum_column(df, "ip")
    er = _sum_column(df, "er")
    baserunners = _sum_column(df, "baserunners")
    era = er * 9 / ip if ip else 0.0
    whip = baserunners / ip if ip else 0.0
    return {
        "ip": round(ip, 1),
        "w": round(_sum_column(df, "w"), 1),
        "sv": round(_sum_column(df, "sv"), 1),
        "hld": round(_sum_column(df, "hld"), 1),
        "k": round(_sum_column(df, "k"), 1),
        "era": round(era, 2),
        "whip": round(whip, 2),
    }


def generate_lineup_analysis(
    current_summary: dict[str, Any],
    optimized: dict[str, Any],
    matched_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate lineup analysis, with optional Bedrock/OpenAI support."""
    fallback = generate_local_lineup_analysis(current_summary, optimized, matched_rows)
    llm_text, bedrock_status = generate_bedrock_lineup_analysis(
        current_summary,
        optimized,
        matched_rows,
    )
    if llm_text:
        llm_text = _with_verified_availability_evidence(llm_text, optimized)
        llm_text = _with_verified_matchup_evidence(llm_text, optimized)
        return {
            "source": "aws_bedrock_converse",
            "text": llm_text,
            "bedrock": bedrock_status,
        }
    llm_text = generate_openai_lineup_analysis(current_summary, optimized, matched_rows)
    if llm_text:
        llm_text = _with_verified_availability_evidence(llm_text, optimized)
        llm_text = _with_verified_matchup_evidence(llm_text, optimized)
        return {
            "source": "openai_responses_api",
            "text": llm_text,
            "bedrock": bedrock_status,
        }
    return {"source": "local_fallback", "text": fallback, "bedrock": bedrock_status}


def generate_projection_chat_response(
    question: str,
    history: list[dict[str, Any]],
    current_summary: dict[str, Any],
    optimized: dict[str, Any],
    matched_rows: list[dict[str, Any]],
    unmatched_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Answer a roster/projection question using Bedrock when configured."""
    fallback = generate_local_projection_chat_response(
        question,
        current_summary,
        optimized,
        matched_rows,
        unmatched_rows or [],
    )
    llm_text, bedrock_status = generate_bedrock_projection_chat_response(
        question,
        history,
        current_summary,
        optimized,
        matched_rows,
        unmatched_rows or [],
    )
    if llm_text:
        llm_text = _with_verified_availability_evidence(llm_text, optimized, question)
        return {
            "source": "aws_bedrock_converse",
            "text": llm_text.strip(),
            "bedrock": bedrock_status,
        }
    return {"source": "local_fallback", "text": fallback, "bedrock": bedrock_status}


def generate_local_projection_chat_response(
    question: str,
    current_summary: dict[str, Any],
    optimized: dict[str, Any],
    matched_rows: list[dict[str, Any]],
    unmatched_rows: list[dict[str, Any]] | None = None,
) -> str:
    """Provide a deterministic fallback answer for projection chat."""
    optimized_summary = optimized.get("summary", {})
    value_gain = (
        float(optimized_summary.get("total_expected_fantasy_value") or 0)
        - float(current_summary.get("total_expected_fantasy_value") or 0)
    )
    parts = [
        "I can answer from the loaded roster, projections, optimizer output, and matchup evidence.",
        (
            f"Current lineup value is {float(current_summary.get('total_expected_fantasy_value') or 0):.2f}; "
            f"optimized lineup value is {float(optimized_summary.get('total_expected_fantasy_value') or 0):.2f}; "
            f"value gain is {value_gain:+.2f}."
        ),
    ]
    changes = optimized.get("changes", [])
    if changes:
        change_lines = []
        for change in changes[:3]:
            sit = change.get("sit") or "an open slot"
            change_lines.append(
                f"Start {change.get('start')} over {sit} at {change.get('slot')} "
                f"for {float(change.get('value_gain') or 0):+.2f} value."
            )
        parts.append("Top optimizer decisions: " + " ".join(change_lines))
    else:
        parts.append("The current active lineup already matches the optimizer's legal lineup recommendation.")

    unavailable = optimized.get("unavailable", [])
    if unavailable:
        parts.append(
            "Unavailable players: "
            + " ".join(
                f"{row.get('name')} is marked {row.get('availability_status') or 'unavailable'} and is not startable."
                for row in unavailable[:4]
            )
        )

    top_notes = _top_player_notes(optimized.get("lineup", []), limit=3)
    if top_notes:
        parts.append("Strong projected starters: " + " ".join(top_notes))
    matchup_notes = _top_matchup_notes(optimized.get("lineup", []), limit=3)
    if matchup_notes:
        parts.append("Verified matchup notes: " + " ".join(matchup_notes))
    if unmatched_rows:
        parts.append(
            "Some entered players were unmatched, so answers may be incomplete for: "
            + ", ".join(str(row.get("input") or row.get("player_name")) for row in unmatched_rows[:4])
            + "."
        )
    if str(question or "").strip():
        parts.append(
            "For a more specific answer, ask about one player, one category, or one start/sit decision."
        )
    return "\n\n".join(parts)


def generate_bedrock_projection_chat_response(
    question: str,
    history: list[dict[str, Any]],
    current_summary: dict[str, Any],
    optimized: dict[str, Any],
    matched_rows: list[dict[str, Any]],
    unmatched_rows: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    """Call AWS Bedrock Converse for the projection chatbot."""
    status = bedrock_configuration_status()
    if os.getenv("ENABLE_BEDROCK_ANALYSIS", "").lower() not in {"1", "true", "yes"}:
        status["reason"] = "ENABLE_BEDROCK_ANALYSIS is not true in this server process."
        return None, status

    model_id = os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)

    try:
        import boto3
    except ImportError:
        status["boto3_installed"] = False
        status["reason"] = "boto3 is not installed in the Python environment running the app."
        return None, status

    status["boto3_installed"] = True
    if not status["region"]:
        status["reason"] = "AWS_REGION or AWS_DEFAULT_REGION is not set in this server process."
        return None, status

    prompt_payload = _chat_prompt_payload(
        question,
        history,
        current_summary,
        optimized,
        matched_rows,
        unmatched_rows,
    )
    try:
        status["attempted"] = True
        client = boto3.client("bedrock-runtime", region_name=status["region"])
        response = client.converse(
            modelId=model_id,
            system=[
                {
                    "text": (
                        "You are a fantasy baseball projection chatbot inside a lineup optimizer. "
                        "Answer the user's question using only the provided roster, projections, "
                        "optimizer decisions, category totals, and verified matchup evidence. "
                        "Be specific: cite player value, weekly stat projections, category totals, "
                        "start/sit value gain, handedness notes, and verified matchup evidence when "
                        "they are relevant. Do not invent injuries, news, starters, opposing pitchers, "
                        "teams, handedness, stats, or schedule details. If the question asks for news, "
                        "injury status, weather, or web information, say this chat does not have web "
                        "search yet and answer only from the projection context. Respect lineup "
                        "constraints: do not suggest a player at an ineligible position, and do not "
                        "treat pitcher slot order as meaningful. If unavailable_players is not empty, "
                        "explain that those players are not startable in this optimizer context. "
                        "Availability is a hard constraint and has priority over projection value, "
                        "category projections, and matchup analysis. If a lineup_decision has "
                        "reason=sit_player_unavailable, explain the replacement as filling an "
                        "unavailable slot, not as being more talented or better projected than the "
                        "unavailable player. "
                        "Keep the answer conversational and "
                        "analyst-like, usually 2 to 5 short paragraphs. No markdown tables."
                    )
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": json.dumps(prompt_payload)}],
                }
            ],
            inferenceConfig={
                "maxTokens": int(os.getenv("BEDROCK_CHAT_MAX_TOKENS", "900")),
                "temperature": float(os.getenv("BEDROCK_CHAT_TEMPERATURE", "0.2")),
            },
        )
        text = response["output"]["message"]["content"][0]["text"]
        status["ok"] = bool(text)
        status["reason"] = "Bedrock returned chat text." if text else "Bedrock returned an empty response."
        return text, status
    except Exception as exc:
        status["ok"] = False
        status["error_type"] = type(exc).__name__
        status["reason"] = str(exc)
        print(f"Bedrock chat failed: {type(exc).__name__}: {exc}")
        return None, status


def generate_local_lineup_analysis(
    current_summary: dict[str, Any],
    optimized: dict[str, Any],
    matched_rows: list[dict[str, Any]],
) -> str:
    """Generate a deterministic analyst-style explanation when no LLM is configured."""
    optimized_summary = optimized.get("summary", {})
    value_gain = round(
        float(optimized_summary.get("total_expected_fantasy_value") or 0)
        - float(current_summary.get("total_expected_fantasy_value") or 0),
        2,
    )
    changes = optimized.get("changes", [])
    batting = optimized_summary.get("batting", {})
    pitching = optimized_summary.get("pitching", {})

    sections = []
    if value_gain > 0:
        sections.append(
            "Lineup verdict:\n"
            f"The optimizer improves projected roster value by {value_gain:+.2f}. "
            f"Current value is {float(current_summary.get('total_expected_fantasy_value') or 0):.2f}; "
            f"optimized value is {float(optimized_summary.get('total_expected_fantasy_value') or 0):.2f}."
        )
    else:
        sections.append(
            "Lineup verdict:\n"
            "Your current active lineup already matches the optimizer's baseline recommendation. "
            f"Current and optimized value are both {float(optimized_summary.get('total_expected_fantasy_value') or 0):.2f}."
        )

    if changes:
        decision_lines = []
        for change in changes[:3]:
            if change.get("sit"):
                decision_lines.append(
                    f"Start {change['start']} over {change['sit']} at {change.get('slot', 'slot')}; "
                    f"that is worth {float(change.get('value_gain') or 0):+.2f} value."
                )
            else:
                decision_lines.append(
                    f"Start {change['start']} in an open {change.get('slot', 'slot')} spot."
                )
        sections.append("Start/sit decisions:\n" + "\n".join(decision_lines))

    unavailable = optimized.get("unavailable", [])
    if unavailable:
        lines = []
        for row in unavailable[:4]:
            status = row.get("availability_status") or "unavailable"
            slot = row.get("slot") or row.get("lineup_status") or "roster"
            lines.append(f"{row.get('name')} ({slot}) is marked {status} and is excluded from optimized starters.")
        sections.append("Availability notes:\n" + "\n".join(lines))

    strong_players = _top_player_notes(optimized.get("lineup", []))
    if strong_players:
        sections.append("Strong projected starters:\n" + "\n".join(strong_players))

    sections.append(
        "Category read:\n"
        f"Optimized hitting projects for {batting.get('hr', 0)} HR, "
        f"{batting.get('rbi', 0)} RBI, {batting.get('sb', 0)} SB, "
        f"a {batting.get('avg', 0):.3f} AVG, and a {batting.get('ops', 0):.3f} OPS. "
        f"Optimized pitching projects for {pitching.get('ip', 0)} IP, "
        f"{pitching.get('k', 0)} K, {pitching.get('sv', 0)} SV, "
        f"a {pitching.get('era', 0):.2f} ERA, and a {pitching.get('whip', 0):.2f} WHIP."
    )
    matchup_notes = _top_matchup_notes(optimized.get("lineup", []))
    if matchup_notes:
        sections.append("Matchup evidence:\n" + "\n".join(matchup_notes))
    sections.append("Optimizer rule:\n" + optimized.get("assumption", _optimizer_assumption()))
    return "\n\n".join(sections)


def _top_player_notes(
    lineup: list[dict[str, Any]],
    limit: int = 4,
) -> list[str]:
    players = sorted(
        lineup,
        key=lambda row: (
            float(row.get("expected_fantasy_value") or 0),
            float(row.get("weekly_projection_index") or 0),
        ),
        reverse=True,
    )
    notes = []
    for row in players[:limit]:
        name = str(row.get("name") or "")
        value = float(row.get("expected_fantasy_value") or 0)
        if not name:
            continue
        if row.get("player_type") == "pitcher":
            notes.append(
                f"{name}: {value:+.2f} value with {float(row.get('ip') or 0):.1f} IP, "
                f"{float(row.get('k') or 0):.1f} K, {float(row.get('era') or 0):.2f} ERA, "
                f"and {float(row.get('whip') or 0):.2f} WHIP projected."
            )
        else:
            notes.append(
                f"{name}: {value:+.2f} value with {float(row.get('hr') or 0):.1f} HR, "
                f"{float(row.get('rbi') or 0):.1f} RBI, {float(row.get('sb') or 0):.1f} SB, "
                f"{float(row.get('avg') or 0):.3f} AVG, and {float(row.get('ops') or 0):.3f} OPS projected."
            )
    return notes


def _top_matchup_notes(
    lineup: list[dict[str, Any]],
    limit: int = 2,
) -> list[str]:
    candidates = []
    for row in lineup:
        if row.get("player_type") != "batter":
            continue
        context = str(row.get("matchup_context") or "").strip()
        favorable = str(row.get("favorable_batter_matchups") or "").strip()
        edge = pd.to_numeric(row.get("matchup_edge_pct"), errors="coerce")
        if not context or (not favorable and (pd.isna(edge) or float(edge) < 4)):
            continue
        candidates.append(
            (
                float(row.get("expected_fantasy_value") or 0),
                str(row.get("name") or ""),
                context,
            )
        )
    candidates.sort(reverse=True)
    return [
        f"{name} has a useful matchup note: {context}."
        for _, name, context in candidates[:limit]
        if name and context
    ]


def _with_verified_availability_evidence(
    analysis_text: str,
    optimized: dict[str, Any],
    question: str | None = None,
) -> str:
    notes = _availability_notes_for_text(
        optimized.get("unavailable", []),
        question=question,
    )
    if not notes:
        return analysis_text

    verified_section = "Availability notes:\n" + "\n".join(notes)
    text = str(analysis_text or "").strip()
    if not text:
        return verified_section

    section_pattern = re.compile(
        r"(^|\n)Availability notes:\s*.*?(?=\n(?:Matchup evidence|Category read|Risk notes|Optimizer rule|Lineup verdict|Start/sit decisions|Strong projected starters):|\Z)",
        flags=re.S,
    )
    if section_pattern.search(text):
        return section_pattern.sub(lambda match: match.group(1) + verified_section, text)
    return verified_section + "\n\n" + text


def _availability_notes_for_text(
    unavailable_rows: list[dict[str, Any]],
    question: str | None = None,
    limit: int = 4,
) -> list[str]:
    if not unavailable_rows:
        return []

    question_key = normalize_player_name(question or "")
    selected = []
    question_tokens = set(question_key.split())
    for row in unavailable_rows:
        name = str(row.get("name") or "")
        name_key = normalize_player_name(name)
        name_tokens = {token for token in name_key.split() if len(token) >= 4}
        if (
            not question_key
            or (name_key and name_key in question_key)
            or bool(name_tokens.intersection(question_tokens))
        ):
            selected.append(row)

    availability_words = {"injured", "injury", "il", "hurt", "unavailable", "available", "out"}
    if not selected and availability_words.intersection(question_key.split()):
        selected = unavailable_rows
    if not selected and question is None:
        selected = unavailable_rows

    notes = []
    for row in selected[:limit]:
        name = row.get("name")
        status = row.get("availability_status") or "unavailable"
        note = row.get("availability_note")
        if not name:
            continue
        sentence = f"{name} is marked {status} and is not startable in the optimized lineup."
        if note:
            sentence += f" Source note: {note}."
        notes.append(sentence)
    return notes


def _with_verified_matchup_evidence(
    analysis_text: str,
    optimized: dict[str, Any],
) -> str:
    notes = _top_matchup_notes(optimized.get("lineup", []), limit=5)
    if not notes:
        return analysis_text

    verified_section = "Matchup evidence:\n" + "\n".join(notes)
    text = str(analysis_text or "").strip()
    if not text:
        return verified_section

    section_pattern = re.compile(
        r"(^|\n)Matchup evidence:\s*.*?(?=\n(?:Category read|Risk notes|Optimizer rule|Lineup verdict|Start/sit decisions|Strong projected starters):|\Z)",
        flags=re.S,
    )
    if section_pattern.search(text):
        return section_pattern.sub(lambda match: match.group(1) + verified_section, text)
    return text + "\n\n" + verified_section


def generate_bedrock_lineup_analysis(
    current_summary: dict[str, Any],
    optimized: dict[str, Any],
    matched_rows: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    """Call AWS Bedrock Converse when explicitly configured."""
    status = bedrock_configuration_status()
    if os.getenv("ENABLE_BEDROCK_ANALYSIS", "").lower() not in {"1", "true", "yes"}:
        status["reason"] = "ENABLE_BEDROCK_ANALYSIS is not true in this server process."
        return None, status

    model_id = os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)

    try:
        import boto3
    except ImportError:
        status["boto3_installed"] = False
        status["reason"] = "boto3 is not installed in the Python environment running the app."
        return None, status

    status["boto3_installed"] = True
    if not status["region"]:
        status["reason"] = "AWS_REGION or AWS_DEFAULT_REGION is not set in this server process."
        return None, status

    prompt_payload = _analysis_prompt_payload(current_summary, optimized, matched_rows)
    try:
        status["attempted"] = True
        client = boto3.client(
            "bedrock-runtime",
            region_name=status["region"],
        )
        response = client.converse(
            modelId=model_id,
            system=[
                {
                    "text": (
                        "You are a sharp fantasy baseball analyst writing for an experienced "
                        "manager. Use baseball-specific language, but stay grounded in the "
                        "provided data. Return plain text with these section labels when "
                        "supported by the data: Lineup verdict, Start/sit decisions, Strong "
                        "projected starters, Matchup evidence, Category read, Risk notes. "
                        "Use 2 to 4 sentences per section. Do not use markdown symbols, "
                        "bold text, or bullet points. Explain why optimized starters are "
                        "preferred over bench players using lineup_decisions and optimized_bench. "
                        "If unavailable_players is not empty, explain that those players were "
                        "excluded from the optimized lineup because of availability status. "
                        "Availability is a hard constraint and has priority over projection value, "
                        "category projections, and matchup analysis. If a lineup_decision has "
                        "reason=sit_player_unavailable, explain the replacement as filling an "
                        "unavailable slot, not as being more talented or better projected than the "
                        "unavailable player. "
                        "If lineup_decisions is empty, say the current active lineup already "
                        "matches the optimizer and focus on the strongest projected starters. "
                        "Never say one active starter is replacing another active starter "
                        "unless that exact pair appears in lineup_decisions. Pitcher slot "
                        "order does not matter, so never describe moving a pitcher from one "
                        "P slot to another as a swap. Mention a player's value number and "
                        "relevant category projections when discussing him. The Matchup "
                        "evidence section must use only verified_matchup_evidence. Do not "
                        "create new player-pitcher pairs, do not move a pitcher matchup from "
                        "one hitter to another, and do not mention specific opposing pitcher "
                        "names outside the Matchup evidence section. Use matchup.batter_side "
                        "and matchup.relevant_probables to identify the correct hitter-side "
                        "split; do not infer handedness from generic LHB/RHB text. Cite "
                        "numeric evidence such as relevant split wOBA allowed or ZiPS ERA/WHIP "
                        "only when it appears in verified_matchup_evidence or the named "
                        "player's matchup fields. Do not invent players, injuries, roles, "
                        "teams, pitchers, or stats."
                    )
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": json.dumps(prompt_payload)}],
                }
            ],
            inferenceConfig={
                "maxTokens": int(os.getenv("BEDROCK_MAX_TOKENS", "760")),
                "temperature": float(os.getenv("BEDROCK_TEMPERATURE", "0.15")),
            },
        )
        text = response["output"]["message"]["content"][0]["text"]
        status["ok"] = bool(text)
        status["reason"] = "Bedrock returned analysis text." if text else "Bedrock returned an empty response."
        return text, status
    except Exception as exc:
        status["ok"] = False
        status["error_type"] = type(exc).__name__
        status["reason"] = str(exc)
        print(f"Bedrock analysis failed: {type(exc).__name__}: {exc}")
        return None, status


def bedrock_configuration_status() -> dict[str, Any]:
    """Return safe Bedrock diagnostics without exposing credentials."""
    enabled_value = os.getenv("ENABLE_BEDROCK_ANALYSIS", "")
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    model_id = os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)

    try:
        import boto3  # noqa: F401

        boto3_installed = True
    except ImportError:
        boto3_installed = False

    return {
        "enabled": enabled_value.lower() in {"1", "true", "yes"},
        "enabled_value": enabled_value,
        "model_id": model_id,
        "region": region,
        "boto3_installed": boto3_installed,
        "attempted": False,
        "ok": False,
        "has_aws_access_key_env": bool(os.getenv("AWS_ACCESS_KEY_ID")),
        "has_aws_profile_env": bool(os.getenv("AWS_PROFILE")),
    }


def generate_openai_lineup_analysis(
    current_summary: dict[str, Any],
    optimized: dict[str, Any],
    matched_rows: list[dict[str, Any]],
) -> str | None:
    """Call OpenAI Responses API when explicitly configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or os.getenv("ENABLE_LLM_ANALYSIS", "").lower() not in {"1", "true", "yes"}:
        return None

    prompt_payload = _analysis_prompt_payload(current_summary, optimized, matched_rows)
    body = {
        "model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        "instructions": (
            "You are a sharp fantasy baseball analyst. Return plain text sections labeled "
            "Lineup verdict, Start/sit decisions, Strong projected starters, Matchup evidence, "
            "Category read, and Risk notes when supported by the provided data. Use baseball-"
            "specific language and cite player values, projected categories, and matchup "
            "numbers. The Matchup evidence section must use only verified_matchup_evidence. "
            "Do not create new player-pitcher pairs or mention specific opposing pitcher names "
            "outside the Matchup evidence section. Use matchup.batter_side to identify the "
            "correct split. Do not invent players or stats."
        ),
        "input": json.dumps(prompt_payload),
    }
    request = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return _extract_response_text(payload)
    except Exception:
        return None


def _analysis_prompt_payload(
    current_summary: dict[str, Any],
    optimized: dict[str, Any],
    matched_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    optimized_lineup = optimized.get("lineup", [])
    return {
        "hard_constraints": [
            "Availability status is a hard lineup constraint.",
            "Unavailable players are not startable even if their projection value is high.",
            "If a decision reason is sit_player_unavailable, explain the replacement as filling an unavailable slot.",
        ],
        "current_summary": current_summary,
        "optimized_summary": optimized.get("summary", {}),
        "changes": optimized.get("changes", [])[:5],
        "lineup_decisions": _analysis_decision_rows(
            optimized.get("changes", []),
            optimized_lineup,
            matched_rows,
        ),
        "top_optimized_players": _top_optimized_players(optimized_lineup),
        "optimized_bench": _analysis_player_rows(optimized.get("bench", []), limit=8),
        "unavailable_players": optimized.get("unavailable", [])[:8],
        "value_explanation": {
            "expected_fantasy_value": (
                "Player-type-relative value index. Around 0 is average for hitters "
                "versus hitters or pitchers versus pitchers; positive is above average."
            ),
            "current_lineup_value": "Sum of expected_fantasy_value for entered active slots.",
            "optimized_lineup_value": (
                "Sum after the optimizer chooses the best legal lineup from active plus bench."
            ),
            "value_gain": "Optimized lineup value minus current lineup value.",
        },
        "optimizer_assumption": optimized.get("assumption"),
        "matched_players": _analysis_player_rows(matched_rows, limit=25),
        "verified_matchup_evidence": _top_matchup_notes(optimized_lineup, limit=5),
        "verified_availability_evidence": _availability_notes_for_text(
            optimized.get("unavailable", []),
        ),
    }


def _chat_prompt_payload(
    question: str,
    history: list[dict[str, Any]],
    current_summary: dict[str, Any],
    optimized: dict[str, Any],
    matched_rows: list[dict[str, Any]],
    unmatched_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    optimized_lineup = optimized.get("lineup", [])
    return {
        "user_question": str(question or "").strip(),
        "hard_constraints": [
            "Availability status is a hard lineup constraint.",
            "Unavailable players are not startable even if their projection value is high.",
            "If a decision reason is sit_player_unavailable, explain the replacement as filling an unavailable slot.",
        ],
        "conversation_history": _safe_chat_history(history),
        "current_summary": current_summary,
        "optimized_summary": optimized.get("summary", {}),
        "lineup_decisions": _analysis_decision_rows(
            optimized.get("changes", []),
            optimized_lineup,
            matched_rows,
        ),
        "optimized_lineup": _analysis_player_rows(optimized_lineup, limit=18),
        "optimized_bench": _analysis_player_rows(optimized.get("bench", []), limit=8),
        "unavailable_players": optimized.get("unavailable", [])[:8],
        "all_matched_players": _analysis_player_rows(matched_rows, limit=28),
        "unmatched_players": unmatched_rows[:8],
        "verified_matchup_evidence": _top_matchup_notes(optimized_lineup, limit=6),
        "verified_availability_evidence": _availability_notes_for_text(
            optimized.get("unavailable", []),
            question=question,
        ),
        "value_explanation": {
            "expected_fantasy_value": (
                "Player-type-relative value index. Around 0 is average for hitters "
                "versus hitters or pitchers versus pitchers; positive is above average."
            ),
            "current_lineup_value": "Sum of expected_fantasy_value for entered active slots.",
            "optimized_lineup_value": (
                "Sum after the optimizer chooses the best legal lineup from active plus bench."
            ),
            "value_gain": "Optimized lineup value minus current lineup value.",
        },
        "optimizer_assumption": optimized.get("assumption"),
        "limitations": (
            "This chat has roster, projection, optimizer, and verified matchup context only. "
            "It does not have live web search, injury feeds, weather, or confirmed MLB lineups."
        ),
    }


def _safe_chat_history(history: list[dict[str, Any]], limit: int = 6) -> list[dict[str, str]]:
    safe_history = []
    for item in history[-limit:]:
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        text = str(item.get("text") or item.get("content") or "").strip()
        if not text:
            continue
        safe_history.append({"role": role, "text": text[:1200]})
    return safe_history


def _top_optimized_players(
    optimized_lineup: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    players = sorted(
        optimized_lineup,
        key=lambda row: (
            float(row.get("expected_fantasy_value") or 0),
            float(row.get("weekly_projection_index") or 0),
        ),
        reverse=True,
    )
    return _analysis_player_rows(players, limit=limit)


def _analysis_player_rows(
    rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    return [_analysis_player_row(row) for row in rows[:limit]]


def _analysis_player_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "name": row.get("name"),
        "team": row.get("team"),
        "player_type": row.get("player_type"),
        "slot": row.get("optimized_slot") or row.get("slot_label"),
        "lineup_status": row.get("lineup_status"),
        "is_available": row.get("is_available"),
        "availability_status": row.get("availability_status"),
        "availability_note": row.get("availability_note"),
        "expected_fantasy_value": row.get("expected_fantasy_value"),
        "weekly_projection_index": row.get("weekly_projection_index"),
        "key_stats": _analysis_key_stats(row),
        "matchup": _analysis_matchup_context(row),
    }


def _analysis_decision_rows(
    changes: list[dict[str, Any]],
    optimized_lineup: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    optimized_lookup = _analysis_name_lookup(optimized_lineup)
    matched_lookup = _analysis_name_lookup(matched_rows)
    decisions = []
    for change in changes[:limit]:
        start_key = normalize_player_name(change.get("start"))
        sit_key = normalize_player_name(change.get("sit"))
        start_row = optimized_lookup.get(start_key) or matched_lookup.get(start_key)
        sit_row = matched_lookup.get(sit_key) if sit_key else None
        decisions.append(
            {
                "slot": change.get("slot"),
                "player_type": change.get("player_type"),
                "value_gain": change.get("value_gain"),
                "reason": change.get("reason"),
                "sit_is_available": change.get("sit_is_available"),
                "sit_availability_status": change.get("sit_availability_status"),
                "sit_availability_note": change.get("sit_availability_note"),
                "start": _analysis_player_row(start_row),
                "sit": _analysis_player_row(sit_row),
            }
        )
    return decisions


def _analysis_name_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        normalize_player_name(row.get("name")): row
        for row in rows
        if normalize_player_name(row.get("name"))
    }


def _analysis_key_stats(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("player_type") == "pitcher":
        return {
            "ip": row.get("ip"),
            "k": row.get("k"),
            "w": row.get("w"),
            "sv": row.get("sv"),
            "era": row.get("era"),
            "whip": row.get("whip"),
        }
    return {
        "ab": row.get("ab"),
        "hr": row.get("hr"),
        "r": row.get("r"),
        "rbi": row.get("rbi"),
        "sb": row.get("sb"),
        "avg": row.get("avg"),
        "ops": row.get("ops"),
    }


def _analysis_matchup_context(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("player_type") != "batter":
        return {}
    return {
        "context": row.get("matchup_context"),
        "edge_pct": row.get("matchup_edge_pct"),
        "batter_side": row.get("batter_side_note"),
        "batter_bats": row.get("batter_bats"),
        "stand_vs_lhp": row.get("stand_vs_lhp"),
        "stand_vs_rhp": row.get("stand_vs_rhp"),
        "games_vs_lhp": row.get("games_vs_lhp"),
        "games_vs_rhp": row.get("games_vs_rhp"),
        "relevant_probables": row.get("relevant_opposing_pitcher_summary"),
        "favorable_relevant_matchups": row.get("favorable_batter_matchups"),
        "tough_relevant_matchups": row.get("tough_batter_matchups"),
        "generic_opposing_pitchers": row.get("opposing_pitcher_summary"),
    }


def projection_metadata(
    schedule_path: Path = SCHEDULE_PATH,
    projections: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Describe projection freshness and table size for the UI."""
    schedule_start = None
    schedule_end = None
    if schedule_path.exists():
        schedule = pd.read_csv(schedule_path)
        if "game_date" in schedule.columns and not schedule.empty:
            dates = pd.to_datetime(schedule["game_date"], errors="coerce")
            schedule_start = dates.min().date().isoformat()
            schedule_end = dates.max().date().isoformat()

    player_count = int(len(projections)) if projections is not None else 0
    return {
        "schedule_start": schedule_start,
        "schedule_end": schedule_end,
        "player_count": player_count,
        "lineup_slots": {
            "hitters": [
                {"id": slot_id, "label": _slot_display_label(slot_id)}
                for slot_id in HITTER_SLOTS
            ],
            "pitchers": [
                {"id": slot_id, "label": _slot_display_label(slot_id)}
                for slot_id in PITCHER_SLOTS
            ],
            "bench": [
                {"id": slot_id, "label": _slot_display_label(slot_id)}
                for slot_id in BENCH_SLOTS
            ],
        },
        "value_definition": (
            "Baseline expected fantasy value is a player-type-relative z-score "
            "from the current weekly projection index."
        ),
        "value_explanation": {
            "plain_language": (
                "Value is a lineup-strength index, not fantasy points. A player's "
                "value shows how far above or below the projection-pool average he "
                "is compared with players of the same type."
            ),
            "player_value": (
                "Player Value compares hitters to hitters and pitchers to pitchers. "
                "About 0 is average for that player type, positive is above average, "
                "and negative is below average."
            ),
            "current_value": (
                "Current Value is the sum of Player Value for the players currently "
                "entered in active lineup slots."
            ),
            "optimized_value": (
                "Optimized Value is the sum after the optimizer chooses the best legal "
                "active lineup from the starters and bench players entered."
            ),
            "value_gain": (
                "Value Gain is Optimized Value minus Current Value. A positive number "
                "means the optimizer found an improvement; 0 means it did not find a "
                "better legal lineup."
            ),
        },
    }


def normalize_player_name(value: object) -> str:
    """Normalize names for matching while preserving display names elsewhere."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_roster_part(part: str) -> RosterEntry:
    raw_text = part.strip()
    tokens = [token.strip() for token in re.split(r"[,|;/]", raw_text) if token.strip()]
    requested_type = None
    name_tokens = []

    for token in tokens or [raw_text]:
        normalized = normalize_player_name(token)
        if normalized in TYPE_WORDS:
            requested_type = TYPE_WORDS[normalized]
            continue
        if normalized in POSITION_WORDS:
            continue
        name_tokens.append(token)

    player_name = name_tokens[0].strip() if name_tokens else raw_text
    player_name = re.sub(r"\((batter|hitter|pitcher|sp|rp|p)\)", "", player_name, flags=re.I)
    return RosterEntry(
        raw_text=raw_text,
        player_name=player_name.strip(),
        requested_type=requested_type,
    )


def _looks_like_comma_list(line: str) -> bool:
    parts = [part.strip() for part in line.split(",") if part.strip()]
    if len(parts) < 3:
        return False
    short_parts = sum(1 for part in parts if len(part) <= 4)
    return short_parts == 0


def _read_projection_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["weekly_projection_index"] = pd.to_numeric(
        df["weekly_projection_index"],
        errors="coerce",
    ).fillna(0)
    return df


def _add_batter_category_projections(
    batters: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    if batters.empty:
        return batters

    projected = batters.copy()
    modeled_columns = [
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
    if "projection_stat_source" in projected.columns and all(
        column in projected.columns for column in modeled_columns
    ):
        return _fill_missing_batter_categories(projected)

    zips_path = processed_dir / "zips_batters_ros.parquet"
    if not zips_path.exists():
        return _fill_missing_batter_categories(projected)

    zips = pd.read_parquet(zips_path)
    zips["mlbamid"] = pd.to_numeric(zips["mlbamid"], errors="coerce")
    projected["mlbamid"] = pd.to_numeric(projected["mlbamid"], errors="coerce")

    zips_columns = [
        "mlbamid",
        "g",
        "pa",
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
        "avg",
        "ops",
    ]
    zips_columns = [column for column in zips_columns if column in zips.columns]
    projected = projected.merge(
        zips[zips_columns].add_prefix("zips_"),
        left_on="mlbamid",
        right_on="zips_mlbamid",
        how="left",
    )

    factor = _safe_divide(
        pd.to_numeric(projected.get("projected_games"), errors="coerce").fillna(0),
        pd.to_numeric(projected.get("zips_g"), errors="coerce"),
    ).fillna(0)

    for column in ["ab", "h", "hr", "r", "rbi", "bb", "hbp", "sf", "sb"]:
        projected[column] = pd.to_numeric(
            projected.get(f"zips_{column}"),
            errors="coerce",
        ).fillna(0) * factor

    for column in ["1b", "2b", "3b"]:
        projected[column] = pd.to_numeric(
            projected.get(f"zips_{column}"),
            errors="coerce",
        ).fillna(0) * factor

    projected["total_bases"] = (
        projected["1b"] + 2 * projected["2b"] + 3 * projected["3b"] + 4 * projected["hr"]
    )
    projected["avg"] = _safe_divide(projected["h"], projected["ab"]).fillna(
        pd.to_numeric(projected.get("zips_avg"), errors="coerce")
    )
    obp_denominator = projected["ab"] + projected["bb"] + projected["hbp"] + projected["sf"]
    obp = _safe_divide(projected["h"] + projected["bb"] + projected["hbp"], obp_denominator)
    slg = _safe_divide(projected["total_bases"], projected["ab"])
    projected["ops"] = (obp + slg).fillna(
        pd.to_numeric(projected.get("zips_ops"), errors="coerce")
    )
    projected["projection_source"] = "zips_scaled_to_week"
    return _fill_missing_batter_categories(projected)


def _add_pitcher_category_projections(
    pitchers: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    zips_path = processed_dir / "zips_pitchers_ros.parquet"
    if not zips_path.exists():
        return _fill_missing_pitcher_categories(pitchers)

    zips = pd.read_parquet(zips_path)
    zips["mlbamid"] = pd.to_numeric(zips["mlbamid"], errors="coerce")
    projected = pitchers.copy()
    projected = projected.drop(
        columns=[column for column in projected.columns if column.startswith("zips_")],
        errors="ignore",
    )

    zips_columns = [
        "name",
        "team",
        "playerid",
        "mlbamid",
        "g",
        "gs",
        "w",
        "sv",
        "hld",
        "ip",
        "h",
        "er",
        "bb",
        "so",
        "era",
        "whip",
        "fpts",
    ]
    zips_columns = [column for column in zips_columns if column in zips.columns]
    zips_subset = zips[zips_columns].copy()

    if projected.empty:
        merged = zips_subset.copy()
        merged["player_type"] = "pitcher"
        merged["player_key"] = merged.get("playerid")
        merged["projected_starts"] = 0
        merged["opponents"] = ""
        merged["skill_score"] = 0
        merged["matchup_score"] = 0
        merged["weekly_projection_index"] = 0
        merged["projection_rank"] = pd.NA
    else:
        projected["mlbamid"] = pd.to_numeric(projected["mlbamid"], errors="coerce")
        merged = projected.merge(
            zips_subset.add_prefix("zips_"),
            left_on="mlbamid",
            right_on="zips_mlbamid",
            how="outer",
        )
        for column in ["name", "team", "playerid", "mlbamid"]:
            if f"zips_{column}" in merged.columns:
                merged[column] = merged[column].fillna(merged[f"zips_{column}"])
        merged["player_type"] = "pitcher"
        merged["player_key"] = merged["player_key"].fillna(merged["playerid"])
        merged["projected_starts"] = pd.to_numeric(
            merged.get("projected_starts"),
            errors="coerce",
        ).fillna(0)
        merged["weekly_projection_index"] = pd.to_numeric(
            merged.get("weekly_projection_index"),
            errors="coerce",
        ).fillna(0)

    zips_games = pd.to_numeric(_first_existing(merged, ["zips_g", "g"]), errors="coerce")
    zips_starts = pd.to_numeric(_first_existing(merged, ["zips_gs", "gs"]), errors="coerce")
    starts = pd.to_numeric(merged.get("projected_starts"), errors="coerce").fillna(0)
    relief_games = (zips_games - zips_starts.fillna(0)).clip(lower=0)

    starter_factor = _safe_divide(starts, zips_starts).fillna(0)
    relief_factor = _safe_divide(pd.Series(3, index=merged.index), relief_games).fillna(0)
    factor = starter_factor.where(starts.gt(0), relief_factor)
    factor = factor.where(factor.le(1), 1).fillna(0)

    for source_column, output_column in [
        ("ip", "ip"),
        ("w", "w"),
        ("sv", "sv"),
        ("hld", "hld"),
        ("so", "k"),
        ("er", "er"),
        ("h", "hits_allowed"),
        ("bb", "bb_allowed"),
    ]:
        values = pd.to_numeric(
            _first_existing(merged, [f"zips_{source_column}", source_column]),
            errors="coerce",
        ).fillna(0)
        fallback_values = values * factor
        if output_column in merged.columns:
            merged[output_column] = pd.to_numeric(
                merged[output_column],
                errors="coerce",
            ).fillna(fallback_values)
        else:
            merged[output_column] = fallback_values

    merged["baserunners"] = merged["hits_allowed"] + merged["bb_allowed"]
    merged["era"] = _safe_divide(merged["er"] * 9, merged["ip"]).fillna(
        pd.to_numeric(_first_existing(merged, ["zips_era", "era"]), errors="coerce")
    )
    merged["whip"] = _safe_divide(merged["baserunners"], merged["ip"]).fillna(
        pd.to_numeric(_first_existing(merged, ["zips_whip", "whip"]), errors="coerce")
    )

    fallback_index = (
        merged["ip"].fillna(0) * 2
        + merged["k"].fillna(0) * 1.5
        + merged["w"].fillna(0) * 5
        + merged["sv"].fillna(0) * 5
        + merged["hld"].fillna(0) * 3
        - merged["er"].fillna(0) * 2
    )
    merged["weekly_projection_index"] = merged["weekly_projection_index"].where(
        merged["weekly_projection_index"].gt(0),
        fallback_index,
    )
    if "projection_stat_source" not in merged.columns:
        merged["projection_stat_source"] = pd.NA
    merged["projection_stat_source"] = merged["projection_stat_source"].fillna(
        "scheduled_starts_or_zips_relief"
    )
    merged["projection_source"] = merged["projection_stat_source"]
    return _fill_missing_pitcher_categories(merged)


def _fill_missing_batter_categories(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()
    for column in [
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
    ]:
        if column not in filled.columns:
            filled[column] = 0.0
        filled[column] = pd.to_numeric(filled[column], errors="coerce").fillna(0)
    return filled


def _fill_missing_pitcher_categories(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()
    for column in [
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
    ]:
        if column not in filled.columns:
            filled[column] = 0.0
        filled[column] = pd.to_numeric(filled[column], errors="coerce").fillna(0)
    return filled


def _slot_requested_type(slot: LineupSlot) -> str | None:
    if slot.slot_type == "pitcher":
        return "pitcher"
    if slot.slot_type == "hitter":
        return "batter"
    return None


def _assign_hitters_to_slots(hitters: pd.DataFrame) -> list[dict[str, Any]]:
    if hitters.empty:
        return []

    rows = [
        _ensure_hitter_eligibility(row.to_dict())
        for _, row in _sort_for_optimizer(hitters).iterrows()
    ]
    states: dict[int, tuple[int, float, float, list[dict[str, Any]]]] = {
        0: (0, 0.0, 0.0, [])
    }

    for row in rows:
        next_states = dict(states)
        for mask, state in states.items():
            count, value, projection_index, assignments = state
            for slot_index, slot_id in enumerate(HITTER_SLOTS):
                bit = 1 << slot_index
                if mask & bit:
                    continue
                if not _can_fill_hitter_slot(row, slot_id):
                    continue

                assigned = dict(row)
                assigned["optimized_slot_id"] = slot_id
                assigned["optimized_slot"] = _slot_display_label(slot_id)
                candidate = (
                    count + 1,
                    value + float(row.get("expected_fantasy_value") or 0),
                    projection_index + float(row.get("weekly_projection_index") or 0),
                    assignments + [assigned],
                )
                new_mask = mask | bit
                if _is_better_assignment(candidate, next_states.get(new_mask)):
                    next_states[new_mask] = candidate
        states = next_states

    best = max(states.values(), key=lambda state: (state[0], state[1], state[2]))
    return sorted(best[3], key=_slot_sort_key)


def _assign_pitchers_to_slots(
    pitchers: pd.DataFrame,
    active_now: pd.DataFrame,
) -> list[dict[str, Any]]:
    if pitchers.empty:
        return []

    selected = _sort_for_optimizer(pitchers).head(len(PITCHER_SLOTS)).to_dict(orient="records")
    selected_keys = {_lineup_unique_key(row) for row in selected}
    active_pitchers = active_now[active_now["player_type"].eq("pitcher")]
    active_by_key = {
        _lineup_unique_key(row): row.to_dict()
        for _, row in active_pitchers.iterrows()
    }
    active_selected_slots = [
        str(row.get("slot_id"))
        for key, row in active_by_key.items()
        if key in selected_keys and row.get("slot_id") in PITCHER_SLOTS
    ]
    open_slots = [
        slot_id
        for slot_id in PITCHER_SLOTS
        if slot_id not in active_selected_slots
    ]

    assigned = []
    for row in selected:
        row = dict(row)
        active_row = active_by_key.get(_lineup_unique_key(row))
        if active_row and active_row.get("slot_id") in PITCHER_SLOTS:
            slot_id = str(active_row["slot_id"])
        else:
            slot_id = open_slots.pop(0)
        row["optimized_slot_id"] = slot_id
        row["optimized_slot"] = _slot_display_label(slot_id)
        assigned.append(row)
    return sorted(assigned, key=_slot_sort_key)


def _build_hitter_set_changes(
    optimized_lineup: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    optimized_hitters = [
        row for row in optimized_lineup if row.get("player_type") == "batter"
    ]
    active_hitters = [
        row for row in active_rows if row.get("player_type") == "batter"
    ]
    active_keys = {_lineup_unique_key(row) for row in active_hitters}
    optimized_keys = {_lineup_unique_key(row) for row in optimized_hitters}
    starts = [
        row for row in optimized_hitters if _lineup_unique_key(row) not in active_keys
    ]
    sits = [
        row for row in active_hitters if _lineup_unique_key(row) not in optimized_keys
    ]

    sit_pool = sorted(
        sits,
        key=lambda row: float(row.get("expected_fantasy_value") or 0),
    )
    changes = []
    for start in sorted(starts, key=_slot_sort_key):
        sit = _find_hitter_sit_for_start(start, sit_pool)
        if sit is not None:
            sit_pool.remove(sit)
        sit_is_available = _is_available_value((sit or {}).get("is_available")) if sit else True
        sit_value = float((sit or {}).get("expected_fantasy_value") or 0) if sit_is_available else 0.0
        changes.append(
            {
                "slot": start.get("optimized_slot"),
                "slot_id": start.get("optimized_slot_id"),
                "start": start.get("name"),
                "start_team": start.get("team"),
                "sit": sit.get("name") if sit else None,
                "sit_team": sit.get("team") if sit else None,
                "sit_is_available": sit_is_available if sit else None,
                "sit_availability_status": (sit or {}).get("availability_status"),
                "sit_availability_note": (sit or {}).get("availability_note"),
                "player_type": "batter",
                "value_gain": round(
                    float(start.get("expected_fantasy_value") or 0) - sit_value,
                    2,
                ),
                "reason": (
                    "sit_player_unavailable"
                    if sit and not sit_is_available
                    else "open_slot"
                    if sit is None
                    else "value_upgrade"
                ),
            }
        )
    return changes


def _find_hitter_sit_for_start(
    start: dict[str, Any],
    sit_pool: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not sit_pool:
        return None
    start_slot_position = _slot_base_position(str(start.get("optimized_slot_id", "")))
    matching_slot_sits = [
        row
        for row in sit_pool
        if _slot_base_position(str(row.get("slot_id", ""))) == start_slot_position
    ]
    if matching_slot_sits:
        return matching_slot_sits[0]
    available_sits = [row for row in sit_pool if _is_available_value(row.get("is_available"))]
    if available_sits:
        return available_sits[0]
    return None


def _build_pitcher_set_changes(
    optimized_lineup: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    optimized_pitchers = [
        row for row in optimized_lineup if row.get("player_type") == "pitcher"
    ]
    active_pitchers = [
        row for row in active_rows if row.get("player_type") == "pitcher"
    ]
    active_keys = {_lineup_unique_key(row) for row in active_pitchers}
    optimized_keys = {_lineup_unique_key(row) for row in optimized_pitchers}
    starts = [
        row for row in optimized_pitchers if _lineup_unique_key(row) not in active_keys
    ]
    sits = [
        row for row in active_pitchers if _lineup_unique_key(row) not in optimized_keys
    ]

    sit_pool = sorted(
        sits,
        key=lambda row: float(row.get("expected_fantasy_value") or 0),
    )
    changes = []
    for start in sorted(
        starts,
        key=lambda row: float(row.get("expected_fantasy_value") or 0),
        reverse=True,
    ):
        sit = sit_pool.pop(0) if sit_pool else None
        sit_is_available = _is_available_value((sit or {}).get("is_available")) if sit else True
        sit_value = float((sit or {}).get("expected_fantasy_value") or 0) if sit_is_available else 0.0
        changes.append(
            {
                "slot": "P",
                "slot_id": start.get("optimized_slot_id"),
                "start": start.get("name"),
                "start_team": start.get("team"),
                "sit": sit.get("name") if sit else None,
                "sit_team": sit.get("team") if sit else None,
                "sit_is_available": sit_is_available if sit else None,
                "sit_availability_status": (sit or {}).get("availability_status"),
                "sit_availability_note": (sit or {}).get("availability_note"),
                "player_type": "pitcher",
                "value_gain": round(
                    float(start.get("expected_fantasy_value") or 0) - sit_value,
                    2,
                ),
                "reason": (
                    "sit_player_unavailable"
                    if sit and not sit_is_available
                    else "open_slot"
                    if sit is None
                    else "value_upgrade"
                ),
            }
        )
    return changes


def _ensure_hitter_eligibility(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    positions = parse_position_list(normalized.get("eligible_positions", []))
    if not positions and normalized.get("lineup_status") == "active":
        slot_position = _slot_base_position(str(normalized.get("slot_id", "")))
        if slot_position:
            positions = [slot_position]
    normalized["eligible_positions"] = positions
    normalized["eligible_positions_label"] = format_positions(positions)
    return normalized


def _can_fill_hitter_slot(row: dict[str, Any], slot_id: str) -> bool:
    if row.get("player_type") != "batter":
        return False
    if slot_id == "UTIL":
        return True
    required_position = _slot_base_position(slot_id)
    return bool(required_position and required_position in row.get("eligible_positions", []))


def _is_better_assignment(
    candidate: tuple[int, float, float, list[dict[str, Any]]],
    current: tuple[int, float, float, list[dict[str, Any]]] | None,
) -> bool:
    if current is None:
        return True
    return candidate[:3] > current[:3]


def _configured_eligibility_for_row(
    row: Any,
    eligibility_lookup: dict[str, list[str]],
) -> list[str]:
    name_key = row.get("name_key") or normalize_player_name(row.get("name"))
    return list(eligibility_lookup.get(str(name_key), []))


def _slot_base_position(slot_id: str) -> str | None:
    label = _slot_display_label(slot_id)
    if label in {"C", "1B", "2B", "3B", "SS", "OF"}:
        return label
    return None


def _dedupe_positions(positions: object) -> list[str]:
    seen = set()
    ordered = []
    for position in positions or []:
        if position in POSITION_ORDER and position not in seen:
            ordered.append(position)
            seen.add(position)
    return sorted(ordered, key=POSITION_ORDER.index)


def _sort_for_optimizer(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work["_available_sort"] = _availability_mask(work).astype(int)
    return work.sort_values(
        ["_available_sort", "expected_fantasy_value", "weekly_projection_index", "name"],
        ascending=[False, False, False, True],
    ).drop(columns=["_available_sort"], errors="ignore")


def _availability_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty or "is_available" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["is_available"].map(_is_available_value)


def _is_available_value(value: object) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"false", "0", "no", "n", "unavailable", "out"}:
        return False
    return True


def _availability_warning_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings = []
    for row in rows:
        if _is_available_value(row.get("is_available")):
            continue
        warnings.append(
            _clean_output_row(
                {
                    "name": row.get("name"),
                    "team": row.get("team"),
                    "player_type": row.get("player_type"),
                    "slot": row.get("slot_label") or row.get("slot_id"),
                    "lineup_status": row.get("lineup_status"),
                    "availability_status": row.get("availability_status"),
                    "availability_note": row.get("availability_note"),
                    "availability_source": row.get("availability_source"),
                    "availability_checked_at": row.get("availability_checked_at"),
                }
            )
        )
    return warnings


def _lineup_unique_key(row: Any) -> str:
    return f"{row.get('player_type')}::{row.get('player_key')}::{row.get('name_key')}"


def _slot_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    slot_id = str(row.get("optimized_slot_id") or row.get("slot_id") or "")
    return (_slot_sort_order(slot_id), slot_id)


def _slot_sort_order(slot_id: str) -> int:
    order = HITTER_SLOTS + PITCHER_SLOTS + BENCH_SLOTS
    if slot_id in order:
        return order.index(slot_id)
    return len(order)


def _optimizer_assumption() -> str:
    return (
        "Optimizer note: hitter swaps are constrained by known fantasy eligibility. "
        "Active hitter slots provide fallback eligibility, and bench hitters without "
        "manual/configured positions are treated as UTIL-only."
    )


def _slot_display_label(slot_id: str) -> str:
    return HITTER_SLOT_LABELS.get(slot_id, slot_id)


def _empty_batting_totals() -> dict[str, float]:
    return {
        "ab": 0.0,
        "h": 0.0,
        "hr": 0.0,
        "r": 0.0,
        "rbi": 0.0,
        "sb": 0.0,
        "avg": 0.0,
        "ops": 0.0,
    }


def _empty_pitching_totals() -> dict[str, float]:
    return {
        "ip": 0.0,
        "w": 0.0,
        "sv": 0.0,
        "hld": 0.0,
        "k": 0.0,
        "era": 0.0,
        "whip": 0.0,
    }


def _sum_column(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def _safe_divide(numerator: Any, denominator: Any) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    if not isinstance(numerator, pd.Series):
        numerator = pd.Series(numerator)
    if not isinstance(denominator, pd.Series):
        denominator = pd.Series(denominator, index=numerator.index)
    return numerator / denominator.mask(denominator == 0)


def _first_existing(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for column in columns:
        if column in df.columns:
            values = values.fillna(pd.to_numeric(df[column], errors="coerce"))
    return values


def _extract_response_text(payload: dict[str, Any]) -> str | None:
    if "output_text" in payload:
        return str(payload["output_text"])
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                return str(content["text"])
    return None


def _choose_best_match(matches: pd.DataFrame) -> pd.Series:
    return matches.sort_values(
        ["value_percentile", "weekly_projection_index"],
        ascending=[False, False],
    ).iloc[0]


def _z_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (values - values.mean()) / std


def _clean_output_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in row.items():
        if isinstance(value, (list, tuple, set)):
            cleaned[key] = list(value)
        elif isinstance(value, dict):
            cleaned[key] = value
        elif pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, float):
            precision = 3 if key in THREE_DECIMAL_OUTPUT_COLUMNS else 2
            cleaned[key] = round(value, precision)
        else:
            cleaned[key] = value
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match roster text to weekly projections and summarize value."
    )
    parser.add_argument(
        "roster",
        nargs="*",
        help="Roster players. Use quotes or pipe text through stdin.",
    )
    args = parser.parse_args()

    roster_text = "\n".join(args.roster)
    if not roster_text:
        import sys

        roster_text = sys.stdin.read()

    result = analyze_roster(roster_text)
    print(pd.DataFrame(result["players"]).to_string(index=False))
    print("\nSummary")
    for key, value in result["summary"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
