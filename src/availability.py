"""Player availability flags for lineup optimization.

The projection system can still estimate a player's talent, but the optimizer
needs a separate availability layer so injured or inactive players are not
treated as startable.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AVAILABILITY_PATH = PROJECT_ROOT / "data" / "processed" / "player_availability.csv"
MANUAL_AVAILABILITY_PATH = PROJECT_ROOT / "config" / "player_availability.csv"
MLB_API_ROOT = "https://statsapi.mlb.com/api/v1"

UNAVAILABLE_STATUS_TERMS = {
    "10-day il",
    "15-day il",
    "60-day il",
    "7-day il",
    "injured",
    "injured list",
    "il",
    "out",
    "suspended",
    "restricted",
    "bereavement",
    "paternity",
    "optioned",
    "minors",
}

STATUS_PRECEDENCE = {
    "manual": 3,
    "injuredlist": 2,
    "active": 1,
}


def load_availability_table(
    availability_path: Path = DEFAULT_AVAILABILITY_PATH,
    manual_path: Path = MANUAL_AVAILABILITY_PATH,
) -> pd.DataFrame:
    """Load cached and manual player availability rows."""
    pieces = []
    for path, default_source in [(availability_path, "mlb_stats_api"), (manual_path, "manual")]:
        if path.exists():
            table = pd.read_csv(path)
            if not table.empty:
                pieces.append(_standardize_availability_table(table, default_source))
    if not pieces:
        return _empty_availability_table()

    combined = pd.concat(pieces, ignore_index=True, sort=False)
    combined["status_priority"] = combined["source"].map(STATUS_PRECEDENCE).fillna(0)
    combined = combined.sort_values(["mlbamid", "name_key", "status_priority"])
    combined = combined.drop_duplicates(["mlbamid", "name_key"], keep="last")
    return combined.drop(columns=["status_priority"], errors="ignore")


def apply_availability_flags(projections: pd.DataFrame) -> pd.DataFrame:
    """Attach availability columns to projection rows."""
    if projections.empty:
        return projections

    work = projections.copy()
    status = load_availability_table()
    work["mlbamid"] = pd.to_numeric(work.get("mlbamid"), errors="coerce")
    if "name_key" not in work.columns:
        work["name_key"] = work["name"].map(_normalize_player_name)
    work["team"] = work["team"].astype("string").str.upper().str.strip()

    defaults = {
        "availability_status": "active_or_unknown",
        "availability_note": "",
        "availability_source": "projection_default",
        "availability_checked_at": "",
        "is_available": True,
    }
    if status.empty:
        for column, value in defaults.items():
            work[column] = value
        return work

    by_id = status.dropna(subset=["mlbamid"]).drop_duplicates("mlbamid")
    id_columns = [
        "mlbamid",
        "availability_status",
        "availability_note",
        "availability_source",
        "availability_checked_at",
        "is_available",
    ]
    work = work.merge(by_id[id_columns], on="mlbamid", how="left")

    missing_status = work["availability_status"].isna()
    if missing_status.any():
        by_name = status.dropna(subset=["name_key"]).drop_duplicates(["name_key", "team"])
        name_columns = [
            "name_key",
            "team",
            "availability_status",
            "availability_note",
            "availability_source",
            "availability_checked_at",
            "is_available",
        ]
        name_match = work.loc[missing_status, ["name_key", "team"]].merge(
            by_name[name_columns],
            on=["name_key", "team"],
            how="left",
        )
        for column in defaults:
            work.loc[missing_status, column] = name_match[column].to_numpy()

    for column, value in defaults.items():
        if column == "is_available":
            work[column] = work[column].map(_coerce_bool)
        else:
            work[column] = work[column].fillna(value)
    return work


def refresh_mlb_availability(
    season: int | None = None,
    output_path: Path = DEFAULT_AVAILABILITY_PATH,
) -> pd.DataFrame:
    """Fetch MLB active and injured-list roster flags and cache them."""
    season = season or datetime.now(UTC).year
    checked_at = datetime.now(UTC).isoformat(timespec="seconds")
    teams_payload = _get_json(f"{MLB_API_ROOT}/teams?sportId=1&season={season}&activeStatus=Y")
    teams = teams_payload.get("teams", [])

    rows = []
    for team in teams:
        team_id = team.get("id")
        team_abbr = str(team.get("abbreviation") or "").upper()
        if not team_id:
            continue
        try:
            roster = _get_json(
                f"{MLB_API_ROOT}/teams/{team_id}/roster?rosterType=40Man"
            ).get("roster", [])
        except Exception:
            continue
        for item in roster:
            person = item.get("person", {})
            status = item.get("status", {}) or {}
            status_text = status.get("description") or status.get("code") or ""
            rows.append(
                {
                    "mlbamid": person.get("id"),
                    "name": person.get("fullName"),
                    "team": team_abbr,
                    "availability_status": status_text,
                    "is_available": _status_is_available(status_text),
                    "availability_note": "MLB rosterType=40Man",
                    "availability_source": "40Man",
                    "availability_checked_at": checked_at,
                }
            )

    table = _standardize_availability_table(pd.DataFrame(rows), "mlb_stats_api")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def _standardize_availability_table(table: pd.DataFrame, default_source: str) -> pd.DataFrame:
    work = table.copy()
    if "status" in work.columns and "availability_status" not in work.columns:
        work = work.rename(columns={"status": "availability_status"})
    if "available" in work.columns and "is_available" not in work.columns:
        work = work.rename(columns={"available": "is_available"})
    for column in ["mlbamid", "name", "team", "availability_status"]:
        if column not in work.columns:
            work[column] = pd.NA
    work["mlbamid"] = pd.to_numeric(work["mlbamid"], errors="coerce")
    work["name"] = work["name"].astype("string").fillna("")
    work["name_key"] = work["name"].map(_normalize_player_name)
    work["team"] = work["team"].astype("string").str.upper().str.strip()
    work["availability_status"] = work["availability_status"].astype("string").fillna("")
    if "is_available" not in work.columns:
        work["is_available"] = work["availability_status"].map(_status_is_available)
    else:
        work["is_available"] = work["is_available"].map(_coerce_bool)
    if "availability_note" not in work.columns:
        work["availability_note"] = ""
    if "availability_source" not in work.columns:
        work["availability_source"] = default_source
    if "availability_checked_at" not in work.columns:
        work["availability_checked_at"] = ""
    work["source"] = work["availability_source"].astype("string").str.lower()
    return work[
        [
            "mlbamid",
            "name",
            "name_key",
            "team",
            "availability_status",
            "is_available",
            "availability_note",
            "availability_source",
            "availability_checked_at",
            "source",
        ]
    ]


def _normalize_player_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _empty_availability_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "mlbamid",
            "name",
            "name_key",
            "team",
            "availability_status",
            "is_available",
            "availability_note",
            "availability_source",
            "availability_checked_at",
            "source",
        ]
    )


def _status_is_available(value: object) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return True
    return not any(term in normalized for term in UNAVAILABLE_STATUS_TERMS)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"false", "0", "no", "n", "unavailable", "out"}:
        return False
    if normalized in {"true", "1", "yes", "y", "available", "active"}:
        return True
    return _status_is_available(value)


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh MLB player availability flags.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_AVAILABILITY_PATH)
    args = parser.parse_args()
    table = refresh_mlb_availability(season=args.season, output_path=args.output)
    unavailable = int((~table["is_available"]).sum()) if not table.empty else 0
    print(f"Wrote {len(table)} availability rows to {args.output}")
    print(f"Unavailable players: {unavailable}")


if __name__ == "__main__":
    main()
