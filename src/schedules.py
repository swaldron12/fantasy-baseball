"""Convert schedule/probable-pitcher files into the project schedule format."""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.projections import DEFAULT_SCHEDULE_PATH, SCHEDULE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_DIR = PROJECT_ROOT / "data" / "raw" / "schedules"
DEFAULT_ROSTER_RESOURCE_PATH = (
    SCHEDULE_DIR / "roster-resource__probables-grid.xlsx.xlsx"
)

DATE_HEADER_PATTERN = re.compile(r"(?P<month>\d{1,2})/(?P<day>\d{1,2})")
PITCHER_PATTERN = re.compile(r"(?P<name>.+?)\s*\((?P<throws>[LR])\)\s*$")


def convert_roster_resource_probables_grid(
    excel_path: Path = DEFAULT_ROSTER_RESOURCE_PATH,
    output_path: Path = DEFAULT_SCHEDULE_PATH,
    year: int | None = None,
    days: int = 7,
) -> pd.DataFrame:
    """Convert a Roster Resource probables grid into team-game schedule rows."""
    if year is None:
        year = date.today().year

    grid = pd.read_excel(excel_path, sheet_name=0, dtype="string")
    if "Team" not in grid.columns:
        raise ValueError("Roster Resource grid must include a Team column.")

    date_columns = [column for column in grid.columns if column != "Team"]
    if days:
        date_columns = date_columns[:days]

    starts = []
    for _, row in grid.iterrows():
        team = _clean_team(row["Team"])
        if not team:
            continue

        for column in date_columns:
            game_date = _parse_date_header(str(column), year)
            parsed_cell = _parse_probables_cell(row[column])
            if parsed_cell is None:
                continue

            starts.append(
                {
                    "week_start": _parse_date_header(str(date_columns[0]), year),
                    "game_date": game_date,
                    "team": team,
                    "opponent": parsed_cell["opponent"],
                    "home_away": parsed_cell["home_away"],
                    "own_probable_pitcher_name": parsed_cell["pitcher_name"],
                    "own_probable_pitcher_id": pd.NA,
                    "own_probable_pitcher_throws": parsed_cell["pitcher_throws"],
                    "own_probable_pitcher_role": parsed_cell["pitcher_role"],
                }
            )

    starts_df = pd.DataFrame(starts)
    if starts_df.empty:
        raise ValueError(f"No probable pitcher rows parsed from {excel_path}")

    opponent_starts = starts_df[
        [
            "game_date",
            "team",
            "opponent",
            "own_probable_pitcher_name",
            "own_probable_pitcher_id",
            "own_probable_pitcher_throws",
        ]
    ].rename(
        columns={
            "team": "opponent",
            "opponent": "team",
            "own_probable_pitcher_name": "opposing_probable_pitcher_name",
            "own_probable_pitcher_id": "opposing_probable_pitcher_id",
            "own_probable_pitcher_throws": "opposing_probable_pitcher_throws",
        }
    )

    schedule = starts_df.merge(
        opponent_starts,
        on=["game_date", "team", "opponent"],
        how="left",
    )
    schedule = schedule[
        SCHEDULE_COLUMNS + ["own_probable_pitcher_role"]
    ].sort_values(["game_date", "team"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(output_path, index=False)
    print(f"Converted schedule: {output_path}")
    print(f"Rows: {len(schedule)}")
    print(f"Dates: {schedule['game_date'].min()} to {schedule['game_date'].max()}")
    return schedule


def _parse_date_header(header: str, year: int) -> str:
    match = DATE_HEADER_PATTERN.search(header)
    if not match:
        raise ValueError(f"Could not parse date from column header: {header}")
    month = int(match.group("month"))
    day = int(match.group("day"))
    return date(year, month, day).isoformat()


def _parse_probables_cell(value: object) -> dict[str, str] | None:
    if pd.isna(value):
        return None

    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    if not lines or lines[0].upper() == "OFF":
        return None

    opponent_text = lines[0]
    home_away = "away" if opponent_text.startswith("@") else "home"
    opponent = opponent_text.replace("@", "").strip().upper()

    pitcher_name = pd.NA
    pitcher_throws = pd.NA
    pitcher_role = "starter"

    pitcher_lines = lines[1:]
    primary_line = _find_role_line(pitcher_lines, "Primary:")
    opener_line = _find_role_line(pitcher_lines, "Opener:")

    if primary_line:
        pitcher_role = "primary"
        pitcher_line = primary_line
    elif opener_line:
        pitcher_role = "opener"
        pitcher_line = opener_line
    elif pitcher_lines:
        pitcher_line = pitcher_lines[0]
    else:
        pitcher_line = ""

    if pitcher_line:
        pitcher_line = re.sub(r"^(Opener|Primary):\s*", "", pitcher_line).strip()
        match = PITCHER_PATTERN.match(pitcher_line)
        if match:
            pitcher_name = match.group("name").strip()
            pitcher_throws = match.group("throws").upper()
        else:
            pitcher_name = pitcher_line

    return {
        "opponent": opponent,
        "home_away": home_away,
        "pitcher_name": pitcher_name,
        "pitcher_throws": pitcher_throws,
        "pitcher_role": pitcher_role,
    }


def _find_role_line(lines: list[str], role_prefix: str) -> str | None:
    for line in lines:
        if line.startswith(role_prefix):
            return line
    return None


def _clean_team(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Roster Resource probables grid to project schedule CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_ROSTER_RESOURCE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--year", type=int, default=date.today().year)
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of date columns to convert. Use 0 to include all dates.",
    )
    args = parser.parse_args()

    convert_roster_resource_probables_grid(
        excel_path=args.input,
        output_path=args.output,
        year=args.year,
        days=args.days,
    )


if __name__ == "__main__":
    main()
