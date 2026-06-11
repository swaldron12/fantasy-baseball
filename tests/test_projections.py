from __future__ import annotations

import pandas as pd

from src.projections import build_batter_weekly_projections, load_probable_pitchers
from src.schedules import convert_roster_resource_probables_grid


def test_load_probable_pitchers_standardizes_team_and_handedness(tmp_path) -> None:
    schedule_path = tmp_path / "probables.csv"
    pd.DataFrame(
        [
            {
                "week_start": "2026-05-18",
                "game_date": "2026-05-18",
                "team": "nyy",
                "opponent": "bos",
                "home_away": "Away",
                "opposing_probable_pitcher_name": "Example Pitcher",
                "opposing_probable_pitcher_id": "123",
                "opposing_probable_pitcher_throws": "left",
                "own_probable_pitcher_name": "Example Starter",
                "own_probable_pitcher_id": "456",
                "own_probable_pitcher_throws": "right",
            }
        ]
    ).to_csv(schedule_path, index=False)

    schedule = load_probable_pitchers(schedule_path)

    assert schedule.loc[0, "team"] == "NYY"
    assert schedule.loc[0, "opponent"] == "BOS"
    assert schedule.loc[0, "home_away"] == "away"
    assert schedule.loc[0, "opposing_probable_pitcher_throws"] == "L"
    assert schedule.loc[0, "own_probable_pitcher_throws"] == "R"


def test_batter_projection_uses_team_schedule_counts() -> None:
    batters = pd.DataFrame(
        [
            {
                "player_key": "1",
                "playerid": "1",
                "mlbamid": "11",
                "name": "Hitter A",
                "team": "NYY",
                "woba": 0.350,
                "xwoba": 0.360,
                "barrel_rate": 0.10,
                "hardhit_rate": 0.45,
                "k_rate": 0.20,
                "bb_rate": 0.10,
                "woba_vs_lhp": 0.390,
                "woba_vs_rhp": 0.340,
                "k_rate_vs_lhp": 0.18,
                "k_rate_vs_rhp": 0.21,
                "iso": 0.220,
                "pa": 100,
                "zips_woba": 0.345,
                "zips_hr_rate": 0.04,
            },
            {
                "player_key": "2",
                "playerid": "2",
                "mlbamid": "22",
                "name": "Hitter B",
                "team": "BOS",
                "woba": 0.310,
                "xwoba": 0.315,
                "barrel_rate": 0.05,
                "hardhit_rate": 0.34,
                "k_rate": 0.27,
                "bb_rate": 0.07,
                "woba_vs_lhp": 0.300,
                "woba_vs_rhp": 0.315,
                "k_rate_vs_lhp": 0.28,
                "k_rate_vs_rhp": 0.26,
                "iso": 0.130,
                "pa": 100,
                "zips_woba": 0.312,
                "zips_hr_rate": 0.02,
            },
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "week_start": "2026-05-18",
                "game_date": "2026-05-18",
                "team": "NYY",
                "opponent": "BOS",
                "home_away": "home",
                "opposing_probable_pitcher_name": "Pitcher One",
                "opposing_probable_pitcher_id": "1",
                "opposing_probable_pitcher_throws": "L",
                "own_probable_pitcher_name": "Starter One",
                "own_probable_pitcher_id": "10",
                "own_probable_pitcher_throws": "R",
            },
            {
                "week_start": "2026-05-18",
                "game_date": "2026-05-19",
                "team": "NYY",
                "opponent": "BOS",
                "home_away": "home",
                "opposing_probable_pitcher_name": "Pitcher Two",
                "opposing_probable_pitcher_id": "2",
                "opposing_probable_pitcher_throws": "R",
                "own_probable_pitcher_name": "Starter Two",
                "own_probable_pitcher_id": "20",
                "own_probable_pitcher_throws": "R",
            },
        ]
    )

    projections = build_batter_weekly_projections(batters, schedule)
    hitter_a = projections[projections["name"].eq("Hitter A")].iloc[0]
    hitter_b = projections[projections["name"].eq("Hitter B")].iloc[0]

    assert hitter_a["projected_games"] == 2
    assert hitter_a["games_vs_lhp"] == 1
    assert hitter_a["games_vs_rhp"] == 1
    assert hitter_b["projected_games"] == 0
    assert hitter_a["weekly_projection_index"] > hitter_b["weekly_projection_index"]


def test_batter_projection_excludes_zero_volume_non_hitters() -> None:
    batters = pd.DataFrame(
        [
            {
                "player_key": "1",
                "playerid": "1",
                "mlbamid": "11",
                "name": "Real Hitter",
                "team": "NYY",
                "woba": 0.350,
                "xwoba": 0.360,
                "barrel_rate": 0.10,
                "hardhit_rate": 0.45,
                "k_rate": 0.20,
                "bb_rate": 0.10,
                "woba_vs_lhp": 0.390,
                "woba_vs_rhp": 0.340,
                "k_rate_vs_lhp": 0.18,
                "k_rate_vs_rhp": 0.21,
                "iso": 0.220,
                "pa": 100,
                "zips_woba": 0.345,
                "zips_hr_rate": 0.04,
            },
            {
                "player_key": "2",
                "playerid": "2",
                "mlbamid": "22",
                "name": "Pitcher Batting Artifact",
                "team": "NYY",
                "woba": 0.0,
                "xwoba": pd.NA,
                "barrel_rate": pd.NA,
                "hardhit_rate": pd.NA,
                "k_rate": pd.NA,
                "bb_rate": pd.NA,
                "woba_vs_lhp": pd.NA,
                "woba_vs_rhp": pd.NA,
                "k_rate_vs_lhp": pd.NA,
                "k_rate_vs_rhp": pd.NA,
                "iso": 0.0,
                "pa": 0,
                "zips_woba": pd.NA,
                "zips_hr_rate": pd.NA,
            },
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "week_start": "2026-05-18",
                "game_date": "2026-05-18",
                "team": "NYY",
                "opponent": "BOS",
                "home_away": "home",
                "opposing_probable_pitcher_name": "Pitcher One",
                "opposing_probable_pitcher_id": "1",
                "opposing_probable_pitcher_throws": "R",
                "own_probable_pitcher_name": "Starter One",
                "own_probable_pitcher_id": "10",
                "own_probable_pitcher_throws": "R",
            }
        ]
    )

    projections = build_batter_weekly_projections(batters, schedule)

    assert projections["name"].tolist() == ["Real Hitter"]


def test_convert_roster_resource_probables_grid(tmp_path) -> None:
    excel_path = tmp_path / "probables.xlsx"
    output_path = tmp_path / "probable_pitchers_week.csv"
    pd.DataFrame(
        {
            "Team": ["NYY", "BOS"],
            "Mon 5/18": ["@ BOS\nCarlos Rodón (L)", "NYY\nBrayan Bello (R)"],
            "Tue 5/19": ["OFF", "OFF"],
        }
    ).to_excel(excel_path, index=False)

    schedule = convert_roster_resource_probables_grid(
        excel_path=excel_path,
        output_path=output_path,
        year=2026,
        days=1,
    )

    yankees = schedule[schedule["team"].eq("NYY")].iloc[0]
    assert output_path.exists()
    assert yankees["home_away"] == "away"
    assert yankees["own_probable_pitcher_name"] == "Carlos Rodón"
    assert yankees["own_probable_pitcher_throws"] == "L"
    assert yankees["opposing_probable_pitcher_name"] == "Brayan Bello"
    assert yankees["opposing_probable_pitcher_throws"] == "R"
