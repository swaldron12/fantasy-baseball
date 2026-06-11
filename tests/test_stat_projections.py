from __future__ import annotations

import pandas as pd
import pytest

from src.stat_projections import add_batter_hybrid_stats


def test_batter_hybrid_stats_derives_avg_and_ops_from_hit_components(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    features_dir = tmp_path / "features"
    processed_dir.mkdir()
    features_dir.mkdir()

    pd.DataFrame(
        [
            {
                "mlbamid": 123,
                "g": 10,
                "ab": 40,
                "h": 10,
                "1b": 6,
                "2b": 2,
                "3b": 1,
                "hr": 1,
                "r": 7,
                "rbi": 8,
                "bb": 5,
                "hbp": 1,
                "sf": 1,
                "sb": 2,
                "woba": 0.360,
            }
        ]
    ).to_parquet(processed_dir / "zips_batters_ros.parquet", index=False)

    batter_projections = pd.DataFrame(
        [
            {
                "player_type": "batter",
                "player_key": "123",
                "playerid": "123",
                "mlbamid": 123,
                "name": "Formula Hitter",
                "team": "NYY",
                "projected_games": 1,
                "weekly_projection_index": 50,
                "overall_woba_signal": 0.360,
                "matchup_woba_signal": 0.360,
            }
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "team": "NYY",
                "opposing_probable_pitcher_id": pd.NA,
            }
        ]
    )

    result = add_batter_hybrid_stats(
        batter_projections,
        schedule,
        processed_dir=processed_dir,
        features_dir=features_dir,
    ).iloc[0]

    total_bases = result["1b"] + 2 * result["2b"] + 3 * result["3b"] + 4 * result["hr"]
    avg = result["h"] / result["ab"]
    obp_denominator = result["ab"] + result["bb"] + result["hbp"] + result["sf"]
    obp = (result["h"] + result["bb"] + result["hbp"]) / obp_denominator
    slg = total_bases / result["ab"]

    assert result["total_bases"] == pytest.approx(total_bases)
    assert result["avg"] == pytest.approx(avg)
    assert result["ops"] == pytest.approx(round(obp + slg, 3))


def test_batter_hybrid_stats_regresses_avg_toward_zips(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    features_dir = tmp_path / "features"
    processed_dir.mkdir()
    features_dir.mkdir()

    pd.DataFrame(
        [
            {
                "mlbamid": 321,
                "g": 100,
                "ab": 400,
                "h": 100,
                "1b": 70,
                "2b": 20,
                "3b": 2,
                "hr": 8,
                "r": 60,
                "rbi": 65,
                "bb": 50,
                "hbp": 3,
                "sf": 3,
                "sb": 4,
                "woba": 0.330,
            }
        ]
    ).to_parquet(processed_dir / "zips_batters_ros.parquet", index=False)
    pd.DataFrame(
        [
            {
                "mlbamid": 321,
                "g": 25,
                "ab": 100,
                "h": 40,
                "1b": 25,
                "2b": 9,
                "3b": 1,
                "hr": 5,
                "r": 20,
                "rbi": 22,
                "bb": 15,
                "hbp": 1,
                "sf": 1,
                "sb": 2,
            }
        ]
    ).to_parquet(processed_dir / "batters_standard_05_06.parquet", index=False)
    pd.DataFrame(
        [
            {
                "mlbamid": 321,
                "week_start": "2026-05-18",
                "pa": 10,
                "hits": 8,
                "hr": 2,
                "xwoba": 0.600,
                "woba": 0.700,
                "bb_rate": 0.100,
            }
        ]
    ).to_parquet(features_dir / "weekly_batter_savant_features.parquet", index=False)

    batter_projections = pd.DataFrame(
        [
            {
                "player_type": "batter",
                "player_key": "321",
                "playerid": "321",
                "mlbamid": 321,
                "name": "Hot Sample Hitter",
                "team": "NYY",
                "projected_games": 1,
                "weekly_projection_index": 50,
                "overall_woba_signal": 0.330,
                "matchup_woba_signal": 0.430,
            }
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "team": "NYY",
                "opposing_probable_pitcher_id": pd.NA,
            }
        ]
    )

    result = add_batter_hybrid_stats(
        batter_projections,
        schedule,
        processed_dir=processed_dir,
        features_dir=features_dir,
    ).iloc[0]

    assert result["avg"] <= 0.295


def test_batter_hybrid_stats_adds_relevant_handedness_matchup(
    tmp_path,
    monkeypatch,
) -> None:
    processed_dir = tmp_path / "processed"
    features_dir = tmp_path / "features"
    statcast_dir = tmp_path / "statcast"
    processed_dir.mkdir()
    features_dir.mkdir()
    statcast_dir.mkdir()
    monkeypatch.setattr("src.stat_projections.STATCAST_RAW_DIR", statcast_dir)

    pd.DataFrame(
        [
            {"batter": 456, "stand": "R", "p_throws": "L"},
            {"batter": 456, "stand": "R", "p_throws": "R"},
            {"batter": 456, "stand": "R", "p_throws": "R"},
        ]
    ).to_parquet(statcast_dir / "statcast_fixture.parquet", index=False)

    pd.DataFrame(
        [
            {
                "mlbamid": 456,
                "g": 100,
                "ab": 400,
                "h": 110,
                "1b": 70,
                "2b": 25,
                "3b": 2,
                "hr": 13,
                "r": 70,
                "rbi": 75,
                "bb": 50,
                "hbp": 3,
                "sf": 3,
                "sb": 5,
                "woba": 0.360,
            }
        ]
    ).to_parquet(processed_dir / "zips_batters_ros.parquet", index=False)
    pd.DataFrame(
        [
            {
                "name": "Favorable Pitcher",
                "mlbamid": 999,
                "era": 5.25,
                "whip": 1.45,
                "k_pct": 0.16,
            }
        ]
    ).to_parquet(processed_dir / "zips_pitchers_ros.parquet", index=False)
    pd.DataFrame(
        [
            {
                "mlbamid": 999,
                "week_start": "2026-06-01",
                "woba_allowed_prev_4w": 0.360,
                "woba_allowed_vs_lhb_prev_4w": 0.210,
                "woba_allowed_vs_rhb_prev_4w": 0.390,
            }
        ]
    ).to_parquet(features_dir / "weekly_pitcher_savant_features.parquet", index=False)

    batter_projections = pd.DataFrame(
        [
            {
                "player_type": "batter",
                "player_key": "456",
                "playerid": "456",
                "mlbamid": 456,
                "name": "Matchup Hitter",
                "team": "NYY",
                "projected_games": 1,
                "games_vs_lhp": 0,
                "games_vs_rhp": 1,
                "weekly_projection_index": 50,
                "overall_woba_signal": 0.360,
                "matchup_woba_signal": 0.390,
            }
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "team": "NYY",
                "opposing_probable_pitcher_name": "Favorable Pitchér",
                "opposing_probable_pitcher_id": pd.NA,
                "opposing_probable_pitcher_throws": "R",
            }
        ]
    )

    result = add_batter_hybrid_stats(
        batter_projections,
        schedule,
        processed_dir=processed_dir,
        features_dir=features_dir,
    ).iloc[0]

    assert result["batter_bats"] == "R"
    assert "bats R" in result["matchup_context"]
    assert "Favorable Pitchér (R)" in result["opposing_pitcher_summary"]
    assert "Favorable Pitchér (R)" in result["favorable_batter_matchups"]
    assert "relevant split .390 wOBA allowed vs RHB" in result["favorable_batter_matchups"]
    assert "ZiPS 5.25 ERA/1.45 WHIP" in result["favorable_batter_matchups"]
    assert "Favorable Pitchér (R)" in result["matchup_context"]
    assert "vs LHB" not in result["matchup_context"]
