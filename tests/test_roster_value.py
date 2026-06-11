from __future__ import annotations

import pandas as pd

from src.roster_value import (
    HITTER_SLOT_LABELS,
    HITTER_SLOTS,
    _analysis_prompt_payload,
    _chat_prompt_payload,
    _clean_output_row,
    _with_verified_availability_evidence,
    _with_verified_matchup_evidence,
    aggregate_batting_categories,
    aggregate_pitching_categories,
    analyze_roster,
    generate_local_projection_chat_response,
    normalize_player_name,
    optimize_lineup,
    parse_roster_text,
    projection_metadata,
)


def test_normalize_player_name_handles_accents_and_suffixes() -> None:
    assert normalize_player_name("Ronald Acuña Jr.") == "ronald acuna"


def test_parse_roster_text_accepts_line_and_type_hint() -> None:
    entries = parse_roster_text("Aaron Judge\nNolan McLean, SP\nMike Trout")

    assert [entry.player_name for entry in entries] == [
        "Aaron Judge",
        "Nolan McLean",
        "Mike Trout",
    ]
    assert entries[1].requested_type == "pitcher"


def test_projection_metadata_explains_value_fields() -> None:
    metadata = projection_metadata()

    assert "value_explanation" in metadata
    assert "current_value" in metadata["value_explanation"]
    assert "optimized_value" in metadata["value_explanation"]
    assert "value_gain" in metadata["value_explanation"]


def test_analyze_roster_matches_players_and_returns_summary(tmp_path) -> None:
    projections_dir = tmp_path / "projections"
    schedule_dir = tmp_path / "schedules"
    projections_dir.mkdir()
    schedule_dir.mkdir()

    pd.DataFrame(
        [
            {
                "player_type": "batter",
                "player_key": "1",
                "playerid": "1",
                "mlbamid": "11",
                "name": "Aaron Judge",
                "team": "NYY",
                "projected_games": 6,
                "games_vs_lhp": 2,
                "games_vs_rhp": 4,
                "skill_score": 95,
                "weekly_projection_index": 95,
                "projection_rank": 1,
            },
            {
                "player_type": "batter",
                "player_key": "2",
                "playerid": "2",
                "mlbamid": "22",
                "name": "Bench Hitter",
                "team": "BOS",
                "projected_games": 6,
                "games_vs_lhp": 2,
                "games_vs_rhp": 4,
                "skill_score": 30,
                "weekly_projection_index": 30,
                "projection_rank": 2,
            },
        ]
    ).to_csv(projections_dir / "weekly_batter_projections.csv", index=False)

    pd.DataFrame(
        [
            {
                "player_type": "pitcher",
                "player_key": "3",
                "playerid": "3",
                "mlbamid": "33",
                "name": "Nolan McLean",
                "team": "NYM",
                "projected_starts": 2,
                "opponents": "WSN, MIA",
                "skill_score": 88,
                "matchup_score": 55,
                "weekly_projection_index": 160,
                "projection_rank": 1,
            },
            {
                "player_type": "pitcher",
                "player_key": "4",
                "playerid": "4",
                "mlbamid": "44",
                "name": "Low Pitcher",
                "team": "SEA",
                "projected_starts": 1,
                "opponents": "TEX",
                "skill_score": 20,
                "matchup_score": 30,
                "weekly_projection_index": 40,
                "projection_rank": 2,
            },
        ]
    ).to_csv(projections_dir / "weekly_pitcher_projections.csv", index=False)

    schedule_path = schedule_dir / "probable_pitchers_week.csv"
    pd.DataFrame(
        [
            {"game_date": "2026-05-18"},
            {"game_date": "2026-05-24"},
        ]
    ).to_csv(schedule_path, index=False)

    result = analyze_roster(
        "Aaron Judge\nNolan McLean, SP\nFake Player",
        projections_dir=projections_dir,
        schedule_path=schedule_path,
    )

    assert result["summary"]["matched_players"] == 2
    assert result["summary"]["unmatched_players"] == 1
    assert result["summary"]["total_expected_fantasy_value"] > 0
    assert result["metadata"]["schedule_start"] == "2026-05-18"
    assert result["players"][0]["name"] == "Aaron Judge"
    assert result["unmatched"][0]["input"] == "Fake Player"


def test_category_aggregates_recompute_rate_stats_from_totals() -> None:
    batting = aggregate_batting_categories(
        [
            {
                "ab": 10,
                "h": 3,
                "bb": 1,
                "hbp": 0,
                "sf": 0,
                "total_bases": 5,
                "hr": 1,
                "r": 2,
                "rbi": 3,
                "sb": 0,
            },
            {
                "ab": 20,
                "h": 6,
                "bb": 2,
                "hbp": 0,
                "sf": 0,
                "total_bases": 10,
                "hr": 1,
                "r": 4,
                "rbi": 5,
                "sb": 2,
            },
        ]
    )
    pitching = aggregate_pitching_categories(
        [
            {"ip": 5, "er": 2, "baserunners": 7, "w": 1, "sv": 0, "hld": 0, "k": 6},
            {"ip": 4, "er": 0, "baserunners": 3, "w": 0, "sv": 1, "hld": 0, "k": 5},
        ]
    )

    assert batting["avg"] == 0.3
    assert batting["ops"] == 0.864
    assert batting["hr"] == 2
    assert pitching["era"] == 2.0
    assert pitching["whip"] == 1.11
    assert pitching["k"] == 11


def test_clean_output_row_preserves_stat_precision() -> None:
    cleaned = _clean_output_row(
        {
            "name": "Matt Olson",
            "avg": 0.277,
            "ops": 0.919,
            "ab": 26.666,
            "expected_fantasy_value": 2.334,
        }
    )

    assert cleaned["avg"] == 0.277
    assert cleaned["ops"] == 0.919
    assert cleaned["ab"] == 26.666
    assert cleaned["expected_fantasy_value"] == 2.33


def test_analysis_prompt_payload_includes_hitter_matchup_context() -> None:
    player = {
        "name": "Aaron Judge",
        "team": "NYY",
        "player_type": "batter",
        "expected_fantasy_value": 2.5,
        "weekly_projection_index": 94.3,
        "avg": 0.287,
        "ops": 1.025,
        "matchup_context": (
            "7 games: 2 vs LHP, 5 vs RHP; bats R; favorable relevant matchups: "
            "Example Pitcher (R); relevant split .390 wOBA allowed vs RHB"
        ),
        "matchup_edge_pct": 4.3,
        "batter_side_note": "bats R",
        "batter_bats": "R",
        "stand_vs_lhp": "R",
        "stand_vs_rhp": "R",
        "games_vs_lhp": 2,
        "games_vs_rhp": 5,
        "relevant_opposing_pitcher_summary": (
            "Example Pitcher (R); relevant split .390 wOBA allowed vs RHB"
        ),
        "favorable_batter_matchups": (
            "Example Pitcher (R); relevant split .390 wOBA allowed vs RHB"
        ),
    }
    payload = _analysis_prompt_payload(
        current_summary={},
        optimized={"summary": {}, "lineup": [player], "bench": [], "changes": []},
        matched_rows=[player],
    )

    matchup = payload["top_optimized_players"][0]["matchup"]
    assert matchup["context"] == player["matchup_context"]
    assert matchup["batter_side"] == "bats R"
    assert matchup["favorable_relevant_matchups"] == (
        "Example Pitcher (R); relevant split .390 wOBA allowed vs RHB"
    )
    assert payload["verified_matchup_evidence"]
    assert "expected_fantasy_value" in payload["value_explanation"]


def test_chat_prompt_payload_includes_projection_context_and_limits_history() -> None:
    player = {
        "name": "Aaron Judge",
        "player_type": "batter",
        "expected_fantasy_value": 2.5,
        "weekly_projection_index": 94.3,
        "hr": 2.1,
        "rbi": 6.2,
        "avg": 0.281,
        "ops": 0.944,
        "matchup_context": "Aaron Judge has a +19.0% handedness/schedule edge.",
    }
    optimized = {
        "summary": {"total_expected_fantasy_value": 4.0},
        "lineup": [player],
        "bench": [],
        "changes": [],
        "assumption": "Best legal lineup by expected fantasy value.",
    }
    history = [{"role": "user", "text": f"question {number}"} for number in range(8)]

    payload = _chat_prompt_payload(
        "Why is Judge strong this week?",
        history,
        {"total_expected_fantasy_value": 4.0},
        optimized,
        [player],
        [],
    )

    assert payload["user_question"] == "Why is Judge strong this week?"
    assert len(payload["conversation_history"]) == 6
    assert payload["conversation_history"][0]["text"] == "question 2"
    assert payload["optimized_lineup"][0]["name"] == "Aaron Judge"
    assert "verified matchup" in payload["limitations"].lower()


def test_local_projection_chat_response_summarizes_value_and_changes() -> None:
    optimized = {
        "summary": {"total_expected_fantasy_value": 3.5},
        "lineup": [
            {
                "name": "Josh Jung",
                "player_type": "batter",
                "expected_fantasy_value": 1.8,
                "weekly_projection_index": 80,
                "hr": 1.0,
                "rbi": 4.0,
                "sb": 0.1,
                "avg": 0.270,
                "ops": 0.820,
            }
        ],
        "bench": [],
        "changes": [
            {
                "start": "Josh Jung",
                "sit": "Michael Busch",
                "slot": "3B",
                "value_gain": 0.4,
            }
        ],
    }

    text = generate_local_projection_chat_response(
        "Who should I start?",
        {"total_expected_fantasy_value": 3.1},
        optimized,
        optimized["lineup"],
        [],
    )

    assert "optimized lineup value is 3.50" in text
    assert "Start Josh Jung over Michael Busch at 3B" in text
    assert "Strong projected starters" in text


def test_verified_availability_evidence_mentions_last_name_question() -> None:
    optimized = {
        "unavailable": [
            {
                "name": "Aaron Judge",
                "availability_status": "Injured 10-Day",
                "availability_note": "MLB rosterType=40Man",
            }
        ]
    }
    bad_text = "Devers is preferred because of matchup context."

    fixed = _with_verified_availability_evidence(
        bad_text,
        optimized,
        question="Why is Devers preferred over Judge?",
    )

    assert fixed.startswith("Availability notes:")
    assert "Aaron Judge is marked Injured 10-Day" in fixed
    assert "Devers is preferred" in fixed


def test_verified_matchup_evidence_replaces_llm_mixed_pitchers() -> None:
    bad_analysis = (
        "Lineup verdict:\nThe lineup is strong.\n\n"
        "Matchup evidence:\nAaron Judge faces Roki Sasaki and Yoshinobu Yamamoto.\n\n"
        "Category read:\nPower looks good."
    )
    optimized = {
        "lineup": [
            {
                "name": "Aaron Judge",
                "player_type": "batter",
                "expected_fantasy_value": 2.5,
                "matchup_context": (
                    "5 games: 2 vs LHP, 3 vs RHP; bats R; favorable relevant matchups: "
                    "Slade Cecconi (R); relevant split .436 wOBA allowed vs RHB"
                ),
                "matchup_edge_pct": 5.5,
                "favorable_batter_matchups": (
                    "Slade Cecconi (R); relevant split .436 wOBA allowed vs RHB"
                ),
            }
        ]
    }

    fixed = _with_verified_matchup_evidence(bad_analysis, optimized)

    assert "Aaron Judge faces Roki Sasaki" not in fixed
    assert "Slade Cecconi (R)" in fixed
    assert "Category read:" in fixed


def test_analysis_prompt_payload_includes_start_sit_decision_details() -> None:
    start = {
        "name": "Bench Bat",
        "team": "TST",
        "player_type": "batter",
        "lineup_status": "bench",
        "expected_fantasy_value": 1.5,
        "weekly_projection_index": 88,
        "hr": 1.8,
        "rbi": 5.2,
        "sb": 0.4,
        "avg": 0.284,
        "ops": 0.862,
    }
    sit = {
        "name": "Active Bat",
        "team": "TST",
        "player_type": "batter",
        "lineup_status": "active",
        "expected_fantasy_value": 0.2,
        "weekly_projection_index": 71,
        "hr": 0.8,
        "rbi": 3.1,
        "sb": 0.2,
        "avg": 0.251,
        "ops": 0.731,
    }

    payload = _analysis_prompt_payload(
        current_summary={},
        optimized={
            "summary": {},
            "lineup": [start],
            "bench": [sit],
            "changes": [
                {
                    "slot": "OF",
                    "player_type": "batter",
                    "start": "Bench Bat",
                    "sit": "Active Bat",
                    "value_gain": 1.3,
                }
            ],
        },
        matched_rows=[start, sit],
    )

    decision = payload["lineup_decisions"][0]
    assert decision["value_gain"] == 1.3
    assert decision["start"]["key_stats"]["ops"] == 0.862
    assert decision["sit"]["key_stats"]["avg"] == 0.251


def test_optimizer_promotes_best_bench_hitter_when_lineup_is_full() -> None:
    active_hitters = []
    for number, slot_id in enumerate(HITTER_SLOTS, start=1):
        active_hitters.append(
            {
                "name": f"Active Hitter {number}",
                "team": "TST",
                "player_type": "batter",
                "player_key": str(number),
                "name_key": f"active hitter {number}",
                "lineup_status": "active",
                "slot_id": slot_id,
                "slot_label": HITTER_SLOT_LABELS.get(slot_id, slot_id),
                "expected_fantasy_value": 11 - number,
                "weekly_projection_index": 100 - number,
            "ab": 20,
            "h": 5,
            "hr": 1,
            "r": 3,
            "rbi": 3,
            "sb": 0,
            "bb": 2,
            "hbp": 0,
                "sf": 0,
                "total_bases": 9,
            }
        )
    bench_hitter = {
        "name": "Bench Star",
        "team": "TST",
        "player_type": "batter",
        "player_key": "99",
        "name_key": "bench star",
        "lineup_status": "bench",
        "expected_fantasy_value": 12,
        "weekly_projection_index": 120,
        "ab": 22,
        "h": 8,
        "hr": 3,
        "r": 5,
        "rbi": 7,
        "sb": 1,
        "bb": 3,
        "hbp": 0,
        "sf": 0,
        "total_bases": 18,
    }

    optimized = optimize_lineup(active_hitters + [bench_hitter])
    optimized_names = {player["name"] for player in optimized["lineup"]}

    assert "Bench Star" in optimized_names
    assert f"Active Hitter {len(HITTER_SLOTS)}" not in optimized_names
    assert optimized["changes"][0]["start"] == "Bench Star"
    assert optimized["changes"][0]["sit"] == f"Active Hitter {len(HITTER_SLOTS)}"
    assert optimized["summary"]["active_players"] == len(HITTER_SLOTS)


def test_optimizer_does_not_pair_bench_hitter_with_pitcher_sit() -> None:
    active_pitchers = [
        {
            "name": f"Active Pitcher {number}",
            "team": "TST",
            "player_type": "pitcher",
            "player_key": str(number),
            "name_key": f"active pitcher {number}",
            "lineup_status": "active",
            "expected_fantasy_value": 9 - number,
            "weekly_projection_index": 90 - number,
            "ip": 5,
            "w": 0,
            "sv": 0,
            "hld": 0,
            "k": 5,
            "er": 2,
            "baserunners": 6,
        }
        for number in range(1, 9)
    ]
    bench_hitter = {
        "name": "Bench Hitter",
        "team": "TST",
        "player_type": "batter",
        "player_key": "99",
        "name_key": "bench hitter",
        "lineup_status": "bench",
        "expected_fantasy_value": 5,
        "weekly_projection_index": 80,
        "ab": 20,
        "h": 6,
        "hr": 2,
        "r": 4,
        "rbi": 5,
        "sb": 1,
        "bb": 2,
        "hbp": 0,
        "sf": 0,
        "total_bases": 14,
    }

    optimized = optimize_lineup(active_pitchers + [bench_hitter])
    hitter_change = next(
        change for change in optimized["changes"] if change["start"] == "Bench Hitter"
    )

    assert hitter_change["player_type"] == "batter"
    assert hitter_change["sit"] is None


def test_optimizer_ignores_pitcher_slot_reordering() -> None:
    pitchers = [
        {
            "name": "Dylan Cease",
            "team": "TST",
            "player_type": "pitcher",
            "player_key": "cease",
            "name_key": "dylan cease",
            "lineup_status": "active",
            "slot_id": "P5",
            "slot_label": "P5",
            "expected_fantasy_value": 3.0,
            "weekly_projection_index": 90,
            "ip": 6,
            "w": 0.5,
            "sv": 0,
            "hld": 0,
            "k": 8,
            "er": 2,
            "baserunners": 7,
        },
        {
            "name": "Parker Messick",
            "team": "TST",
            "player_type": "pitcher",
            "player_key": "messick",
            "name_key": "parker messick",
            "lineup_status": "active",
            "slot_id": "P3",
            "slot_label": "P3",
            "expected_fantasy_value": 2.8,
            "weekly_projection_index": 88,
            "ip": 6,
            "w": 0.5,
            "sv": 0,
            "hld": 0,
            "k": 7,
            "er": 2,
            "baserunners": 7,
        },
    ]

    optimized = optimize_lineup(pitchers)

    assert optimized["changes"] == []
    optimized_by_name = {player["name"]: player for player in optimized["lineup"]}
    assert optimized_by_name["Dylan Cease"]["optimized_slot"] == "P5"
    assert optimized_by_name["Parker Messick"]["optimized_slot"] == "P3"


def test_optimizer_respects_hitter_position_eligibility() -> None:
    def hitter(
        name: str,
        slot_id: str,
        value: float,
        positions: str | None = None,
    ) -> dict[str, object]:
        return {
            "name": name,
            "team": "TST",
            "player_type": "batter",
            "player_key": name.lower().replace(" ", "-"),
            "name_key": name.lower(),
            "lineup_status": "active",
            "slot_id": slot_id,
            "slot_label": HITTER_SLOT_LABELS.get(slot_id, slot_id),
            "eligible_positions": positions or "",
            "expected_fantasy_value": value,
            "weekly_projection_index": value * 10,
            "ab": 20,
            "h": 5,
            "hr": 1,
            "r": 3,
            "rbi": 3,
            "sb": 0,
            "bb": 2,
            "hbp": 0,
            "sf": 0,
            "total_bases": 9,
        }

    active_hitters = [
        hitter("Roster Catcher", "C", -5),
        hitter("Roster First Baseman", "1B", 0),
        hitter("Jazz Chisholm", "2B", 1, "2B"),
        hitter("Roster Third Baseman", "3B", 0),
        hitter("Roster Shortstop", "SS", 0),
        hitter("Aaron Judge", "OF1", 8, "OF"),
        hitter("Roster Outfielder Two", "OF2", 0, "OF"),
        hitter("Low Outfielder", "OF3", -2, "OF"),
        hitter("Utility Hitter", "UTIL", -1),
    ]
    bench_springer = {
        "name": "George Springer",
        "team": "TOR",
        "player_type": "batter",
        "player_key": "george-springer",
        "name_key": "george springer",
        "lineup_status": "bench",
        "slot_id": "BN1",
        "slot_label": "BN1",
        "eligible_positions": "OF",
        "expected_fantasy_value": 5,
        "weekly_projection_index": 50,
        "ab": 20,
        "h": 6,
        "hr": 2,
        "r": 4,
        "rbi": 5,
        "sb": 1,
        "bb": 2,
        "hbp": 0,
        "sf": 0,
        "total_bases": 14,
    }

    optimized = optimize_lineup(active_hitters + [bench_springer])
    optimized_by_name = {player["name"]: player for player in optimized["lineup"]}
    springer_change = next(
        change for change in optimized["changes"] if change["start"] == "George Springer"
    )

    assert optimized_by_name["Aaron Judge"]["optimized_slot"] == "OF"
    assert optimized_by_name["Aaron Judge"]["optimized_slot"] != "C"
    assert optimized_by_name["Jazz Chisholm"]["optimized_slot"] == "2B"
    assert springer_change["slot"] == "OF"
    assert springer_change["sit"] != "Jazz Chisholm"


def test_optimizer_excludes_unavailable_players() -> None:
    active_unavailable = {
        "name": "Injured Starter",
        "team": "TST",
        "player_type": "batter",
        "player_key": "1",
        "name_key": "injured starter",
        "slot_id": "3B",
        "slot_label": "3B",
        "lineup_status": "active",
        "eligible_positions": ["3B"],
        "expected_fantasy_value": 4.0,
        "weekly_projection_index": 100,
        "is_available": False,
        "availability_status": "10-day IL",
        "availability_note": "test status",
    }
    bench_replacement = {
        "name": "Healthy Bench",
        "team": "TST",
        "player_type": "batter",
        "player_key": "2",
        "name_key": "healthy bench",
        "slot_id": "BN1",
        "slot_label": "BN",
        "lineup_status": "bench",
        "eligible_positions": ["3B"],
        "expected_fantasy_value": 1.0,
        "weekly_projection_index": 60,
        "is_available": True,
    }

    optimized = optimize_lineup([active_unavailable, bench_replacement])

    optimized_names = {player["name"] for player in optimized["lineup"]}
    assert "Healthy Bench" in optimized_names
    assert "Injured Starter" not in optimized_names
    assert optimized["unavailable"][0]["name"] == "Injured Starter"
    assert optimized["changes"][0]["start"] == "Healthy Bench"
    assert optimized["changes"][0]["sit"] == "Injured Starter"
    assert optimized["changes"][0]["reason"] == "sit_player_unavailable"
    assert optimized["changes"][0]["sit_availability_status"] == "10-day IL"
    assert optimized["changes"][0]["value_gain"] == 1.0


def test_optimizer_does_not_pair_incompatible_unavailable_hitter_sit() -> None:
    active_unavailable_of = {
        "name": "Aaron Judge",
        "team": "NYY",
        "player_type": "batter",
        "player_key": "judge",
        "name_key": "aaron judge",
        "slot_id": "OF1",
        "slot_label": "OF",
        "lineup_status": "active",
        "eligible_positions": ["OF"],
        "expected_fantasy_value": 4.0,
        "weekly_projection_index": 100,
        "is_available": False,
        "availability_status": "Injured 10-Day",
    }
    bench_third_baseman = {
        "name": "Rafael Devers",
        "team": "SFG",
        "player_type": "batter",
        "player_key": "devers",
        "name_key": "rafael devers",
        "slot_id": "BN1",
        "slot_label": "BN",
        "lineup_status": "bench",
        "eligible_positions": ["3B"],
        "expected_fantasy_value": -0.1,
        "weekly_projection_index": 45,
        "is_available": True,
    }

    optimized = optimize_lineup([active_unavailable_of, bench_third_baseman])

    assert optimized["unavailable"][0]["name"] == "Aaron Judge"
    assert optimized["changes"][0]["start"] == "Rafael Devers"
    assert optimized["changes"][0]["sit"] is None


def test_optimizer_does_not_report_active_outfielder_slot_shuffle() -> None:
    active_hitters = [
        {
            "name": "Juan Soto",
            "team": "NYM",
            "player_type": "batter",
            "player_key": "soto",
            "name_key": "juan soto",
            "lineup_status": "active",
            "slot_id": "OF1",
            "slot_label": "OF",
            "eligible_positions": "OF",
            "expected_fantasy_value": 2.5,
            "weekly_projection_index": 90,
            "ab": 20,
            "h": 6,
            "hr": 2,
            "r": 5,
            "rbi": 5,
            "sb": 0,
            "bb": 4,
            "hbp": 0,
            "sf": 0,
            "total_bases": 14,
        },
        {
            "name": "Aaron Judge",
            "team": "NYY",
            "player_type": "batter",
            "player_key": "judge",
            "name_key": "aaron judge",
            "lineup_status": "active",
            "slot_id": "OF2",
            "slot_label": "OF",
            "eligible_positions": "OF",
            "expected_fantasy_value": 2.58,
            "weekly_projection_index": 91,
            "ab": 20,
            "h": 6,
            "hr": 2,
            "r": 5,
            "rbi": 5,
            "sb": 0,
            "bb": 4,
            "hbp": 0,
            "sf": 0,
            "total_bases": 14,
        },
    ]

    optimized = optimize_lineup(active_hitters)

    assert optimized["changes"] == []
    assert {player["name"] for player in optimized["lineup"]} == {
        "Juan Soto",
        "Aaron Judge",
    }
