"""A small local multi-agent pattern for fantasy baseball analysis.

This file is intentionally lightweight for an assignment:
- No OpenAI API call
- No AWS
- No RAG database

The goal is to demonstrate the multi-agent pattern in code. Each agent has one
responsibility, and the coordinator passes structured outputs between them.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"


@dataclass
class AgentResult:
    """Standard output format for every agent."""

    agent_name: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryPlan:
    """The coordinator's plan for answering the user question."""

    question: str
    player_name: str
    player_type: str
    tasks: list[str]


class CoordinatorAgent:
    """Finds the player and decides which specialist agents should run."""

    def create_plan(
        self,
        question: str,
        batters: pd.DataFrame,
        pitchers: pd.DataFrame,
    ) -> QueryPlan:
        batter_match = _find_player(question, batters)
        pitcher_match = _find_player(question, pitchers)

        if batter_match is None and pitcher_match is None:
            raise ValueError(
                "I could not find a player from the question in the feature tables."
            )

        player_type = self._choose_player_type(
            question,
            batter_match,
            pitcher_match,
            batters,
            pitchers,
        )
        player_name = batter_match if player_type == "batter" else pitcher_match

        return QueryPlan(
            question=question,
            player_name=str(player_name),
            player_type=player_type,
            tasks=["stats_analysis", "risk_analysis", "recommendation"],
        )

    def _choose_player_type(
        self,
        question: str,
        batter_match: str | None,
        pitcher_match: str | None,
        batters: pd.DataFrame,
        pitchers: pd.DataFrame,
    ) -> str:
        question_lower = question.lower()

        pitcher_words = {"pitcher", "starter", "era", "xfip", "siera", "ip"}
        batter_words = {"batter", "hitter", "hr", "homer", "woba", "barrel"}

        if pitcher_match and any(word in question_lower for word in pitcher_words):
            return "pitcher"
        if batter_match and any(word in question_lower for word in batter_words):
            return "batter"
        if batter_match and pitcher_match:
            batter_pa = _player_metric(batters, batter_match, "pa")
            pitcher_ip = _player_metric(pitchers, pitcher_match, "ip")
            if pitcher_ip is not None and pitcher_ip >= 5:
                if batter_pa is None or batter_pa < 25:
                    return "pitcher"
        if batter_match:
            return "batter"
        return "pitcher"


class StatsAgent:
    """Summarizes player skill by comparing features to table medians."""

    def analyze(self, player_row: pd.Series, population: pd.DataFrame) -> AgentResult:
        player_type = str(player_row["player_type"])
        comparison_pool = _comparison_population(population, player_type)

        if player_type == "batter":
            metric_rules = {
                "woba": "higher",
                "xwoba": "higher",
                "barrel_rate": "higher",
                "hardhit_rate": "higher",
                "k_rate": "lower",
                "bb_rate": "higher",
                "iso": "higher",
                "zips_woba": "higher",
            }
        else:
            metric_rules = {
                "k_rate": "higher",
                "bb_rate": "lower",
                "xfip": "lower",
                "siera": "lower",
                "barrel_rate_allowed": "lower",
                "hardhit_rate_allowed": "lower",
                "zips_era": "lower",
                "zips_k_rate": "higher",
            }

        comparisons = {}
        positive_signals = []
        negative_signals = []

        for metric, direction in metric_rules.items():
            if metric not in player_row.index or metric not in population.columns:
                continue

            value = _to_float(player_row[metric])
            median = _to_float(comparison_pool[metric].median(numeric_only=True))
            if value is None or median is None:
                continue

            is_positive = value >= median if direction == "higher" else value <= median
            comparisons[metric] = {
                "player_value": round(value, 4),
                "table_median": round(median, 4),
                "signal": "positive" if is_positive else "negative",
            }

            phrase = f"{metric} {value:.3f} vs median {median:.3f}"
            if is_positive:
                positive_signals.append(phrase)
            else:
                negative_signals.append(phrase)

        summary_parts = []
        if positive_signals:
            summary_parts.append("Positive signals: " + "; ".join(positive_signals[:3]))
        if negative_signals:
            summary_parts.append("Concerns: " + "; ".join(negative_signals[:3]))
        if not summary_parts:
            summary_parts.append("Not enough feature data to compare this player.")

        return AgentResult(
            agent_name="StatsAgent",
            summary=" ".join(summary_parts),
            details={"comparisons": comparisons},
        )


class RiskAgent:
    """Flags uncertainty from playing time, missing data, and risk indicators."""

    def analyze(self, player_row: pd.Series) -> AgentResult:
        player_type = str(player_row["player_type"])
        feature_columns = [
            column
            for column in player_row.index
            if column not in {"player_key", "playerid", "mlbamid", "name", "team", "player_type"}
        ]

        risk_flags = []
        missing_count = int(player_row[feature_columns].isna().sum())
        if missing_count:
            risk_flags.append(f"{missing_count} feature values are missing")

        if player_type == "batter":
            pa = _to_float(player_row.get("pa"))
            k_rate = _to_float(player_row.get("k_rate"))
            zips_woba = _to_float(player_row.get("zips_woba"))

            if pa is not None and pa < 50:
                risk_flags.append("low plate appearances sample")
            if k_rate is not None and k_rate > 0.30:
                risk_flags.append("high strikeout rate")
            if zips_woba is not None and zips_woba < 0.310:
                risk_flags.append("modest ZiPS projected wOBA")
        else:
            ip = _to_float(player_row.get("ip"))
            bb_rate = _to_float(player_row.get("bb_rate"))
            zips_era = _to_float(player_row.get("zips_era"))

            if ip is not None and ip < 20:
                risk_flags.append("low innings sample")
            if bb_rate is not None and bb_rate > 0.10:
                risk_flags.append("high walk rate")
            if zips_era is not None and zips_era > 4.50:
                risk_flags.append("elevated ZiPS projected ERA")

        summary = (
            "Risk flags: " + "; ".join(risk_flags)
            if risk_flags
            else "No major risk flags from the available feature table."
        )

        return AgentResult(
            agent_name="RiskAgent",
            summary=summary,
            details={"risk_flags": risk_flags, "missing_feature_count": missing_count},
        )


class RecommendationAgent:
    """Combines specialist outputs into a final recommendation."""

    def recommend(
        self,
        plan: QueryPlan,
        stats_result: AgentResult,
        risk_result: AgentResult,
    ) -> AgentResult:
        comparisons = stats_result.details.get("comparisons", {})
        risk_flags = risk_result.details.get("risk_flags", [])

        positives = sum(
            1 for result in comparisons.values() if result.get("signal") == "positive"
        )
        negatives = sum(
            1 for result in comparisons.values() if result.get("signal") == "negative"
        )

        score = positives - negatives - len(risk_flags)

        if score >= 2:
            recommendation = "START / FAVOR"
            reason = "the positive feature signals outweigh the risk flags"
        elif score <= -2:
            recommendation = "BENCH / AVOID"
            reason = "the risk flags and weak feature signals outweigh the positives"
        else:
            recommendation = "HOLD / MONITOR"
            reason = "the feature profile is mixed"

        summary = (
            f"{recommendation}: {plan.player_name} is a {plan.player_type}. "
            f"The recommendation is to {recommendation.lower()} because {reason}."
        )

        return AgentResult(
            agent_name="RecommendationAgent",
            summary=summary,
            details={
                "recommendation": recommendation,
                "score": score,
                "positive_signal_count": positives,
                "negative_signal_count": negatives,
                "risk_flag_count": len(risk_flags),
            },
        )


class MultiAgentFantasyAssistant:
    """Runs the full multi-agent workflow."""

    def __init__(self, features_dir: Path = FEATURES_DIR) -> None:
        self.features_dir = features_dir
        self.coordinator = CoordinatorAgent()
        self.stats_agent = StatsAgent()
        self.risk_agent = RiskAgent()
        self.recommendation_agent = RecommendationAgent()

    def answer(self, question: str) -> str:
        batters = _load_features(self.features_dir / "batters_features.parquet")
        pitchers = _load_features(self.features_dir / "pitchers_features.parquet")

        plan = self.coordinator.create_plan(question, batters, pitchers)

        if plan.player_type == "batter":
            population = batters.assign(player_type="batter")
        else:
            population = pitchers.assign(player_type="pitcher")

        player_row = _get_player_row(population, plan.player_name)

        stats_result = self.stats_agent.analyze(player_row, population)
        risk_result = self.risk_agent.analyze(player_row)
        recommendation_result = self.recommendation_agent.recommend(
            plan,
            stats_result,
            risk_result,
        )

        return _format_report(
            plan,
            [stats_result, risk_result, recommendation_result],
        )


def _load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature table: {path}. Run `python -m src.features` first."
        )

    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "pandas needs pyarrow to read the Parquet feature tables. "
            "Run `python -m pip install -r requirements.txt` in this environment, "
            "then try the command again."
        ) from exc


def _find_player(question: str, features: pd.DataFrame) -> str | None:
    if features.empty or "name" not in features.columns:
        return None

    question_normalized = _normalize_text(question)
    names = features["name"].dropna().astype(str).drop_duplicates()

    for name in sorted(names, key=len, reverse=True):
        if _normalize_text(name) in question_normalized:
            return name

    return None


def _get_player_row(population: pd.DataFrame, player_name: str) -> pd.Series:
    matches = population[
        population["name"].astype(str).str.lower() == player_name.lower()
    ]
    if matches.empty:
        raise ValueError(f"Could not locate {player_name} in the selected feature table.")
    return matches.iloc[0]


def _player_metric(features: pd.DataFrame, player_name: str, metric: str) -> float | None:
    if metric not in features.columns or "name" not in features.columns:
        return None

    matches = features[
        features["name"].astype(str).str.lower() == player_name.lower()
    ]
    if matches.empty:
        return None

    return _to_float(matches.iloc[0][metric])


def _format_report(plan: QueryPlan, results: list[AgentResult]) -> str:
    lines = [
        "# Multi-Agent Fantasy Baseball Report",
        "",
        f"Question: {plan.question}",
        f"Coordinator plan: analyze {plan.player_name} as a {plan.player_type}",
        f"Tasks: {', '.join(plan.tasks)}",
        "",
    ]

    for result in results:
        lines.append(f"## {result.agent_name}")
        lines.append(result.summary)
        lines.append("")

    return "\n".join(lines).strip()


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _comparison_population(population: pd.DataFrame, player_type: str) -> pd.DataFrame:
    if player_type == "batter" and "pa" in population.columns:
        qualified = population[pd.to_numeric(population["pa"], errors="coerce") >= 25]
    elif player_type == "pitcher" and "ip" in population.columns:
        qualified = population[pd.to_numeric(population["ip"], errors="coerce") >= 5]
    else:
        qualified = population

    return qualified if not qualified.empty else population


def _to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local multi-agent fantasy baseball recommendation pattern."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="Should I start Aaron Judge?",
        help="Fantasy baseball question that includes a player name.",
    )
    args = parser.parse_args()

    assistant = MultiAgentFantasyAssistant()
    print(assistant.answer(args.question))


if __name__ == "__main__":
    main()
