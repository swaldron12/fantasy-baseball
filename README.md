# Fantasy Baseball AI Lineup Optimizer

A local fantasy baseball decision-support app that combines weekly category
projections, lineup optimization, player availability checks, and AWS Bedrock
analysis.

The app answers a practical weekly roster question:

```text
Given my roster, projected categories, schedule, matchups, and player
availability, who should I start and why?
```

## Highlights

- Local web app for entering a fantasy roster and bench
- Player search for hitters, pitchers, and bench slots
- Weekly category projections for hitters and pitchers
- Legal lineup optimizer with hitter position constraints
- MLB availability layer for IL, paternity list, and inactive players
- AWS Bedrock lineup analysis and projection chatbot
- Local fallback analysis when Bedrock is not configured
- Historical Statcast/Savant feature and ML training utilities

## Pipeline

```text
Fangraphs / ZiPS / pybaseball / MLB schedule data
   ↓
Data cleaning with pandas
   ↓
Feature engineering
   - rolling Statcast/Savant features
   - handedness splits
   - probable pitcher schedule context
   - matchup strength
   ↓
Hybrid weekly category projections
   ↓
Availability filter
   ↓
Lineup optimizer
   ↓
AWS Bedrock analysis + projection chatbot
```

## Tech Stack

```text
Python
pandas / numpy / pyarrow
pybaseball
scikit-learn
openpyxl
boto3 / AWS Bedrock
pytest
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python -m src.web_app --host 127.0.0.1 --port 8770
```

Open:

```text
http://127.0.0.1:8770
```

Run tests:

```bash
python -m pytest
```

## Optional Bedrock Setup

The optimizer and projections run locally. AWS Bedrock is used only for the
written analysis and chatbot.

Set these environment variables before starting the app:

```bash
export ENABLE_BEDROCK_ANALYSIS=true
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
```

Optional controls:

```bash
export BEDROCK_MAX_TOKENS=760
export BEDROCK_TEMPERATURE=0.15
export BEDROCK_CHAT_MAX_TOKENS=900
export BEDROCK_CHAT_TEMPERATURE=0.2
```

Check Bedrock visibility from the running app:

```bash
curl http://127.0.0.1:8770/api/bedrock-status
```

## Security

Do not commit AWS keys, OpenAI keys, `.env`, or credential files.

This repo ignores local environment files:

```text
.env
.env.*
```

Use `.env.example` as a template only. It contains placeholders and no real
secrets.

## Using The App

The front end includes:

```text
C, 1B, 2B, 3B, SS, OF, OF, OF, UTIL
P1, P2, P3, P4, P5, P6, P7
BN1, BN2, BN3
```

Each slot has a player search box. Hitter slots search hitters, pitcher slots
search pitchers, and bench slots search the full projection pool.

Hitter and bench rows include an eligibility field. The optimizer uses that
field to prevent illegal swaps, such as starting an outfielder at catcher.

After clicking **Optimize Lineup**, the app shows:

- Current lineup value
- Optimized lineup value
- Lineup value gain
- Projected hitting categories
- Projected pitching categories
- Recommended lineup changes
- Availability warnings
- AI-generated analysis
- Projection chatbot

## Projection Outputs

Hitter projections include:

```text
AB, H, HR, R, RBI, SB, AVG, OPS
```

Pitcher projections include:

```text
IP, W, SV, HLD, K, ERA, WHIP
```

Rate stats are computed from projected components. For example:

```text
AVG = H / AB
OPS = OBP + SLG
ERA = ER * 9 / IP
WHIP = baserunners / IP
```

The app also exposes a player-type-relative value index:

```text
Player Value = value compared to players of the same type
Current Lineup Value = sum of active player values
Optimized Lineup Value = sum after legal optimization
Lineup Value Gain = optimized value - current value
```

This value is not fantasy points. It is a lineup-strength index where positive
is above average and negative is below average for that player type.

## Projection Method

The main category projections are hybrid projections. They blend:

```text
ZiPS rest-of-season projections
current-year performance
recent Statcast/Savant rolling features
handedness splits
weekly team schedule
probable pitcher matchups
opponent strength
```

ZiPS anchors the projection to a stable talent estimate. Recent Savant features
allow the system to react to current form without letting one small sample drive
the entire forecast.

The projection source label for the main rows is:

```text
hybrid_zips_ytd_recent_matchup
```

## Schedule And Probable Pitchers

Weekly projections use a team-game schedule file:

```text
data/raw/schedules/probable_pitchers_week.csv
```

The schedule contains one row per team-game. A single MLB game appears twice:
once from each team's perspective.

Key fields:

```text
game_date
team
opponent
home_away
opposing_probable_pitcher_name
opposing_probable_pitcher_throws
own_probable_pitcher_name
own_probable_pitcher_throws
```

Convert a Fangraphs Roster Resource Probables Grid Excel file:

```bash
python -m src.schedules
```

Build weekly projections:

```bash
python -m src.projections
```

Projection outputs are written to:

```text
data/projections/weekly_batter_projections.csv
data/projections/weekly_pitcher_projections.csv
data/projections/weekly_projections.csv
```

## Availability Layer

Player availability is separate from player projection.

A player may still have a valid projection, but if he is on the IL or otherwise
inactive, the optimizer should not treat him as startable.

Refresh MLB availability:

```bash
python -m src.availability --season 2026 --output data/processed/player_availability.csv
```

The cache uses MLB 40-man roster status values such as:

```text
Active
Injured 10-Day
Injured 15-Day
Injured 60-Day
Paternity List
Reassigned to Minors
```

Unavailable players remain visible for auditability, but they are excluded from
optimized starters and surfaced in the UI as availability warnings.

Manual overrides can be added here:

```text
config/player_availability.csv
```

## AI Layer

AWS Bedrock is used for two user-facing AI features:

1. **Lineup analysis**
   The app summarizes why the optimizer selected the lineup, which categories
   look strong, and where risk exists.

2. **Projection chatbot**
   The user can ask follow-up questions about players, start/sit decisions,
   categories, matchups, and availability.

The LLM does not create the projections. It receives structured context from
the app:

```text
current lineup
optimized lineup
bench players
start/sit decisions
category totals
player values
verified matchup evidence
availability notes
```

This is a lightweight RAG pattern. Instead of retrieving article chunks from a
vector database, the app retrieves relevant structured rows from its own
projection system and sends them to Bedrock.

Prompt guardrails tell the model not to invent:

```text
injuries
teams
opposing pitchers
handedness
matchups
stats
illegal lineup swaps
```

Availability is treated as a hard constraint. If a player is marked unavailable,
the LLM is instructed to explain that before discussing projections or matchups.

## ML Layer

The main projection engine is not a black-box ML model. Baseball is noisy week
to week, so the production projection is anchored by ZiPS, current stats,
Savant features, and schedule context.

The ML model is used as a supporting over/under signal:

```text
Given recent Statcast/Savant features, is this player likely to perform above
or below his normal weekly category-value baseline?
```

Train the current models:

```bash
python -m src.modeling \
  --training-data data/modeling/weekly_batter_training.parquet \
  --target category_value \
  --over-under \
  --model-output models/batter_over_under_model.pkl \
  --report-output models/batter_over_under_report.json

python -m src.modeling \
  --training-data data/modeling/weekly_pitcher_training.parquet \
  --target category_value \
  --over-under \
  --model-output models/pitcher_over_under_model.pkl \
  --report-output models/pitcher_over_under_report.json
```

The historical training tables use one row per player per week:

```text
player + week + pre-week features + actual weekly category outcome
```

Train/validation/test splitting is chronological to avoid leaking future
performance into earlier predictions.

## Data Pipeline Commands

Clean raw CSV exports:

```bash
python -m src.ingest
```

Build player-level features:

```bash
python -m src.features
```

Pull/update Statcast data and rolling Savant features:

```bash
python -m src.pybaseball_data --statcast --start-date 2025-07-01 --end-date 2025-09-28
python -m src.pybaseball_data --statcast --start-date 2026-03-26 --end-date 2026-06-02
python -m src.pybaseball_data --savant-features
```

Build Statcast-derived weekly targets and training tables:

```bash
python -m src.pybaseball_data --statcast-outcome-targets --training-tables
```

## Project Structure

```text
src/
  availability.py      MLB availability cache and status flags
  category_values.py   category-value target helpers
  clean.py             raw CSV cleaning utilities
  features.py          player feature tables
  ingest.py            raw-to-processed ingestion
  modeling.py          ML training helpers
  projections.py       weekly projection builder
  pybaseball_data.py   Statcast/Savant data utilities
  roster_value.py      roster matching, optimizer, AI context
  schedules.py         probable pitcher schedule conversion
  stat_projections.py  hybrid stat projection logic
  web_app.py           local web app and API

tests/
  pytest coverage for projections, optimizer, modeling, and AI context

data/
  raw/                 source files and schedules
  processed/           cleaned data and availability cache
  features/            engineered feature tables
  projections/         weekly projection outputs
  modeling/            training tables and targets

models/
  trained over/under model artifacts and reports
```

## Current Limitations

- The app runs locally rather than as a deployed web service.
- Bedrock analysis requires local AWS credentials and model access.
- The chatbot does not yet search live news.
- Availability comes from MLB roster status plus optional manual overrides.
- Historical weekly outcomes derived from Statcast are useful for model
  experimentation but are not a perfect replacement for official box-score
  category data.

## Future Work

- Add a web-search agent limited to news from the last seven days
- Add citations for injury/news risk notes
- Persist user rosters
- Add league-specific scoring/category settings
- Improve weekly backtesting against official fantasy outcomes
- Deploy the web app
