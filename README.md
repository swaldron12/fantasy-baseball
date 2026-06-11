# Fantasy Baseball AI Lineup Optimizer

This project is a local fantasy baseball prototype that turns baseball data into
weekly category projections, optimized lineup recommendations, and AI-generated
analysis.

The full pipeline is:

```text
Fangraphs / pybaseball / MLB schedule data
   ↓
Data cleaning with pandas
   ↓
Feature engineering: rolling Savant stats, splits, matchups, ZiPS context
   ↓
Weekly category projections
   ↓
Lineup optimizer with position and availability constraints
   ↓
AWS Bedrock analysis + projection chatbot
```

The goal is to answer a practical fantasy question: given my roster, current
weekly schedule, probable pitchers, player projections, and injury/availability
status, who should I start and why?

## Quick Start

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Run the local web app:

```bash
python -m src.web_app --host 127.0.0.1 --port 8770
```

Open:

```text
http://127.0.0.1:8770
```

The app works without AWS credentials by using a local fallback analysis. To use
AWS Bedrock for the written lineup analysis and chatbot, set environment
variables in your terminal before starting the app:

```bash
export ENABLE_BEDROCK_ANALYSIS=true
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
```

Do not commit AWS keys or `.env` files. Use AWS CLI profiles, shell exports, or
a local ignored `.env` file.

## Core Dependencies

The main packages are:

```text
pandas / numpy / pyarrow        data cleaning and local tables
pybaseball                      Statcast/Savant data pulls
scikit-learn                    ML training helpers
openpyxl                        Fangraphs Excel schedule conversion
boto3                           AWS Bedrock calls
pytest                          tests
```

## Repository Safety

This repo is configured to ignore local secret files:

```text
.env
.env.*
```

Use `.env.example` as a template only. It contains no real secrets.

## What This Ingestion Step Does

The ingestion script loads every `.csv` file from `data/raw/`, applies light cleaning, and writes a cleaned `.parquet` file to `data/processed/`.

Cleaning includes:

- Standardizing column names to lowercase `snake_case`
- Converting percentage values like `25.4%` to `0.254`
- Converting numeric-looking columns to numeric types
- Preserving identifiers like `name`, `team`, `playerid`, and `mlbamid`
- Warning about missing identifiers or very high missing-value rates

We use pandas because the data is tabular and small enough to understand locally. We save Parquet because it preserves cleaner data types than CSV and loads efficiently later.

## Raw Data Location

Put manually downloaded Fangraphs CSV files here:

```bash
data/raw/
```

Do not edit these raw files directly. They are the source of truth.

## Expected Raw Files

The current ingestion step expects Fangraphs exports like:

```text
batters_standard_05_06.csv
batters_advanced_05_06.csv
batters_statcast_05_06.csv
batters_splits_lhp_05_06.csv
batters_splits_rhp_05_06.csv
pitchers_standard_05_06.csv
pitchers_advanced_05_06.csv
pitchers_statcast_05_06.csv
pitchers_split_lhh_05_06.csv
pitchers_split_rhh_05_06.csv
team_batting_lhp_05_06.csv
team_batting_rhp_05_06.csv
zips_batters_ros.csv
zips_pitchers_ros.csv
```

The script will load every `.csv` in `data/raw/`, even if the filename is slightly different. For example, local files like `standard-batting-5-06.csv` are cleaned into a standard dataset name.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the ingestion pipeline from the project root:

```bash
python -m src.ingest
```

## Created Files

Cleaned files are written to:

```bash
data/processed/
```

Example outputs:

```text
data/processed/batters_standard_05_06.parquet
data/processed/batters_advanced_05_06.parquet
data/processed/batters_statcast_05_06.parquet
data/processed/zips_batters_ros.parquet
data/processed/zips_pitchers_ros.parquet
```

Each run prints a summary for every raw CSV:

- Rows
- Columns
- Total missing values
- Output path
- Validation warnings, if any

## Quiz Questions

1. What is the difference between raw and processed data?
2. Why do we preserve the original CSV files?
3. Why do we standardize column names before modeling?

## Player-Level Feature Engineering

A feature is a model-ready column that describes something useful about a player. For example, `woba`, `xwoba`, `k_rate`, `barrel_rate`, and `ip` are features because they summarize skill, contact quality, plate discipline, or volume in a way a model can use later.

We combine multiple Fangraphs datasets because no single table tells the full story. Standard stats give volume, advanced stats give skill rates, Statcast gives contact quality, splits show handedness performance, and ZiPS gives rest-of-season projection context.

Splits are important because baseball performance depends heavily on handedness. A batter may be strong overall but weaker against left-handed pitching, while a pitcher may dominate left-handed hitters but struggle against right-handed hitters. These are still player-level features; we are not joining batters to opposing pitchers yet.

The feature pipeline creates:

```text
data/features/batters_features.parquet
data/features/pitchers_features.parquet
```

Run it from the project root:

```bash
python -m src.features
```

These tables will be used later as inputs to the modeling step. The model will learn from the feature columns, not from the raw Fangraphs exports directly.

## Feature Quiz Questions

1. What is a feature in a machine learning model?
2. Why are splits (vs LHP/RHP) important?
3. Why do we use left joins instead of inner joins?

## Weekly Projection Inputs

Weekly projections need probable pitchers and schedules in a team-game format. Put the completed file here:

```bash
data/raw/schedules/probable_pitchers_week.csv
```

A blank template already exists here:

```bash
data/raw/schedules/probable_pitchers_template.csv
```

If you downloaded the Roster Resource Probables Grid Excel file from Fangraphs, put it in:

```bash
data/raw/schedules/
```

Then convert it:

```bash
python -m src.schedules
```

By default this reads:

```bash
data/raw/schedules/roster-resource__probables-grid.xlsx.xlsx
```

and writes:

```bash
data/raw/schedules/probable_pitchers_week.csv
```

The Roster Resource grid usually includes more than one fantasy week. The converter uses the first 7 date columns by default.

Required columns:

```text
week_start
game_date
team
opponent
home_away
opposing_probable_pitcher_name
opposing_probable_pitcher_id
opposing_probable_pitcher_throws
own_probable_pitcher_name
own_probable_pitcher_id
own_probable_pitcher_throws
```

Use Fangraphs-style team abbreviations where possible, such as `NYY`, `BOS`, `LAD`, and `ATL`. Use `L` or `R` for pitcher handedness. One MLB game should appear twice: once from each team's perspective.

Why this format works:

- Hitter projections need each team's number of games vs LHP and RHP.
- Pitcher projections need each probable starter's opponent and handedness.
- A team-game row keeps both problems simple and readable.

Run baseline weekly projections:

```bash
python -m src.projections
```

Outputs are written to:

```text
data/projections/weekly_batter_projections.csv
data/projections/weekly_pitcher_projections.csv
data/projections/weekly_projections.csv
```

These are baseline projection indexes, not final trained ML predictions. They combine current player features, splits, ZiPS signals, and the weekly schedule so we can inspect rankings before training a model.

## Roster Front End

The local front end now works like a weekly lineup tool. You enter players into
roster slots, run the optimizer, and get projected category totals plus expected
fantasy value.

Run:

```bash
python -m src.web_app
```

Then open:

```text
http://127.0.0.1:8765
```

Current roster slots:

```text
C, 1B, 2B, 3B, SS, OF, OF, OF, UTIL
P1, P2, P3, P4, P5, P6, P7
BN1, BN2, BN3
```

Each slot has a player search box. Hitter slots search hitters, pitcher slots
search pitchers, and bench slots search the full projection pool.

Hitter and bench rows also have an eligibility field. Active hitter rows default
to their slot position. For bench hitters, enter fantasy-legal positions such as:

```text
OF
2B, OF
C
```

If a bench hitter has no eligibility entered or configured, the optimizer treats
him as `UTIL` only. This conservative fallback prevents illegal moves like
starting an outfielder at catcher.

You can also store repeat player eligibility here:

```text
config/player_eligibility.csv
```

Format:

```text
player_name,eligible_positions
Aaron Judge,OF
Jazz Chisholm,"2B, OF"
```

The app reads from:

```text
data/projections/weekly_batter_projections.csv
data/projections/weekly_pitcher_projections.csv
```

Weekly category projections now use a hybrid projection layer. They are still
projections, not known future stats, but they are no longer simple one-source
estimates. The model starts with ZiPS as the talent baseline, blends in
current-year form, adds the latest weekly Savant/Statcast features when
available, then adjusts the result for the current weekly schedule and matchup
context.

The main sources are:

```text
data/processed/zips_batters_ros.parquet
data/processed/zips_pitchers_ros.parquet
data/processed/batters_fangraphs_2026_ytd.parquet, if present
data/processed/pitchers_fangraphs_2026_ytd.parquet, if present
data/features/weekly_batter_savant_features.parquet
data/features/weekly_pitcher_savant_features.parquet
data/raw/schedules/probable_pitchers_week.csv
```

The hitter totals include:

```text
AB, H, HR, R, RBI, SB, AVG, OPS
```

The pitcher totals include:

```text
IP, W, SV, HLD, K, ERA, WHIP
```

Rate stats are recomputed from totals. For example, weekly AVG is total hits
divided by total at-bats, not the average of every hitter's AVG. ERA and WHIP
use the same idea with total earned runs, baserunners, and innings.

For hitters:

```text
weekly volume = projected team games * blended AB per game
hitting components = projected 1B, 2B, 3B, HR, BB, HBP, SF from ZiPS/current form/recent quality
weekly context = opposing probable pitcher quality + team offense strength + LHP/RHP split fit
AVG = H / AB
OPS = ((H + BB + HBP) / (AB + BB + HBP + SF)) + ((1B + 2*2B + 3*3B + 4*HR) / AB)
```

For pitchers:

```text
weekly volume = projected starts * blended IP per start
skill rates = ZiPS rates + current-year rates + recent Savant rates
weekly context = opponent strength from the scheduled matchup score
```

Why this design works for this project:

- ZiPS keeps the projection anchored to a stable talent estimate.
- Current-year stats let the model respond to what has actually happened this season.
- Rolling Savant features add current contact/strikeout/walk quality without overreacting to one tiny week.
- Matchup strength changes the weekly expectation because fantasy value depends on who the player faces this week.

The current local Savant sample covers:

```text
2025-07-01 through 2025-09-28
2026-03-26 through 2026-06-02
```

The projection layer prefers prior rolling Savant features, such as previous
four-week xwOBA, rather than raw latest-week hit rate. This keeps tiny samples
from pushing AVG too far away from the ZiPS true-talent baseline.

The AI summary can also mention hitter matchup notes. These are generated from
the probable-pitcher schedule, handedness mix, and pitcher difficulty signals in
the projection table. The LLM is instructed to cite only those provided matchup
fields, so it can say something like `Aaron Judge has a favorable matchup note
against Patrick Corbin (L)` without inventing matchups.

The matchup evidence is handedness-aware. Statcast/Savant rows provide the
hitter's batting side (`stand`) and pitcher throwing hand (`p_throws`). The
projection layer turns that into fields like:

```text
batter_bats
stand_vs_lhp
stand_vs_rhp
favorable_batter_matchups
tough_batter_matchups
```

This matters because the LLM should not decide whether a pitcher split vs LHB or
RHB applies. The projection layer decides the relevant split first. For example,
Aaron Judge bats right-handed, so a matchup note against Patrick Corbin should
cite Corbin's relevant split vs RHB, not his split vs LHB.

The source label for these rows is:

```text
hybrid_zips_ytd_recent_matchup
```

Relievers or unscheduled pitchers without a listed start can still appear from
ZiPS as a fallback source:

```text
scheduled_starts_or_zips_relief
```

One important cleanup rule: batter projections require either current-year
plate appearances or ZiPS hitter projection signals. This prevents pitcher
batting artifacts with zero at-bats from appearing as eligible hitters.

`weekly_projection_index` is the raw baseline projection score.
`expected_fantasy_value` is a player-type-relative value score, so hitters are
compared to hitters and pitchers are compared to pitchers. This keeps two-start
pitchers from automatically overwhelming hitter values just because their raw
projection index is on a different scale.

In the front end:

```text
Player Value = player-type-relative value index
Current Lineup Value = sum of Player Value for active slots entered now
Optimized Lineup Value = sum after the optimizer picks the best legal lineup
Lineup Value Gain = Optimized Lineup Value - Current Lineup Value
```

This value is not fantasy points and it is not a category total. It is a lineup
strength index. Around `0` means average for that player type, positive means
above average, and negative means below average.

The optimizer is deterministic. It finds the highest-value legal hitter-slot
assignment, starts the top projected pitchers in pitcher slots, then shows the
recommended slot-level changes. Hitter swaps are constrained by known fantasy
eligibility, so an `OF` cannot replace a `C` or `2B` unless that player is also
eligible there.

## Bedrock Analysis Layer

AWS Bedrock is optional and only used for the written analysis. The projection
math and optimizer still run locally first. This is intentional: the model
calculates the numbers, while the LLM explains category strengths, risk, and the
most important lineup move.

Install dependencies:

```bash
pip install -r requirements.txt
```

Then configure AWS credentials in your normal local environment and set:

```bash
export ENABLE_BEDROCK_ANALYSIS=true
export AWS_REGION="us-east-1"
```

The default small model is now Nova Lite, which is still lightweight but gives
the analysis layer more room for baseball-specific language than Nova Micro:

```bash
amazon.nova-lite-v1:0
```

Override it only if you want a different Bedrock model:

```bash
export BEDROCK_MODEL_ID="amazon.nova-lite-v1:0"
```

Optional controls:

```bash
export BEDROCK_MAX_TOKENS=760
export BEDROCK_TEMPERATURE=0.15
```

The default prompt asks Bedrock for structured plain-text sections: lineup
verdict, start/sit decisions, strong projected starters, matchup evidence,
category read, and risk notes. It also tells Bedrock that pitcher slot order
does not matter, so it should only discuss pitcher moves when a different
pitcher enters or leaves the active lineup.

Check the running server's Bedrock configuration:

```bash
curl http://127.0.0.1:8765/api/bedrock-status
```

This does not expose AWS secrets. It shows whether the server process can see
Bedrock env vars, which model ID it will use, the region, and whether `boto3` is
installed.

Cost note: each Bedrock analysis call sends the lineup summary to the selected
model and is billed according to that model's input/output token pricing. Keep
the max tokens small while testing. If Bedrock is not enabled or fails, the app
uses a local fallback analysis, so the front end still works.

## Projection Chatbot

The UI includes a projection chatbot. This is not a general fantasy baseball
chatbot. It is a data-grounded chat layer over the current roster and optimizer
result.

When a user asks a question, the backend retrieves:

```text
entered roster slots
matched projection rows
current lineup summary
optimized lineup summary
start/sit decisions
bench players
category totals
verified matchup evidence
availability / IL notes
```

That compact context is sent to AWS Bedrock. The prompt tells the LLM to answer
only from the provided projection context and to avoid inventing injuries,
matchups, teams, or stats. If Bedrock is disabled, the app returns a local
fallback answer.

This is a lightweight RAG pattern. Instead of retrieving article chunks from a
vector database, the app retrieves the relevant structured rows from its own
projection system.

## Availability / IL Layer

The app has a separate player availability layer. This matters because a player
can have a strong projection but still be unavailable for the fantasy week.

Refresh MLB availability:

```bash
python -m src.availability --season 2026 --output data/processed/player_availability.csv
```

The availability cache uses MLB Stats API 40-man roster status codes such as:

```text
Active
Injured 10-Day
Injured 15-Day
Injured 60-Day
Paternity List
Reassigned to Minors
```

Unavailable players are not deleted from the projection tables. They remain
visible for auditing, but the optimizer excludes them from startable lineups and
the UI shows an availability warning. This separation keeps the pipeline clear:

```text
Projection model = what the player would be expected to do if available
Availability layer = whether the player can actually be started
Optimizer = best legal lineup from available players
AI layer = explanation of the decision
```

Manual overrides can be added here:

```text
config/player_availability.csv
```

## Modeling

The main weekly category projections come from the hybrid projection system, not
from a black-box ML model. That was a deliberate design decision because weekly
baseball outcomes are noisy and a small training set should not replace stable
sources like ZiPS, year-to-date stats, schedule context, and handedness splits.

The ML layer is used as a supporting Savant over/under signal. It asks:

```text
Based on recent Statcast/Savant features, is this player likely to perform above
or below his normal weekly category-value baseline?
```

We use category value as the ML target, not fantasy points. That means the model should learn whether a player helps across fantasy categories.

Default hitter categories:

```text
R, HR, RBI, SB, AVG impact
```

Default pitcher categories:

```text
W, SV, K, ERA impact, WHIP impact
```

Rate categories need volume context. For example, a .400 AVG in 5 at-bats matters less than a .350 AVG in 25 at-bats, so the target uses `AVG impact` instead of raw AVG. The same idea applies to ERA and WHIP with innings pitched.

The ML model should train on historical weekly rows, not one current snapshot. Each row should mean:

```text
player + week + features known before that week + actual weekly outcome
```

The target will be `category_value`. The split should be chronological:

```text
old weeks -> train
middle weeks -> validation
newest weeks -> test
```

This avoids testing on information from the same time period the model trained on.

The category target helper lives in:

```bash
src/category_values.py
```

The train/validation/test helper lives in:

```bash
src/modeling.py
```

Train the current over/under models:

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

The model comparison is against a player baseline. This is important because a
model only matters if it beats the simple baseline of "what does this player
usually do?"

## Pybaseball Historical Data Plan

Static links are optional. The preferred path now is pybaseball. In this local environment, the pybaseball FanGraphs leaderboard endpoint returned a 403 from Fangraphs, so the most reliable path is Savant/Statcast features plus weekly range outcomes:

```text
pybaseball Statcast/Savant pulls
   ↓
weekly rolling features and handedness splits

pybaseball weekly range stats
   ↓
weekly category outcomes
   ↓
category_value target
```

Recommended historical range:

```text
2025-07-01 through the end of the 2025 fantasy/regular season
2026 opening day through the most recent completed game date
```

That gives useful recent history plus 2026 year-to-date without wasting pulls on
the offseason.

Commands:

```bash
python -m src.pybaseball_data --statcast --start-date 2025-07-01 --end-date 2025-09-28
python -m src.pybaseball_data --statcast --start-date 2026-03-26 --end-date 2026-06-02
python -m src.pybaseball_data --savant-features
```

Those commands update the Savant feature layer used by projections. For true ML
training, also pull weekly outcomes over the same ranges before building the
training tables:

```bash
python -m src.pybaseball_data --weekly-outcomes --start-date 2025-07-01 --end-date 2025-09-28
python -m src.pybaseball_data --weekly-outcomes --start-date 2026-03-26 --end-date 2026-06-02
python -m src.pybaseball_data --training-tables
```

The internet-backed commands may take time over multiple months. For a fast
smoke test, use a three-day range first:

```bash
python -m src.pybaseball_data --weekly-outcomes --start-date 2026-05-18 --end-date 2026-05-20
python -m src.pybaseball_data --statcast --start-date 2026-05-18 --end-date 2026-05-20
python -m src.pybaseball_data --savant-features
python -m src.pybaseball_data --training-tables
```

## Projection Quiz Questions

1. Why do projections need schedule and probable pitcher context?
2. Why should train/validation/test be split by week instead of random rows?
3. Why is category value a better target than raw fantasy points for a category league?

## Submission Checklist

Before submitting the GitHub repo:

```bash
python -m pytest
git status --short
```

Confirm the repo includes:

```text
README.md
requirements.txt
src/
tests/
config/
data/projections/
data/processed/player_availability.csv
models/
```

Confirm the repo does not include:

```text
.env
AWS keys
OpenAI keys
private credentials
```

The `.gitignore` blocks `.env` and `.env.*`. The included `.env.example` is only
a template and contains no real secrets.

## Demo Video Outline

Keep the video under 10 minutes. A good structure is:

1. Problem, about 1 minute

Explain that fantasy baseball managers have to make weekly start/sit decisions
using projections, schedules, probable pitchers, injuries, and category needs.
The problem is that those pieces usually live in different places, so this tool
puts them into one workflow.

2. Start the prototype, about 1 minute

Show the terminal:

```bash
python -m src.web_app --host 127.0.0.1 --port 8770
```

Open:

```text
http://127.0.0.1:8770
```

Mention that Bedrock is optional. If enabled, the app uses AWS Bedrock for
analysis and chat. If not, it still works with a local fallback.

3. Demo the UI, about 2 minutes

Use `Load Sample` or enter a roster manually. Point out:

```text
9 hitter slots
7 pitcher slots
3 bench slots
player search
hitter eligibility
Optimize Lineup button
```

Run the optimizer. Show:

```text
Current Lineup Value
Optimized Lineup Value
Lineup Value Gain
category totals
hitter projection table
pitcher projection table
start/sit recommendations
availability warnings
```

4. Explain the AI layer, about 1.5 minutes

Show the generated lineup analysis and the projection chatbot. Ask a question
like:

```text
Why is this player starting over my bench player?
Why is Judge not startable?
Which categories are strongest this week?
```

Explain that the LLM is not making the projections. The LLM receives structured
context from the projection system and explains it in baseball language.

5. Under the hood, about 3 minutes

Use this explanation:

```text
Raw data comes from Fangraphs, pybaseball/Statcast, ZiPS projections, probable
pitcher schedules, and MLB availability status.

pandas cleans the data, standardizes columns, and saves processed Parquet/CSV
tables.

Feature engineering creates rolling Savant features, handedness splits, schedule
features, and matchup context.

The projection layer blends ZiPS, current-year performance, recent Savant
signals, and weekly opponent/schedule context.

The optimizer enforces fantasy lineup constraints: hitters must fit eligible
positions, pitchers fill pitcher slots, and unavailable players are excluded.

The AI layer uses AWS Bedrock to explain the optimizer output and answer
follow-up questions using only the retrieved projection context.
```

6. Design decisions and next steps, about 1.5 minutes

Good design decisions to mention:

```text
pandas instead of a database because the data is small and tabular
Parquet because it preserves data types better than CSV
ZiPS as the projection anchor because weekly baseball is noisy
ML as a supporting Savant over/under signal, not the main projection engine
Bedrock for explanation, not stat prediction
availability as a separate layer so injured players remain auditable but are not startable
```

Good next steps:

```text
Add recent-news web search from the last 7 days
Add citations to AI risk notes
Improve validation against actual weekly category outcomes
Add league-specific scoring/category settings
Persist user rosters
Deploy the app instead of running it locally
```
