# pip install pandas numpy scikit-learn boto3 spacy botocore
# python -m spacy download en_core_web_md
# Make sure your AWS credentials are configured for Amazon Bedrock access.

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from botocore.config import Config
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spacy
except ImportError:
    spacy = None


# -----------------------------
# 1) Input data
# -----------------------------
articles = [
    {
        "source": "article_1",
        "text": """The upcoming week in fantasy baseball sees most teams’ number-two and number-three starters making multiple appearances. The matchups aren’t as optimal as the current two-start pitchers. My “No-Brainer” selections don’t have the best matchups, but so far they’ve proven to be reliable fantasy assets regardless of the matchup. More great Fantasy Baseball Analysis: Fantasy Baseball Waiver Wire | Weekly SP Rankings | Pitching Streamers | Two-Start Pitchers | Lineup Analysis | MLB Injury Report | Rookie Report | Advanced Stats | Under the Radar | Overvalued & Undervalued | Bullpen Report

In my breakdown, the first tier is the “No-Brainers,” with ideal pitchers and matchups that require no thought to start. Then the “Next In Line” tier, which will include other optimal matchups but carries a sense of risk.

The third tier is the “Streamer” tier, which is what the name suggests: pitchers who can produce decent stats, but the matchups aren’t ideal, so start at your own risk. The last tier is the “Avoid” tier, which includes pitchers with terrible matchups or who have been pitching badly and can be left on the bench or on the waiver wire. I also share my insights into my favorite two-start pitchers for the upcoming week.

April 20-26 Two-Start Pitchers
My Two Favorite Two-Start Pitchers
Dylan Cease, Toronto Blue Jays (@ Angels, vs Guardians)
Cease has been lights out this season, posting a 1.74 ERA over 20.2 innings in his first four starts on the season. He’s second in MLB in strikeouts with 32, thanks to a 41% whiff rate, which ranks in the 98th percentile. A major flaw throughout his eight-year career has been walking batters, averaging 3.84 walks per nine innings, and this season is no exception.

So far, Cease has his second-worst career walk rate (5.23 per nine innings), but he’s proven he can leave these runners on base with a 77.8% left-on-base rate. With the high strikeout and walk rates, Cease has only thrown six or more innings once.

The righty’s pitch count will be tested against an Angels team that has the second-most strikeouts and most walks in MLB, and a Guardians team that ranks 21st in strikeouts, but 12th in walks. However, Cease’s last two outings were against better lineups in the Dodgers and Brewers, where he allowed only six hits and one earned run over 11 combined innings. Look for two more solid outings from Cease if he can keep the pitch count down.

Nolan McLean, New York Mets (vs Twins, vs Rockies)
The Mets’ rookie has looked elite over his young career. This season, McLean has a 2.28 ERA with a 28:8 strikeout-to-walk ratio over 23.2 innings. Through his four starts, he has struck out eight batters in three of them and has walked two batters in each start.

Although it’s only a small sample size, McLean has been better on the road than at home, allowing four earned runs in two games, but hitters have only had a .175 batting average off the Mets’ righty. Even though the games are at home, McLean gets two optimal teams next week, with the Twins ranking 10th in strikeouts and the Rockies ranking first.

No-Brainers
Shota Imanaga, Chicago Cubs (vs Phillies, @ Dodgers)

Aaron Nola, Philadelphia Phillies (@ Cubs, @ Braves)

Reynaldo Lopez, Atlanta Braves (@ Nationals, vs Phillies)

Bryce Elder, Atlanta Braves (@ Nationals, vs Phillies)

Emerson Hancock, Seattle Mariners (vs Athletics, @ Cardinals)

Next In Line
Jesus Luzardo, Philadelphia Phillies (@ Cubs, @ Braves)

Kyle Bradish, Baltimore Orioles (@ Royals, vs Red Sox)

Seth Lugo, Kansas City Royals (vs Orioles, vs Angels)

Sonny Gray, Boston Red Sox (vs Tigers, @ Orioles)

Chad Patrick, Milwaukee Brewers (@ Tigers, vs Pirates)

Carmen Mlodzinski, Pittsburgh Pirates (@ Rangers, @ Brewers)

Jack Flaherty, Detroit Tigers (@ Red Sox, @ Reds)

Kumar Rocker, Texas Rangers (@ Pirates, @ Athletics)

Max Meyer, Miami Marlins (vs Cardinals, @ Giants)

Connelly Early, Boston Red Sox (vs Yankees, @ Orioles)

Rhett Lowder, Cincinnati Reds (@ Rays, vs Tigers)

Streamers
Tyler Mahle, San Francisco Giants (@ Dodgers, vs Marlins)

Sean Burke, Chicago White Sox (@ Diamondbacks, vs Nationals)

Foster Griffin, Washington Nationals (vs Braves, @ White Sox)

Michael McGreevy, St. Louis Cardinals (@ Marlins, vs Mariners)

Slade Cecconi, Cleveland Guardians (vs Astros, @ Blue Jays)

J.T. Ginn, Athletics (@ Mariners, @ Rangers)

Mick Abel, Minnesota Twins (@ Mets, @ Rays)

Avoid
Colin Rea, Chicago Cubs (vs Phillies, @ Dodgers)

Luis Gil, New York Yankees (@ Red Sox, @ Astros)

Jake Irvin, Washington Nationals (vs Braves, @ White Sox)

Jack Kochanowicz, Los Angeles Angels (vs Blue Jays, @ Royals)

Tomoyuki Sugano, Colorado Rockies (@ Padres, @ Mets)

Jose Quintana, Colorado Rockies (vs Dodgers, @ Mets)
"""
    },
    {
        "source": "article_2",
        "text": """Every weekend, Scott White ranks the two-start pitchers for the upcoming scoring period and then categorizes them by how usable they are. The names depicted here require some forecasting and are, therefore, subject to change.







The video player is currently playing an ad. You can skip the ad in 5 sec with a mouse or keyboard
Below are the two-start pitchers for Fantasy Week 5 (April 20-26). All information is up to date as of Monday evening.

Must-start, all formats
1	
Nolan McLean SP  NYM
vs
team logo
Minnesota
vs
team logo
Colorado
2	
Dylan Cease SP  TOR
@
team logo
L.A. Angels
vs
team logo
Cleveland
3	
Shota Imanaga RP  CHC
vs
team logo
Philadelphia
@
team logo
L.A. Dodgers
4	
Kyle Bradish SP  BAL
@
team logo
Kansas City
vs
team logo
Boston
Advisable in most cases
5	
Emerson Hancock SP  SEA
vs
team logo
Athletics
@
team logo
St. Louis
6	
Connelly Early SP  BOS
vs
team logo
N.Y. Yankees
@
team logo
Baltimore
7	
Seth Lugo SP  KC
vs
team logo
Baltimore
vs
team logo
L.A. Angels
8	
Jesus Luzardo SP  PHI
@
team logo
Chi. Cubs
@
team logo
Atlanta
9	
Sonny Gray SP  BOS
vs
team logo
Detroit
@
team logo
Baltimore
10	
Reid Detmers SP  LAA
vs
team logo
Toronto
@
team logo
Kansas City
11	
Spencer Arrighetti P  HOU
@
team logo
Cleveland
vs
team logo
N.Y. Yankees
12	
Bryce Elder SP  ATL
@
team logo
Washington
vs
team logo
Philadelphia
Better left for points leagues
13	
Aaron Nola SP  PHI
@
team logo
Chi. Cubs
@
team logo
Atlanta
14	
Reynaldo Lopez RP  ATL
@
team logo
Washington
vs
team logo
Philadelphia
15	
Rhett Lowder SP  CIN
@
team logo
Tampa Bay
vs
team logo
Detroit
16	
Max Meyer SP  MIA
vs
team logo
St. Louis
@
team logo
San Francisco
17	
Chad Patrick P  MIL
@
team logo
Detroit
vs
team logo
Pittsburgh
No thanks
18	
Jack Flaherty SP  DET
@
team logo
Boston
@
team logo
Cincinnati
19	
Carmen Mlodzinski RP  PIT
@
team logo
Texas
@
team logo
Milwaukee
20	
Michael McGreevy P  STL
@
team logo
Miami
vs
team logo
Seattle
21	
Justin Wrobleski P  LAD
@
team logo
Colorado
vs
team logo
Chi. Cubs
22	
Slade Cecconi SP  CLE
vs
team logo
Houston
@
team logo
Toronto
23	
Sean Burke P  CHW
@
team logo
Arizona
vs
team logo
Washington
24	
Tyler Mahle SP  SF
vs
team logo
L.A. Dodgers
vs
team logo
Miami
25	
Foster Griffin SP  WAS
vs
team logo
Atlanta
@
team logo
Chi. White Sox
26	
Keider Montero SP  DET
vs
team logo
Milwaukee
@
team logo
Cincinnati
27	
Kumar Rocker P  TEX
vs
team logo
Pittsburgh
vs
team logo
Athletics
28	
Jesse Scholtens RP  TB
vs
team logo
Cincinnati
vs
team logo
Minnesota
29	
Jake Irvin SP  WAS
vs
team logo
Atlanta
@
team logo
Chi. White Sox
30	
J.T. Ginn P  ATH
@
team logo
Seattle
@
team logo
Texas
31	
Tomoyuki Sugano SP  COL
vs
team logo
San Diego
@
team logo
N.Y. Mets
32	
Luis Gil SP  NYY
@
team logo
Boston
@
team logo
Houston
33	
Jose Quintana SP  COL
vs
team logo
L.A. Dodgers
@
team logo
N.Y. Mets

Join the Conversation
"""
    },
    {
        "source": "article_3",
        "text": """
Plan ahead in fantasy baseball with help from ESPN forecaster projections. This outlook provides a preview of the next 10 days for every team, projecting the starting pitcher for each game and their corresponding projected fantasy points using ESPN standard scoring: 2 points per win, -2 per loss, 3 per inning pitched, 1 per strikeout, -1 per hit or walk allowed, and -2 per earned run allowed.

This page is updated daily throughout the season and is designed to give fantasy managers a current 10-day pitching outlook. It also has a companion page for 10-day hitting matchup projections.

Pitching Matchups: April 22 to May 1

Arizona Diamondbacks:
April 23 vs White Sox - Eduardo Rodriguez (L), 7.2 FPTS
April 24 vs White Sox - Michael Soroka (R), 9.9 FPTS
April 26 vs Padres - Zac Gallen (R), 0.9 FPTS
April 27 vs Padres - Ryne Nelson (R), 0.4 FPTS
April 29 at Brewers - Merrill Kelly (R), 8.1 FPTS
April 30 at Brewers - Eduardo Rodriguez (L), 6.9 FPTS
May 1 at Cubs - Michael Soroka (R), 8.9 FPTS

Atlanta Braves:
April 22 at Nationals - Didier Fuentes (R), 6.1 FPTS
April 23 at Nationals - Martin Perez (L), 7.5 FPTS
April 24 vs Phillies - Grant Holmes (R), 7.9 FPTS
April 25 vs Phillies - Bryce Elder (R), 6.2 FPTS
April 26 vs Phillies - Chris Sale (L), 13.0 FPTS
April 28 vs Tigers - Reynaldo Lopez (R), 8.1 FPTS
April 29 vs Tigers - Martin Perez (L), 5.6 FPTS
April 30 vs Tigers - Grant Holmes (R), 8.0 FPTS
May 1 at Rockies - Bryce Elder (R), 0.7 FPTS

Boston Red Sox:
April 22 vs Yankees - Ranger Suarez (L), 7.5 FPTS
April 23 vs Yankees - Brayan Bello (R), 5.3 FPTS
April 24 at Orioles - Garrett Crochet (L), 11.3 FPTS
April 25 at Orioles - Payton Tolle (L), 4.2 FPTS
April 26 at Orioles - Connelly Early (L), 7.7 FPTS
April 27 at Blue Jays - Ranger Suarez (L), 10.6 FPTS
April 28 at Blue Jays - Brayan Bello (R), 9.0 FPTS
April 29 at Blue Jays - Garrett Crochet (L), 12.8 FPTS
May 1 vs Astros - Payton Tolle (L), 2.9 FPTS

Chicago Cubs:
April 22 vs Phillies - Matthew Boyd (L), 11.3 FPTS
April 23 vs Phillies - Edward Cabrera (R), 9.0 FPTS
April 24 at Dodgers - Jameson Taillon (R), 9.3 FPTS
April 25 at Dodgers - Colin Rea (R), 8.2 FPTS
April 26 at Dodgers - Shota Imanaga (L), 12.3 FPTS
April 27 at Padres - Matthew Boyd (L), 11.3 FPTS
April 28 at Padres - Edward Cabrera (R), 10.7 FPTS
April 29 at Padres - Jameson Taillon (R), 10.4 FPTS
May 1 vs Diamondbacks - Colin Rea (R), 8.4 FPTS

Detroit Tigers:
April 22 vs Brewers - Casey Mize (R), 6.7 FPTS
April 23 vs Brewers - Tarik Skubal (L), 14.0 FPTS
April 24 at Reds - Framber Valdez (L), 10.9 FPTS
April 25 at Reds - Jack Flaherty (R), 10.5 FPTS
April 26 at Reds - Keider Montero (R), 7.8 FPTS
April 27 vs Brewers - Casey Mize (R), 6.0 FPTS
April 28 vs Brewers - Tarik Skubal (L), 13.0 FPTS
April 29 vs Brewers - Framber Valdez (L), 9.6 FPTS
May 1 vs Rangers - Jack Flaherty (R), 9.5 FPTS

Los Angeles Dodgers:
April 22 at Giants - Shohei Ohtani (R), 10.5 FPTS
April 23 at Giants - Tyler Glasnow (R), 12.7 FPTS
April 24 vs Cubs - Emmet Sheehan (R), 9.8 FPTS
April 25 vs Cubs - Roki Sasaki (R), 8.5 FPTS
April 26 vs Cubs - Justin Wrobleski (L), 6.8 FPTS
April 27 vs Marlins - Yoshinobu Yamamoto (R), 13.3 FPTS
April 28 vs Marlins - Shohei Ohtani (R), 11.3 FPTS
April 29 vs Marlins - Tyler Glasnow (R), 13.5 FPTS
May 1 at Cardinals - Emmet Sheehan (R), 9.4 FPTS

Minnesota Twins:
April 22 at Mets - Kendry Rojas (L), 9.0 FPTS
April 23 at Mets - Joe Ryan (R), 14.6 FPTS
April 24 at Rays - Taj Bradley (R), 9.4 FPTS
April 25 at Rays - Bailey Ober (R), 9.5 FPTS
April 26 at Rays - Simeon Woods Richardson (R), 7.4 FPTS
April 27 vs Mariners - Kendry Rojas (L), 6.9 FPTS
April 28 vs Mariners - Joe Ryan (R), 11.2 FPTS
April 29 vs Mariners - Taj Bradley (R), 9.1 FPTS
April 30 vs Blue Jays - Bailey Ober (R), 9.1 FPTS
May 1 vs Blue Jays - Simeon Woods Richardson (R), 7.0 FPTS

New York Mets:
April 22 vs Twins - Clay Holmes (R), 10.7 FPTS
April 23 vs Twins - Christian Scott (R), 10.4 FPTS
April 24 vs Rockies - Freddy Peralta (R), 13.9 FPTS
April 25 vs Rockies - Kodai Senga (R), 10.9 FPTS
April 26 vs Rockies - Nolan McLean (R), 16.0 FPTS
April 27 vs Nationals - Clay Holmes (R), 10.7 FPTS
April 28 vs Nationals - Christian Scott (R), 10.8 FPTS
April 29 vs Nationals - Freddy Peralta (R), 13.4 FPTS
May 1 at Angels - Kodai Senga (R), 7.0 FPTS

Philadelphia Phillies:
April 22 at Cubs - Taijuan Walker (R), 4.3 FPTS
April 23 at Cubs - Cristopher Sanchez (L), 11.9 FPTS
April 24 at Braves - Andrew Painter (R), 3.0 FPTS
April 25 at Braves - Zack Wheeler (R), 13.6 FPTS
April 26 at Braves - Aaron Nola (R), 7.6 FPTS
April 27 vs Giants - Jesus Luzardo (L), 13.7 FPTS
April 28 vs Giants - Cristopher Sanchez (L), 13.4 FPTS
April 29 vs Giants - Andrew Painter (R), 4.6 FPTS
May 1 at Marlins - Zack Wheeler (R), 15.6 FPTS

Seattle Mariners:
April 22 vs Athletics - Logan Gilbert (R), 13.4 FPTS
April 24 at Cardinals - George Kirby (R), 12.0 FPTS
April 25 at Cardinals - Bryan Woo (R), 13.6 FPTS
April 26 at Cardinals - Emerson Hancock (R), 8.2 FPTS
April 27 at Twins - Luis Castillo (R), 9.6 FPTS
April 28 at Twins - Logan Gilbert (R), 12.0 FPTS
April 29 at Twins - George Kirby (R), 11.0 FPTS
May 1 vs Royals - Bryan Woo (R), 13.6 FPTS

Texas Rangers:
April 22 vs Pirates - Jack Leiter (R), 6.7 FPTS
April 23 vs Pirates - Jacob deGrom (R), 9.8 FPTS
April 24 vs Athletics - Nathan Eovaldi (R), 10.4 FPTS
April 25 vs Athletics - MacKenzie Gore (L), 7.8 FPTS
April 26 vs Athletics - Kumar Rocker (R), 7.1 FPTS
April 27 vs Yankees - Jack Leiter (R), 6.4 FPTS
April 28 vs Yankees - Jacob deGrom (R), 9.8 FPTS
April 29 vs Yankees - Nathan Eovaldi (R), 9.6 FPTS
May 1 at Tigers - MacKenzie Gore (L), 8.1 FPTS
"""
    },
]


OUTPUT_DIR = Path("embedding_assignment_outputs")
TEXT_SAMPLES_CSV = OUTPUT_DIR / "fantasy_baseball_text_samples.csv"
EMBEDDINGS_PICKLE = OUTPUT_DIR / "fantasy_baseball_embeddings.pkl"
SIMILARITY_RESULTS_CSV = OUTPUT_DIR / "fantasy_baseball_similarity_results.csv"

EMBEDDING_MODELS = [
    "spacy_en_core_web_md",
    "amazon_titan_v2_256d",
    "amazon_titan_v2_512d",
    "amazon_titan_v2_1024d",
]

AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
SPACY_MODEL_NAME = "en_core_web_md"

def build_sample_entries(article_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Manually create short, meaningful sample units for embeddings.

    These snippets are intentionally smaller than full articles so the embeddings
    capture one fantasy concept at a time.
    """
    sources_available = {row["source"] for row in article_data}
    required_sources = {"article_1", "article_2", "article_3"}
    missing_sources = required_sources - sources_available
    if missing_sources:
        raise ValueError(f"Missing expected article sources: {sorted(missing_sources)}")

    samples = [
        {
            "sample_id": "cease_matchup",
            "text": "Dylan Cease, Toronto Blue Jays (@ Angels, vs Guardians)",
            "category": "pitcher_matchup_line",
            "source": "article_1",
        },
        {
            "sample_id": "cease_outlook",
            "text": "Cease has been lights out this season, posting a 1.74 ERA over 20.2 innings in his first four starts.",
            "category": "two_start_pitcher_note",
            "source": "article_1",
        },
        {
            "sample_id": "mclean_matchup",
            "text": "Nolan McLean, New York Mets (vs Twins, vs Rockies)",
            "category": "pitcher_matchup_line",
            "source": "article_1",
        },
        {
            "sample_id": "mclean_outlook",
            "text": "McLean gets two optimal teams next week, with the Twins ranking 10th in strikeouts and the Rockies ranking first.",
            "category": "two_start_pitcher_note",
            "source": "article_1",
        },
        {
            "sample_id": "must_start_rank",
            "text": "Must-start, all formats: Nolan McLean SP NYM vs Minnesota and vs Colorado.",
            "category": "two_start_ranking",
            "source": "article_2",
        },
        {
            "sample_id": "points_league_rank",
            "text": "Better left for points leagues: Aaron Nola SP PHI at Cubs and at Braves.",
            "category": "two_start_ranking",
            "source": "article_2",
        },
        {
            "sample_id": "fpts_projection_high",
            "text": "April 26 vs Rockies - Nolan McLean (R), 16.0 FPTS",
            "category": "projected_fantasy_points",
            "source": "article_3",
        },
        {
            "sample_id": "fpts_projection_mid",
            "text": "April 28 vs Tigers - Reynaldo Lopez (R), 8.1 FPTS",
            "category": "projected_fantasy_points",
            "source": "article_3",
        },
        {
            "sample_id": "fpts_projection_low",
            "text": "May 1 at Rockies - Bryce Elder (R), 0.7 FPTS",
            "category": "projected_fantasy_points",
            "source": "article_3",
        },
        {
            "sample_id": "streamer_definition",
            "text": "The Streamer tier includes pitchers who can produce decent stats, but the matchups are not ideal, so start at your own risk.",
            "category": "strategy_note",
            "source": "article_1",
        },
        {
            "sample_id": "avoid_definition",
            "text": "The Avoid tier includes pitchers with terrible matchups or who have been pitching badly and can be left on the bench.",
            "category": "strategy_note",
            "source": "article_1",
        },
        {
            "sample_id": "scoring_rules",
            "text": "ESPN standard scoring: 2 points per win, -2 per loss, 3 per inning pitched, 1 per strikeout, -1 per hit or walk allowed, and -2 per earned run allowed.",
            "category": "scoring_rules",
            "source": "article_3",
        },
        {
            "sample_id": "ad_text",
            "text": "The video player is currently playing an ad. You can skip the ad in 5 sec with a mouse or keyboard.",
            "category": "out_of_scope_ui_text",
            "source": "article_2",
        },
        {
            "sample_id": "site_nav",
            "text": "More great Fantasy Baseball Analysis: Fantasy Baseball Waiver Wire | Weekly SP Rankings | Pitching Streamers | Two-Start Pitchers | Lineup Analysis | MLB Injury Report.",
            "category": "out_of_scope_navigation_text",
            "source": "article_1",
        },
        {
            "sample_id": "conversation_prompt",
            "text": "Join the Conversation",
            "category": "out_of_scope_ui_text",
            "source": "article_2",
        },
    ]

    return samples


def make_embedding_column_name(model_name: str) -> str:
    """Create a clean DataFrame column name from a model name."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return f"embedding_{cleaned}"


def load_spacy_model():
    """Load spaCy once so we can reuse it for all text samples."""
    if spacy is None:
        raise ImportError(
            "spaCy is not installed. Run: pip install spacy && python -m spacy download en_core_web_md"
        )

    try:
        return spacy.load(SPACY_MODEL_NAME)
    except OSError as exc:
        raise OSError(
            "The spaCy model 'en_core_web_md' is not installed. "
            "Run: python -m spacy download en_core_web_md"
        ) from exc


def generate_spacy_vector_embedding(text: str, nlp_model) -> np.ndarray:
    """Generate a 300-dimensional embedding with spaCy."""
    doc = nlp_model(text)
    return np.array(doc.vector)


def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = AWS_REGION,
    runtime: Optional[bool] = True,
    external_id: Optional[str] = None,
):
    """Create a Bedrock client using the local AWS credentials configuration."""
    session_kwargs = {"region_name": region}
    client_kwargs = {"region_name": region}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=region,
        retries={"max_attempts": 10, "mode": "standard"},
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        sts = session.client("sts")
        assume_role_kwargs = {
            "RoleArn": str(assumed_role),
            "RoleSessionName": "fantasy-baseball-embeddings",
        }
        if external_id:
            assume_role_kwargs["ExternalId"] = external_id
        response = sts.assume_role(**assume_role_kwargs)
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    service_name = "bedrock-runtime" if runtime else "bedrock"
    return session.client(service_name=service_name, config=retry_config, **client_kwargs)


def generate_titan_vector_embedding(
    text: str,
    embedding_size: int,
    aws_client,
) -> np.ndarray:
    """Generate a normalized Amazon Titan embedding."""
    model_id = "amazon.titan-embed-text-v2:0"
    request_body = json.dumps(
        {
            "inputText": text,
            "dimensions": embedding_size,
            "normalize": True,
        }
    )
    response = aws_client.invoke_model(
        body=request_body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response.get("body").read())
    return np.array(response_body["embedding"])


def generate_embeddings_for_models(
    df: pd.DataFrame, model_names: List[str]
) -> Dict[str, str]:
    """
    Generate embeddings for every text sample and store them in the DataFrame.

    Returns a mapping from model name to embedding column name.
    """
    model_to_column = {}
    texts = df["text"].tolist()
    nlp_model = None
    aws_client = None

    for model_name in model_names:
        print(f"\nLoading embedding model: {model_name}")
        column_name = make_embedding_column_name(model_name)

        if model_name == "spacy_en_core_web_md":
            if nlp_model is None:
                nlp_model = load_spacy_model()
            embeddings = [generate_spacy_vector_embedding(text, nlp_model) for text in texts]
        elif model_name == "amazon_titan_v2_256d":
            if aws_client is None:
                aws_client = get_bedrock_client()
            embeddings = [
                generate_titan_vector_embedding(text, embedding_size=256, aws_client=aws_client)
                for text in texts
            ]
        elif model_name == "amazon_titan_v2_512d":
            if aws_client is None:
                aws_client = get_bedrock_client()
            embeddings = [
                generate_titan_vector_embedding(text, embedding_size=512, aws_client=aws_client)
                for text in texts
            ]
        elif model_name == "amazon_titan_v2_1024d":
            if aws_client is None:
                aws_client = get_bedrock_client()
            embeddings = [
                generate_titan_vector_embedding(text, embedding_size=1024, aws_client=aws_client)
                for text in texts
            ]
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")

        df[column_name] = list(embeddings)
        model_to_column[model_name] = column_name
        print(f"Stored embeddings in DataFrame column: {column_name}")

    return model_to_column


def build_comparison_pairs() -> List[Dict[str, str]]:
    """Create text pairs for high-similarity and low-similarity testing."""
    return [
        {
            "pair_name": "Cease matchup vs McLean matchup",
            "expected_relationship": "high",
            "sample_id_a": "cease_matchup",
            "sample_id_b": "mclean_matchup",
        },
        {
            "pair_name": "McLean matchup vs must-start ranking",
            "expected_relationship": "high",
            "sample_id_a": "mclean_matchup",
            "sample_id_b": "must_start_rank",
        },
        {
            "pair_name": "McLean outlook vs McLean FPTS projection",
            "expected_relationship": "high",
            "sample_id_a": "mclean_outlook",
            "sample_id_b": "fpts_projection_high",
        },
        {
            "pair_name": "Streamer note vs Avoid note",
            "expected_relationship": "high",
            "sample_id_a": "streamer_definition",
            "sample_id_b": "avoid_definition",
        },
        {
            "pair_name": "Cease matchup vs ad text",
            "expected_relationship": "low",
            "sample_id_a": "cease_matchup",
            "sample_id_b": "ad_text",
        },
        {
            "pair_name": "McLean FPTS projection vs Join the Conversation",
            "expected_relationship": "low",
            "sample_id_a": "fpts_projection_high",
            "sample_id_b": "conversation_prompt",
        },
        {
            "pair_name": "Avoid note vs ad text",
            "expected_relationship": "low",
            "sample_id_a": "avoid_definition",
            "sample_id_b": "ad_text",
        },
        {
            "pair_name": "Scoring rules vs site navigation",
            "expected_relationship": "low",
            "sample_id_a": "scoring_rules",
            "sample_id_b": "site_nav",
        },
    ]


def compute_similarity_results(
    df: pd.DataFrame, model_to_column: Dict[str, str], comparison_pairs: List[Dict[str, str]]
) -> pd.DataFrame:
    """Compute cosine similarity for every comparison pair and every model."""
    rows = []
    sample_lookup = df.set_index("sample_id").to_dict(orient="index")

    for pair in comparison_pairs:
        sample_a = sample_lookup[pair["sample_id_a"]]
        sample_b = sample_lookup[pair["sample_id_b"]]

        for model_name, column_name in model_to_column.items():
            vector_a = np.array(sample_a[column_name]).reshape(1, -1)
            vector_b = np.array(sample_b[column_name]).reshape(1, -1)
            similarity_score = cosine_similarity(vector_a, vector_b)[0][0]

            rows.append(
                {
                    "model_name": model_name,
                    "pair_name": pair["pair_name"],
                    "expected_relationship": pair["expected_relationship"],
                    "sample_id_a": pair["sample_id_a"],
                    "sample_id_b": pair["sample_id_b"],
                    "text_a": sample_a["text"],
                    "text_b": sample_b["text"],
                    "category_a": sample_a["category"],
                    "category_b": sample_b["category"],
                    "cosine_similarity": round(float(similarity_score), 4),
                }
            )

    return pd.DataFrame(rows)


def evaluate_models(similarity_df: pd.DataFrame) -> pd.DataFrame:
    """Score each model by how well it separates expected-high and expected-low pairs."""
    summary_rows = []

    for model_name, group in similarity_df.groupby("model_name"):
        high_mean = group.loc[
            group["expected_relationship"] == "high", "cosine_similarity"
        ].mean()
        low_mean = group.loc[
            group["expected_relationship"] == "low", "cosine_similarity"
        ].mean()
        separation_score = high_mean - low_mean

        summary_rows.append(
            {
                "model_name": model_name,
                "average_high_similarity": round(float(high_mean), 4),
                "average_low_similarity": round(float(low_mean), 4),
                "separation_score": round(float(separation_score), 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="separation_score", ascending=False
    )
    return summary_df.reset_index(drop=True)


def save_outputs(df: pd.DataFrame, similarity_df: pd.DataFrame) -> None:
    """Save metadata, embeddings, and similarity results to files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_columns = ["sample_id", "text", "category", "source"]
    df[metadata_columns].to_csv(TEXT_SAMPLES_CSV, index=False)
    df.to_pickle(EMBEDDINGS_PICKLE)
    similarity_df.to_csv(SIMILARITY_RESULTS_CSV, index=False)


def build_assignment_writeup(best_model_name: str) -> str:
    """Create a reusable short writeup for the assignment."""
    return (
        "I split the fantasy baseball articles into smaller sample units so each text "
        "embedding represents one clear idea, such as a pitcher matchup, projection, "
        "or strategy note, instead of mixing several concepts inside one long article. "
        "Cosine similarity is useful because it measures how close two embeddings are in "
        "vector space, which lets me test whether fantasy-relevant snippets group together "
        "while unrelated site or ad text stays farther away. In this comparison, "
        f"{best_model_name} appeared to be the best fit for the fantasy baseball retrieval "
        "use case because it created the strongest separation between similar and "
        "dissimilar text pairs."
    )


def print_results(
    df: pd.DataFrame, similarity_df: pd.DataFrame, model_summary_df: pd.DataFrame
) -> None:
    """Print an assignment-ready summary."""
    best_model_name = model_summary_df.iloc[0]["model_name"]
    assignment_writeup = build_assignment_writeup(best_model_name)

    preview_columns = ["sample_id", "category", "source", "text"]

    print("\n" + "=" * 80)
    print("EMBEDDINGS ASSIGNMENT SUMMARY")
    print("=" * 80)
    print(f"Number of text samples created: {len(df)}")
    print(f"Embedding model names tested: {', '.join(EMBEDDING_MODELS)}")

    print("\nDataFrame preview:")
    print(df[preview_columns].head(10).to_string(index=False))

    print("\nCosine similarity results for expected high-similarity pairs:")
    high_pairs = similarity_df[similarity_df["expected_relationship"] == "high"]
    print(
        high_pairs[["model_name", "pair_name", "cosine_similarity"]]
        .sort_values(["model_name", "cosine_similarity"], ascending=[True, False])
        .to_string(index=False)
    )

    print("\nCosine similarity results for expected low-similarity pairs:")
    low_pairs = similarity_df[similarity_df["expected_relationship"] == "low"]
    print(
        low_pairs[["model_name", "pair_name", "cosine_similarity"]]
        .sort_values(["model_name", "cosine_similarity"], ascending=[True, True])
        .to_string(index=False)
    )

    print("\nModel comparison summary:")
    print(model_summary_df.to_string(index=False))

    print("\nConclusion:")
    print(
        f"{best_model_name} appears best for this fantasy baseball retrieval use case "
        "because it produced the strongest gap between the high-similarity and "
        "low-similarity comparison pairs."
    )

    print("\nReusable assignment writeup:")
    print(assignment_writeup)

    print("\nSaved files:")
    print(f"- {TEXT_SAMPLES_CSV}")
    print(f"- {EMBEDDINGS_PICKLE}")
    print(f"- {SIMILARITY_RESULTS_CSV}")


def main() -> None:
    sample_entries = build_sample_entries(articles)
    samples_df = pd.DataFrame(sample_entries)

    model_to_column = generate_embeddings_for_models(samples_df, EMBEDDING_MODELS)
    comparison_pairs = build_comparison_pairs()
    similarity_df = compute_similarity_results(samples_df, model_to_column, comparison_pairs)
    model_summary_df = evaluate_models(similarity_df)

    save_outputs(samples_df, similarity_df)
    print_results(samples_df, similarity_df, model_summary_df)


if __name__ == "__main__":
    main()
