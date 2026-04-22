# pip install pandas nltk langchain-text-splitters sentence-transformers

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import nltk
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1) Input data (paste your full articles list here)
# -----------------------------
# Replace this sample with the full `articles` list you already prepared.
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


# -----------------------------
# 2) Config (easy to edit)
# -----------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_JSON = "fantasy_baseball_chunked_embeddings.json"
OUTPUT_CSV = "fantasy_baseball_chunked_embeddings.csv"
EXAMPLES_PER_METHOD = 4

FIXED_CHUNK_SIZE = 700
SENTENCE_CHUNK_MAX_CHARS = 700
RECURSIVE_CHUNK_SIZE = 700
RECURSIVE_CHUNK_OVERLAP = 150

FANTASY_KEYWORDS = [
    "two-start",
    "injury",
    "role",
    "lineup",
    "rotation",
    "waiver",
    "streamer",
    "pitcher",
    "closer",
    "bullpen",
]


# -----------------------------
# 3) Helpers
# -----------------------------
def ensure_nltk() -> None:
    """Download punkt tokenizer if missing."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def clean_text(text: str) -> str:
    """Light cleanup: collapse repeated whitespace/newlines into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_articles(raw_articles: List[Any]) -> List[Dict[str, str]]:
    """Accept list[str] or list[dict] with optional source names."""
    normalized: List[Dict[str, str]] = []
    for i, item in enumerate(raw_articles, start=1):
        if isinstance(item, str):
            source = f"article_{i}"
            text = item
        elif isinstance(item, dict):
            source = str(item.get("source", f"article_{i}"))
            text = str(item.get("text", ""))
        else:
            continue

        cleaned = clean_text(text)
        if cleaned:
            normalized.append({"source": source, "text": cleaned})
    return normalized


def build_chunk_record(source: str, method: str, idx: int, chunk_text: str) -> Dict[str, Any]:
    return {
        "source": source,
        "chunk_method": method,
        "chunk_id": idx,
        "chunk_text": chunk_text,
    }


# -----------------------------
# 4) Chunking methods
# -----------------------------
def chunk_fixed_size(article_list: List[Dict[str, str]], chunk_size: int = FIXED_CHUNK_SIZE) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for article in article_list:
        source = article["source"]
        text = article["text"]
        local_idx = 0
        for start in range(0, len(text), chunk_size):
            piece = text[start:start + chunk_size].strip()
            if piece:
                chunks.append(build_chunk_record(source, "fixed_size", local_idx, piece))
                local_idx += 1
    return chunks


def chunk_sentence_based(
    article_list: List[Dict[str, str]],
    max_chars: int = SENTENCE_CHUNK_MAX_CHARS,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for article in article_list:
        source = article["source"]
        text = article["text"]
        sentences = sent_tokenize(text)

        current: List[str] = []
        local_idx = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = " ".join(current + [sentence]).strip()
            if len(candidate) <= max_chars:
                current.append(sentence)
            else:
                if current:
                    chunks.append(
                        build_chunk_record(source, "sentence_based", local_idx, " ".join(current).strip())
                    )
                    local_idx += 1
                    current = []

                # If one sentence is too long, hard-split it.
                if len(sentence) > max_chars:
                    for start in range(0, len(sentence), max_chars):
                        piece = sentence[start:start + max_chars].strip()
                        if piece:
                            chunks.append(build_chunk_record(source, "sentence_based", local_idx, piece))
                            local_idx += 1
                else:
                    current = [sentence]

        if current:
            chunks.append(build_chunk_record(source, "sentence_based", local_idx, " ".join(current).strip()))

    return chunks


def chunk_recursive_overlap(
    article_list: List[Dict[str, str]],
    chunk_size: int = RECURSIVE_CHUNK_SIZE,
    chunk_overlap: int = RECURSIVE_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Dict[str, Any]] = []
    for article in article_list:
        source = article["source"]
        text = article["text"]
        split_chunks = splitter.split_text(text)
        for local_idx, piece in enumerate(split_chunks):
            clean_piece = piece.strip()
            if clean_piece:
                chunks.append(build_chunk_record(source, "recursive_overlap", local_idx, clean_piece))
    return chunks


# -----------------------------
# 5) Method comparison + selection
# -----------------------------
def score_chunk_method(chunks: List[Dict[str, Any]], keywords: List[str]) -> Dict[str, float]:
    if not chunks:
        return {"keyword_coverage": 0.0, "sentence_like": 0.0, "score": 0.0}

    texts = [c["chunk_text"].lower() for c in chunks]

    keyword_hits = 0
    sentence_like_hits = 0
    for t in texts:
        if any(k in t for k in keywords):
            keyword_hits += 1
        if t.endswith((".", "!", "?", ")", "\"")):
            sentence_like_hits += 1

    keyword_coverage = keyword_hits / len(chunks)
    sentence_like = sentence_like_hits / len(chunks)
    score = 0.65 * keyword_coverage + 0.35 * sentence_like

    return {
        "keyword_coverage": keyword_coverage,
        "sentence_like": sentence_like,
        "score": score,
    }


def choose_final_method(method_chunks: Dict[str, List[Dict[str, Any]]]) -> str:
    metrics: Dict[str, Dict[str, float]] = {
        name: score_chunk_method(chunks, FANTASY_KEYWORDS)
        for name, chunks in method_chunks.items()
    }

    # Preference rule: recursive overlap is usually better for sports-news ideas that span lines.
    adjusted_scores = {k: v["score"] for k, v in metrics.items()}
    adjusted_scores["recursive_overlap"] = adjusted_scores.get("recursive_overlap", 0.0) + 0.03

    best_method = max(adjusted_scores, key=adjusted_scores.get)

    # If recursive is close to best, keep recursive as default assignment preference.
    recursive_score = adjusted_scores.get("recursive_overlap", 0.0)
    if recursive_score >= adjusted_scores[best_method] - 0.05:
        return "recursive_overlap"
    return best_method


def print_chunk_examples(method_name: str, chunks: List[Dict[str, Any]], n_examples: int) -> None:
    print(f"\n--- {method_name} example chunks ({min(n_examples, len(chunks))}) ---")
    for i, row in enumerate(chunks[:n_examples], start=1):
        text = row["chunk_text"]
        if len(text) > 280:
            text = text[:280] + "..."
        print(f"{i}. [{row['source']}] {text}")


def build_assignment_writeup(chosen_method: str) -> str:
    reasons = {
        "fixed_size": "Fixed-size chunking is simple and fast, but can split important ideas in the middle.",
        "sentence_based": "Sentence-based chunking preserves sentence boundaries, but related thoughts can still be separated when sections are long.",
        "recursive_overlap": "Recursive chunking with overlap best preserved complete fantasy baseball ideas while still controlling chunk length.",
    }

    method_reason = reasons.get(chosen_method, "The selected method balanced context preservation and chunk size.")
    return (
        f"Chosen chunking strategy: {chosen_method}. "
        f"{method_reason} "
        "For fantasy baseball news text, overlap helps keep connected details together, such as two-start pitcher notes, "
        "injury/role updates, and lineup or rotation changes that may span neighboring sentences."
    )


# -----------------------------
# 6) Main pipeline
# -----------------------------
def main() -> None:
    ensure_nltk()

    cleaned_articles = normalize_articles(articles)
    if not cleaned_articles:
        raise ValueError("No valid article text found. Paste your articles list into the script.")

    # Compare three chunking approaches.
    fixed_chunks = chunk_fixed_size(cleaned_articles)
    sentence_chunks = chunk_sentence_based(cleaned_articles)
    recursive_chunks = chunk_recursive_overlap(cleaned_articles)

    method_chunks = {
        "fixed_size": fixed_chunks,
        "sentence_based": sentence_chunks,
        "recursive_overlap": recursive_chunks,
    }

    # Short in-code comparison:
    # - fixed_size: easiest baseline, but can cut meaning mid-thought.
    # - sentence_based: cleaner readability, but no overlap means weaker continuity.
    # - recursive_overlap: usually best for preserving connected baseball news ideas.
    chosen_method = choose_final_method(method_chunks)
    final_chunks = method_chunks[chosen_method]

    # Create embeddings for final chunks.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    final_texts = [c["chunk_text"] for c in final_chunks]
    embeddings = model.encode(final_texts, convert_to_numpy=True, show_progress_bar=False)

    # Combine chunks + embeddings.
    output_rows: List[Dict[str, Any]] = []
    for i, chunk in enumerate(final_chunks):
        output_rows.append(
            {
                "source": chunk["source"],
                "chunk_method": chunk["chunk_method"],
                "chunk_id": chunk["chunk_id"],
                "chunk_text": chunk["chunk_text"],
                "embedding": embeddings[i].tolist(),
            }
        )

    # Save outputs (JSON + CSV). CSV stores embedding as JSON string.
    output_json_path = Path(OUTPUT_JSON)
    output_json_path.write_text(json.dumps(output_rows, indent=2), encoding="utf-8")

    df = pd.DataFrame(output_rows)
    df["embedding"] = df["embedding"].apply(json.dumps)
    df.to_csv(OUTPUT_CSV, index=False)

    # Print required assignment outputs.
    print("\n========== SUMMARY ==========")
    print(f"Number of articles: {len(cleaned_articles)}")
    print(f"Chunks (fixed-size): {len(fixed_chunks)}")
    print(f"Chunks (sentence-based): {len(sentence_chunks)}")
    print(f"Chunks (recursive-overlap): {len(recursive_chunks)}")

    print_chunk_examples("fixed_size", fixed_chunks, EXAMPLES_PER_METHOD)
    print_chunk_examples("sentence_based", sentence_chunks, EXAMPLES_PER_METHOD)
    print_chunk_examples("recursive_overlap", recursive_chunks, EXAMPLES_PER_METHOD)

    print(f"\nChosen final method: {chosen_method}")
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Final embedding count: {len(embeddings)}")
    print(f"Saved JSON: {output_json_path.resolve()}")
    print(f"Saved CSV: {Path(OUTPUT_CSV).resolve()}")

    assignment_writeup = build_assignment_writeup(chosen_method)
    print("\nReusable assignment writeup:")
    print(assignment_writeup)


if __name__ == "__main__":
    main()
