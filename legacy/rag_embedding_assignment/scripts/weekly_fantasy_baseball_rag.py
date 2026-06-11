# pip install pandas numpy scikit-learn requests beautifulsoup4 boto3 botocore spacy sentence-transformers langchain-text-splitters
# python -m spacy download en_core_web_md
# Make sure your AWS credentials are configured for Amazon Bedrock access if you use Titan.

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from botocore.config import Config
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spacy
except ImportError:
    spacy = None


# -----------------------------------------------------------------------------
# 1) Input options
# -----------------------------------------------------------------------------
# Weekly RAG is a good fit here because fantasy baseball source material changes
# every scoring period. This script supports either:
# 1. manually pasted page text, or
# 2. URL-based fetching for a lightweight weekly refresh workflow.

USE_MANUAL_DOCUMENTS = True
USE_URL_FETCHING = False

URLS = [
    "https://www.cbssports.com/fantasy/baseball/news/fantasy-baseball-week-6-preview-two-start-pitcher-rankings/",
    "https://www.cbssports.com/fantasy/baseball/news/week-6-preview-top-10-sleeper-hitters-highlight-moises-ballesteros-miguel-vargas/",
    "https://www.cbssports.com/fantasy/baseball/news/fantasy-baseball-week-6-preview-top-10-sleeper-pitchers/",
]

# Replace the raw_text values below with the exact CBS page text you want to use.
documents = [
    {
        "source": "cbs_two_start_pitchers",
        "url": "https://www.cbssports.com/fantasy/baseball/news/fantasy-baseball-week-6-preview-two-start-pitcher-rankings/",
        "raw_text": """
        Week 6 fantasy baseball preview focuses on two-start pitcher rankings for the upcoming scoring period.
        Nolan McLean and Dylan Cease stand out as attractive options because of workload upside and favorable matchups.
        Managers in weekly leagues should pay close attention to pitchers expected to take the mound twice.
        Some arms are safer in points leagues, while others fit better as category league streamers.
        Matchup quality, strikeout upside, and expected innings remain key weekly decision variables.
        """,
    },
    {
        "source": "cbs_sleeper_hitters",
        "url": "https://www.cbssports.com/fantasy/baseball/news/week-6-preview-top-10-sleeper-hitters-highlight-moises-ballesteros-miguel-vargas/",
        "raw_text": """
        The Week 6 sleeper hitter article highlights lower-rostered bats who could help fantasy lineups in the upcoming week.
        Playing time trends, recent production, and favorable schedules all matter when identifying sleeper hitters.
        Although this page is more hitter-focused, it still contributes useful weekly fantasy context about lineup decisions.
        """,
    },
    {
        "source": "cbs_sleeper_pitchers",
        "url": "https://www.cbssports.com/fantasy/baseball/news/fantasy-baseball-week-6-preview-top-10-sleeper-pitchers/",
        "raw_text": """
        The Week 6 sleeper pitchers article discusses under-the-radar options who may be useful in fantasy leagues.
        Recent performance, matchup quality, strikeout potential, and role stability help identify viable pitcher sleepers.
        Weekly fantasy managers can use this type of article to find streamers or back-end roster options for short scoring periods.
        """,
    },
]


# -----------------------------------------------------------------------------
# 2) Settings
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path("weekly_rag_outputs")
RAW_DOCUMENTS_CSV = OUTPUT_DIR / "raw_documents.csv"
CHUNKED_DOCUMENTS_CSV = OUTPUT_DIR / "chunked_documents.csv"
RETRIEVED_RESULTS_CSV = OUTPUT_DIR / "retrieved_results.csv"

AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
SPACY_MODEL_NAME = "en_core_web_md"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

CHUNKING_METHOD = "recursive"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
FIXED_CHUNK_SIZE = 500
FIXED_CHUNK_OVERLAP = 100

TOP_K = 3
SAMPLE_QUERIES = [
    "Which pitchers appear to have two starts this week?",
    "What recent pitching information looks most useful for weekly fantasy lineup decisions?",
]


# -----------------------------------------------------------------------------
# 3) Ingestion
# -----------------------------------------------------------------------------
def fetch_page_text(url: str, timeout: int = 20) -> str:
    """Fetch page text from a URL with a browser-like header."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return normalize_text(text)


def build_documents_from_urls(urls: List[str]) -> List[Dict[str, str]]:
    """Build a document list by fetching text from URLs."""
    built_documents = []

    for index, url in enumerate(urls, start=1):
        source_name = f"cbs_page_{index}"
        try:
            raw_text = fetch_page_text(url)
        except Exception as exc:
            raw_text = f"FETCH_FAILED: {exc}"

        built_documents.append(
            {
                "source": source_name,
                "url": url,
                "raw_text": raw_text,
            }
        )

    return built_documents


def ingest_documents() -> pd.DataFrame:
    """Load documents either from manual text or optional URL fetching."""
    if USE_URL_FETCHING and not USE_MANUAL_DOCUMENTS:
        document_rows = build_documents_from_urls(URLS)
    else:
        document_rows = documents

    df = pd.DataFrame(document_rows)
    df["raw_text"] = df["raw_text"].fillna("").map(normalize_text)
    return df


# -----------------------------------------------------------------------------
# 4) Cleaning and chunking
# -----------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Remove extra whitespace while preserving baseball phrases."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text_fixed(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple fixed-size chunking for comparison."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_text_recursive(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Recursive chunking works well for article-style weekly news text."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )
    return splitter.split_text(text)


def build_chunk_dataframe(
    raw_df: pd.DataFrame,
    chunking_method: str = CHUNKING_METHOD,
) -> pd.DataFrame:
    """Convert document-level rows into chunk-level rows."""
    chunk_rows = []

    for _, row in raw_df.iterrows():
        raw_text = row["raw_text"]

        if chunking_method == "recursive":
            chunks = chunk_text_recursive(
                raw_text,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )
        elif chunking_method == "fixed":
            chunks = chunk_text_fixed(
                raw_text,
                chunk_size=FIXED_CHUNK_SIZE,
                overlap=FIXED_CHUNK_OVERLAP,
            )
        else:
            raise ValueError("chunking_method must be 'recursive' or 'fixed'")

        for chunk_index, chunk_text in enumerate(chunks, start=1):
            chunk_rows.append(
                {
                    "source": row["source"],
                    "url": row["url"],
                    "chunk_id": f"{row['source']}_chunk_{chunk_index}",
                    "chunk_text": chunk_text,
                    "chunking_method": chunking_method,
                }
            )

    return pd.DataFrame(chunk_rows)


# -----------------------------------------------------------------------------
# 5) Embeddings
# -----------------------------------------------------------------------------
def make_embedding_column_name(model_name: str) -> str:
    """Create a DataFrame-safe embedding column name."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return f"embedding_{cleaned}"


def load_spacy_model():
    """Load the spaCy model for local embeddings."""
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


def generate_spacy_embedding(text: str, nlp_model) -> np.ndarray:
    """Generate a spaCy embedding for a chunk or query."""
    return np.array(nlp_model(text).vector)


def load_sentence_transformer_model() -> SentenceTransformer:
    """Load a sentence-transformers model for local semantic embeddings."""
    return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)


def generate_sentence_transformer_embedding(
    text: str,
    model: SentenceTransformer,
) -> np.ndarray:
    """Generate a normalized sentence-transformers embedding."""
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = AWS_REGION,
    runtime: Optional[bool] = True,
    external_id: Optional[str] = None,
):
    """Create a Bedrock client using local AWS credentials."""
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
            "RoleSessionName": "weekly-fantasy-baseball-rag",
        }
        if external_id:
            assume_role_kwargs["ExternalId"] = external_id
        response = sts.assume_role(**assume_role_kwargs)
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    service_name = "bedrock-runtime" if runtime else "bedrock"
    return session.client(service_name=service_name, config=retry_config, **client_kwargs)


def generate_titan_embedding(
    text: str,
    aws_client,
    embedding_size: int = 1024,
) -> np.ndarray:
    """Generate a normalized Amazon Titan embedding."""
    request_body = json.dumps(
        {
            "inputText": text,
            "dimensions": embedding_size,
            "normalize": True,
        }
    )
    response = aws_client.invoke_model(
        body=request_body,
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response.get("body").read())
    return np.array(response_body["embedding"])


def add_embeddings_to_chunks(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """Create spaCy, Titan, and sentence-transformers embeddings for each chunk."""
    chunk_df = chunk_df.copy()

    print("\nCreating spaCy embeddings...")
    nlp_model = load_spacy_model()
    spacy_column = make_embedding_column_name("spacy_en_core_web_md")
    chunk_df[spacy_column] = chunk_df["chunk_text"].map(
        lambda text: generate_spacy_embedding(text, nlp_model)
    )

    print("Creating sentence-transformers embeddings...")
    st_model = load_sentence_transformer_model()
    st_column = make_embedding_column_name(SENTENCE_TRANSFORMER_MODEL)
    chunk_df[st_column] = chunk_df["chunk_text"].map(
        lambda text: generate_sentence_transformer_embedding(text, st_model)
    )

    print("Creating Amazon Titan embeddings...")
    aws_client = get_bedrock_client()
    titan_column = make_embedding_column_name("amazon_titan_v2_1024d")
    chunk_df[titan_column] = chunk_df["chunk_text"].map(
        lambda text: generate_titan_embedding(text, aws_client, embedding_size=1024)
    )

    return chunk_df


# -----------------------------------------------------------------------------
# 6) Retrieval
# -----------------------------------------------------------------------------
def embed_query(query: str, model_name: str, helpers: Dict[str, object]) -> np.ndarray:
    """Embed a user query with the same model used for chunk vectors."""
    if model_name == "spacy_en_core_web_md":
        return generate_spacy_embedding(query, helpers["spacy_model"])
    if model_name == SENTENCE_TRANSFORMER_MODEL:
        return generate_sentence_transformer_embedding(query, helpers["st_model"])
    if model_name == "amazon_titan_v2_1024d":
        return generate_titan_embedding(query, helpers["aws_client"], embedding_size=1024)
    raise ValueError(f"Unsupported query embedding model: {model_name}")


def retrieve_top_k_chunks(
    query: str,
    chunk_df: pd.DataFrame,
    model_name: str,
    top_k: int = TOP_K,
    helpers: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Rank chunks by cosine similarity and return only the top_k results."""
    if helpers is None:
        helpers = {}

    embedding_column = make_embedding_column_name(model_name)
    query_vector = embed_query(query, model_name, helpers).reshape(1, -1)

    scored_rows = []
    for _, row in chunk_df.iterrows():
        chunk_vector = np.array(row[embedding_column]).reshape(1, -1)
        score = cosine_similarity(query_vector, chunk_vector)[0][0]

        scored_rows.append(
            {
                "query": query,
                "model_name": model_name,
                "source": row["source"],
                "url": row["url"],
                "chunk_id": row["chunk_id"],
                "chunk_text": row["chunk_text"],
                "cosine_similarity": round(float(score), 4),
            }
        )

    results_df = pd.DataFrame(scored_rows).sort_values(
        by="cosine_similarity",
        ascending=False,
    )
    return results_df.head(top_k).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 7) Answer generation
# -----------------------------------------------------------------------------
def answer_query_with_context(
    query: str,
    retrieved_chunks: pd.DataFrame,
    use_live_llm: bool = False,
) -> str:
    """
    Return a concise grounded answer.

    A live LLM call can be added later. For this assignment-ready version,
    the fallback answer uses only the retrieved chunks.
    """
    if retrieved_chunks.empty:
        return "No relevant chunks were retrieved for this query."

    context_lines = []
    for _, row in retrieved_chunks.iterrows():
        context_lines.append(
            f"[{row['source']} | {row['chunk_id']}] {row['chunk_text']}"
        )

    if use_live_llm:
        return (
            "LLM placeholder: replace this branch with your model call later. "
            "For now, use the fallback grounded summary."
        )

    answer_parts = [
        f"Grounded answer for query: {query}",
        "Summary based only on retrieved chunks:",
    ]
    answer_parts.extend(context_lines)
    answer_parts.append(
        "These results come from the cited sources and chunk IDs listed above."
    )
    return "\n".join(answer_parts)


# -----------------------------------------------------------------------------
# 8) Saving and reporting
# -----------------------------------------------------------------------------
def save_outputs(
    raw_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    retrieved_results_df: pd.DataFrame,
) -> None:
    """Save raw documents, chunked documents, and retrieval results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df.to_csv(RAW_DOCUMENTS_CSV, index=False)

    chunk_export_df = chunk_df.copy()
    embedding_columns = [col for col in chunk_export_df.columns if col.startswith("embedding_")]
    for column in embedding_columns:
        chunk_export_df[column] = chunk_export_df[column].map(lambda x: json.dumps(np.array(x).tolist()))
    chunk_export_df.to_csv(CHUNKED_DOCUMENTS_CSV, index=False)

    retrieved_results_df.to_csv(RETRIEVED_RESULTS_CSV, index=False)


def build_assignment_writeup() -> str:
    """Create a short weekly-RAG writeup for the assignment."""
    return (
        "RAG is appropriate for this fantasy baseball use case because the source articles "
        "change every week, so the system needs to retrieve current pitcher and matchup "
        "information instead of relying on a static knowledge base. By chunking recent CBS "
        "Fantasy Baseball pages, embedding those chunks, and retrieving the most relevant "
        "ones for each query, the pipeline can generate grounded answers that reflect the "
        "latest weekly analysis while keeping the retrieval context small and focused."
    )


def print_summary(
    raw_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    retrieval_outputs: List[pd.DataFrame],
    answers: List[Dict[str, str]],
) -> None:
    """Print the assignment-ready summary."""
    print("\n" + "=" * 80)
    print("WEEKLY FANTASY BASEBALL RAG SUMMARY")
    print("=" * 80)
    print(f"Number of documents ingested: {len(raw_df)}")
    print(f"Number of chunks created: {len(chunk_df)}")

    print("\nSample chunks:")
    print(
        chunk_df[["source", "chunk_id", "chunk_text"]]
        .head(5)
        .to_string(index=False)
    )

    for retrieval_df in retrieval_outputs:
        if retrieval_df.empty:
            continue

        query = retrieval_df.iloc[0]["query"]
        model_name = retrieval_df.iloc[0]["model_name"]
        print(f"\nRetrieval results for query: {query}")
        print(f"Embedding model: {model_name}")
        print(
            retrieval_df[["source", "chunk_id", "cosine_similarity", "chunk_text"]]
            .to_string(index=False)
        )

    print("\nGrounded answers:")
    for item in answers:
        print("\n" + item["answer"])

    print("\nReusable assignment writeup:")
    print(build_assignment_writeup())

    print("\nSaved files:")
    print(f"- {RAW_DOCUMENTS_CSV}")
    print(f"- {CHUNKED_DOCUMENTS_CSV}")
    print(f"- {RETRIEVED_RESULTS_CSV}")


# -----------------------------------------------------------------------------
# 9) Main pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    raw_df = ingest_documents()
    chunk_df = build_chunk_dataframe(raw_df, chunking_method=CHUNKING_METHOD)
    chunk_df = add_embeddings_to_chunks(chunk_df)

    helper_objects = {
        "spacy_model": load_spacy_model(),
        "st_model": load_sentence_transformer_model(),
        "aws_client": get_bedrock_client(),
    }

    retrieval_outputs = []
    answer_outputs = []

    # Use one main retrieval model for the final sample output.
    primary_retrieval_model = "amazon_titan_v2_1024d"

    for query in SAMPLE_QUERIES:
        retrieved_df = retrieve_top_k_chunks(
            query=query,
            chunk_df=chunk_df,
            model_name=primary_retrieval_model,
            top_k=TOP_K,
            helpers=helper_objects,
        )
        retrieval_outputs.append(retrieved_df)

        answer_text = answer_query_with_context(query, retrieved_df, use_live_llm=False)
        answer_outputs.append(
            {
                "query": query,
                "answer": answer_text,
            }
        )

    retrieved_results_df = pd.concat(retrieval_outputs, ignore_index=True)
    save_outputs(raw_df, chunk_df, retrieved_results_df)
    print_summary(raw_df, chunk_df, retrieval_outputs, answer_outputs)


if __name__ == "__main__":
    main()
