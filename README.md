# Fantasy Baseball Chunking and Embeddings

A single-file Python project for comparing chunking strategies on fantasy baseball article text and creating SentenceTransformer embeddings for the final chunks.

## What it does
- Accepts a list of article texts with optional source names
- Cleans text lightly
- Compares fixed-size, sentence-based, and recursive-overlap chunking
- Creates embeddings for the final chosen chunks
- Saves results to JSON and CSV

## Run
1. Install dependencies:
   ```bash
   pip install pandas nltk langchain-text-splitters sentence-transformers
   ```
2. Edit the `articles` list in `chunking_embeddings_fantasy_baseball.py`
3. Run:
   ```bash
   python3 chunking_embeddings_fantasy_baseball.py
   ```
# fantasy-baseball
