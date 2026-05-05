# Conversation RAG Chatbot

This project builds a local RAG system over a CSV of conversations where each row is one day's conversation. It processes messages chronologically, creates topic checkpoints when the conversation shifts, creates independent 100-message summaries, extracts an evidence-backed persona JSON, and serves a small Flask chatbot.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_index.py --csv data/conversations.csv --out artifacts
PORT=5055 python -m src.ragchat.app
```

Open `http://localhost:5055`.

You can also ask from the terminal:

```bash
python scripts/ask.py "What kind of person is this user?"
python scripts/ask.py "What are their habits?"
python scripts/ask.py "What did the user say about Portland?"
```

## What Gets Built

Running `scripts/build_index.py` creates:

- `artifacts/messages.json`: every parsed message with global chronological IDs
- `artifacts/topic_checkpoints.json`: topic segments with start/end message IDs and summaries
- `artifacts/hundred_checkpoints.json`: summaries for every 100 chronological messages
- `artifacts/message_chunks.json`: overlapping retrievable chunks of raw messages
- `artifacts/persona.json`: structured persona with evidence message IDs
- `artifacts/topic_pagerank.json` and `artifacts/chunk_pagerank.json`: graph centrality scores used for reranking
- `artifacts/*_index.joblib`: local TF-IDF indexes for topics, chunks, and 100-message checkpoints

No external API is used. Retrieval is local TF-IDF plus a PageRank reranking boost via scikit-learn and NumPy.

## Topic Change Detection

The parser first converts the single-column CSV into chronological messages:

```text
day 1 message 1
day 1 message 2
...
day 2 message 1
```

Topic checkpoints are created in `src/ragchat/segmentation.py`.

The splitter keeps a rolling keyword counter for the active segment. For each new message it computes lexical cosine similarity between the new message keywords and the current segment. A split is created when:

- the day changes and the new message is semantically distant from the prior day/segment, or
- the rolling similarity stays low after enough context has accumulated, or
- a row boundary indicates a new conversation thread.

Each checkpoint stores:

- topic ID
- start/end message IDs
- start/end day
- message count
- extractive summary of that segment only
- top keywords for inspection

This means the full conversation is not treated as one topic. Topic summaries are tied to specific chronological message ranges.

## 100-Message Checkpoints

The 100-message checkpoints are independent from topic checkpoints. They are created after parsing by slicing the full chronological message stream into fixed windows of 100 messages and summarizing each window.

These are useful when a query benefits from broader context even if topic segmentation is granular.

## Retrieval

Query handling is implemented in `src/ragchat/retriever.py`.

For each user question, the system retrieves from three local indexes:

- topic checkpoint summaries
- raw overlapping message chunks
- 100-message checkpoint summaries

The answer combines high-scoring topic summaries with raw message evidence. Persona-style questions such as “What are their habits?” use the persona JSON first, while still returning retrieved evidence for transparency.

## PageRank Reranking

The chatbot remains a RAG system. PageRank improves only the retrieval stage.

During indexing, topic checkpoints and message chunks are converted into local similarity graphs:

- each topic checkpoint or message chunk is a node
- edges connect nodes with high TF-IDF cosine similarity
- PageRank estimates which nodes are central across recurring conversation themes

At query time, TF-IDF still finds query-relevant records. PageRank is then used as a small graph-based boost:

```text
candidate_pool = top query-similar records
final_score = 0.93 * query_similarity + 0.07 * pagerank_centrality
```

This keeps answers query-specific while preferring evidence that is central to repeated patterns in the conversation archive.

## Persona Extraction

Persona extraction lives in `src/ragchat/persona.py`.

It uses rule-based signals from the target speaker, defaulting to `User 1`:

- habits: phrases like “I usually...”, “I always...”, “I love to...”, recurring activity statements
- personal facts: statements like “I am a...”, “I am studying...”, “I have...”, “I live in...”
- personality traits: counted signals such as questions, exclamation marks, appreciation, and empathetic phrasing
- communication style: average message length, short-message ratio, question rate, exclamation rate, emoji count, contraction count, greeting count

Every extracted habit or fact includes an evidence message ID and original message excerpt. Generic low-signal phrases are filtered rather than guessed.

## Cloud Hosting

The repo includes `render.yaml` and a `Procfile` for Render.

1. Push this repository to GitHub.
2. Create a new Render Web Service from the repo.
3. Render will run:

```bash
pip install -r requirements.txt && python scripts/build_index.py --csv data/conversations.csv --out artifacts
```

4. It will start the app with:

```bash
gunicorn src.ragchat.app:app
```

Repository:

```text
GitHub repo URL: https://github.com/Hootsworth/RAG_Project.git
GitHub Pages chatbot URL: https://hootsworth.github.io/RAG_Project/
```

For GitHub Pages, use the static frontend in `docs/`:

1. Go to repository Settings.
2. Open Pages.
3. Choose `Deploy from a branch`.
4. Select branch `main` and folder `/docs`.
5. After Pages deploys, open the Pages URL and paste your deployed backend API URL into the Backend API URL field.

Add the deployed URLs here after deployment:

```text
Hosted chatbot URL: https://hootsworth.github.io/RAG_Project/
Loom demo URL: <paste-loom-url-here>
```

## Demo Checklist

Useful demo questions:

- What kind of person is this user?
- What are their habits?
- How do they talk?
- What did the user say about Portland?
- What conversations mention pets?

The right-side panel in the web app shows retrieved topic summaries and message chunks for each answer.
