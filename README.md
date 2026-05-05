# RAG Journal: Conversation RAG + Persona Chatbot

RAG Journal is an end-to-end local RAG system for a CSV of daily conversations. It processes messages chronologically, creates topic checkpoints when the conversation changes, builds independent 100-message checkpoints, extracts an evidence-backed user persona, and serves a chatbot through a GitHub Pages frontend with a Render-hosted Flask backend.

## Submission Links

```text
GitHub Repository: https://github.com/Hootsworth/RAG_Project
Live Chatbot:      https://hootsworth.github.io/RAG_Project/
Render Backend:   https://rag-project-bufn.onrender.com
Loom Demo:         <add Loom link here>
```

> Note: The backend is hosted on Render Free. If the service is asleep, the first request can take 30-60 seconds while it wakes up.

## Screenshots

Add screenshots here before final submission.

### Empty Chatbot State

```text
<add screenshot: chatbot before asking a question>
```

### Chatbot With Response + Evidence

```text
<add screenshot: chatbot after asking a question>
```

### Code Snippets

Use CodeSnap screenshots for the snippets below.

```text
<add CodeSnap: topic checkpoint splitting logic from src/ragchat/segmentation.py>
<add CodeSnap: retrieval + PageRank reranking from src/ragchat/retriever.py>
<add CodeSnap: persona extraction from src/ragchat/persona.py>
<add CodeSnap: artifact build pipeline from src/ragchat/pipeline.py>
```

## Features

- Chronological message parsing from a single-column CSV
- Topic checkpointing whenever the conversation topic changes
- Independent 100-message checkpoint summaries
- Local TF-IDF retrieval over topic summaries, message chunks, and 100-message summaries
- PageRank graph reranking for better evidence ordering
- Rule-based persona extraction with evidence message IDs
- GitHub Pages chatbot frontend
- Render Flask backend API
- Suggested question chips
- Backend health/wake status
- Copy-answer button
- Evidence panel with relevance and centrality scores

## Project Structure

```text
RAG_Project/
├── data/
│   └── conversations.csv
├── docs/
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── scripts/
│   ├── ask.py
│   └── build_index.py
├── src/ragchat/
│   ├── app.py
│   ├── data_loader.py
│   ├── pagerank.py
│   ├── persona.py
│   ├── pipeline.py
│   ├── retriever.py
│   ├── segmentation.py
│   ├── summarizer.py
│   └── text_utils.py
├── templates/
├── static/
├── render.yaml
├── Procfile
├── requirements.txt
└── README.md
```

## How To Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_index.py --csv data/conversations.csv --out artifacts
PORT=5055 python -m src.ragchat.app
```

Open:

```text
http://localhost:5055
```

Ask from the terminal:

```bash
python scripts/ask.py "What kind of person is this user?"
python scripts/ask.py "What are their habits?"
python scripts/ask.py "How do they talk?"
python scripts/ask.py "What did the user say about Portland?"
```

## What Gets Built

Running `scripts/build_index.py` creates local artifacts:

- `artifacts/messages.json`: parsed chronological messages
- `artifacts/topic_checkpoints.json`: topic segments with summaries
- `artifacts/hundred_checkpoints.json`: summaries for every 100 chronological messages
- `artifacts/message_chunks.json`: overlapping raw message chunks
- `artifacts/persona.json`: structured persona with evidence IDs
- `artifacts/topic_pagerank.json`: graph centrality for topic checkpoints
- `artifacts/chunk_pagerank.json`: graph centrality for message chunks
- `artifacts/*_index.joblib`: local TF-IDF indexes

No external LLM or paid API is required. The system uses Python, Flask, scikit-learn, NumPy, and local rule-based logic.

## Topic Change Detection

The CSV contains one conversation per row. The loader parses each row message by message and assigns a global chronological message ID.

Topic splitting happens in `src/ragchat/segmentation.py`.

The splitter maintains a rolling keyword counter for the current segment. For every new message, it compares the message keywords against the active segment using cosine similarity over keyword counts.

A new topic checkpoint is created when:

- the day changes and the new message is distant from the prior segment
- the rolling similarity stays low after enough context exists
- a row boundary indicates a new conversation thread

Each topic checkpoint stores:

- topic ID
- start/end message ID
- start/end day
- message count
- summary for that topic segment only
- top keywords

This avoids treating the full conversation history as one topic.

## 100-Message Checkpoints

100-message checkpoints are independent from topic checkpoints. After all messages are parsed chronologically, the system slices the global message stream into fixed windows of 100 messages and summarizes each window.

These summaries provide broader context when a query spans multiple small topic segments.

## Retrieval

Retrieval is implemented in `src/ragchat/retriever.py`.

For every question, the system searches:

- topic checkpoint summaries
- overlapping raw message chunks
- 100-message checkpoint summaries

The answer is generated from retrieved topic summaries, retrieved raw chunks, and persona data when relevant.

## PageRank Reranking

The system is still a RAG system. PageRank improves the retrieval stage only.

During indexing:

- topic checkpoints become graph nodes
- message chunks become graph nodes
- edges connect nodes with high TF-IDF cosine similarity
- PageRank gives each node a centrality score

At query time:

```text
candidate_pool = top query-similar records
final_score = 0.93 * query_similarity + 0.07 * pagerank_centrality
```

TF-IDF still controls relevance. PageRank only reranks already-relevant candidates so central but unrelated memories do not override the query.

The frontend shows both:

- `relevance`: query similarity
- `centrality`: PageRank score

## Persona Extraction

Persona extraction is implemented in `src/ragchat/persona.py`.

It extracts structured JSON for `User 1`:

- habits
- personal facts
- personality traits
- communication style
- interests

Persona signals are rule-based and evidence-backed. Extracted facts and habits include:

- value
- evidence message ID
- source message excerpt

The system avoids guessing when signal is weak.

## Chatbot Frontend

The frontend is hosted with GitHub Pages from the `docs/` folder:

```text
https://hootsworth.github.io/RAG_Project/
```

Frontend features:

- minimal journal-style UI
- suggested question chips
- backend status check
- Render wake-up messaging
- copy answer button
- retrieved evidence panel
- relevance and centrality labels

The frontend calls the Render backend:

```text
https://rag-project-bufn.onrender.com
```

## Backend Hosting

The backend is a Flask app hosted on Render.

Render build command:

```bash
pip install -r requirements.txt && python scripts/build_index.py --csv data/conversations.csv --out artifacts
```

Render start command:

```bash
gunicorn src.ragchat.app:app
```

Health check:

```text
https://rag-project-bufn.onrender.com/health
```

Expected response:

```json
{"status":"ok"}
```

## Demo Questions

Use these in the Loom demo:

- What kind of person is this user?
- What are their habits?
- How do they talk?
- Summarize recurring topics
- What did the user say about Portland?

## Loom Demo Plan

Recommended 1-2 minute flow:

1. Open the chatbot: `https://hootsworth.github.io/RAG_Project/`
2. Show the backend status card.
3. Mention Render may take 30-60 seconds to wake up.
4. Ask: `What kind of person is this user?`
5. Show the answer and copy button.
6. Show the evidence panel with relevance and centrality.
7. Ask: `What are their habits?`
8. Briefly show the README/code snippets.

Add final Loom link here:

```text
Loom Demo: <add Loom link here>
```
