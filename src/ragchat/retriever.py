from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import Message


RAG_STOP_WORDS = list(ENGLISH_STOP_WORDS | {"user", "users", "message", "messages", "conversation", "conversations"})


def build_message_chunks(messages: list[Message], size: int = 12, overlap: int = 3) -> list[dict]:
    chunks: list[dict] = []
    step = max(1, size - overlap)
    for start in range(0, len(messages), step):
        window = messages[start : start + size]
        if not window:
            continue
        chunks.append(
            {
                "id": len(chunks) + 1,
                "start_message_id": window[0].id,
                "end_message_id": window[-1].id,
                "start_day": window[0].day,
                "end_day": window[-1].day,
                "text": "\n".join(m.as_text() for m in window),
            }
        )
    return chunks


class RagRetriever:
    def __init__(self, artifact_dir: str | Path = "artifacts"):
        self.artifact_dir = Path(artifact_dir)
        self.topic_index = joblib.load(self.artifact_dir / "topic_index.joblib")
        self.chunk_index = joblib.load(self.artifact_dir / "chunk_index.joblib")
        self.hundred_index = joblib.load(self.artifact_dir / "hundred_index.joblib")
        self.persona = json.loads((self.artifact_dir / "persona.json").read_text(encoding="utf-8"))

    def retrieve(self, query: str, topic_k: int = 4, chunk_k: int = 6, hundred_k: int = 2) -> dict:
        return {
            "topics": _search_index(self.topic_index, query, topic_k),
            "chunks": _search_index(self.chunk_index, query, chunk_k),
            "hundred_checkpoints": _search_index(self.hundred_index, query, hundred_k),
        }

    def answer(self, query: str) -> dict:
        retrieval = self.retrieve(query)
        persona_answer = _answer_from_persona(query, self.persona)
        if persona_answer:
            answer = persona_answer
        else:
            answer = _extractive_answer(query, retrieval)
        return {"answer": answer, "retrieval": retrieval}


def fit_index(records: list[dict], text_key: str) -> dict:
    texts = [record[text_key] for record in records]
    vectorizer = TfidfVectorizer(stop_words=RAG_STOP_WORDS, ngram_range=(1, 2), min_df=1, max_features=60000)
    matrix = vectorizer.fit_transform(texts)
    return {"records": records, "text_key": text_key, "vectorizer": vectorizer, "matrix": matrix}


def _search_index(index: dict, query: str, k: int) -> list[dict]:
    if not query.strip():
        return []
    query_vector = index["vectorizer"].transform([query])
    scores = cosine_similarity(query_vector, index["matrix"]).ravel()
    if scores.size == 0:
        return []
    pagerank_scores = np.array([float(record.get("pagerank", 0.0)) for record in index["records"]])
    candidate_count = min(scores.size, max(k * 10, k))
    candidates = np.argsort(scores)[::-1][:candidate_count]
    final_scores = (0.93 * scores) + (0.07 * pagerank_scores)
    top = candidates[np.argsort(final_scores[candidates])[::-1][:k]]
    results = []
    for row in top:
        score = float(scores[row])
        if score <= 0:
            continue
        record = dict(index["records"][int(row)])
        record["score"] = round(score, 4)
        record["final_score"] = round(float(final_scores[row]), 4)
        record["centrality"] = round(float(record.get("pagerank", 0.0)), 4)
        results.append(record)
    return results


def _answer_from_persona(query: str, persona: dict) -> str | None:
    q = query.lower()
    if any(term in q for term in ["habit", "routine", "food", "sleep", "wake"]):
        items = persona.get("habits", [])[:8]
        return _format_persona_items("Habits found from the conversations", items)
    if any(term in q for term in ["talk", "speak", "communication", "tone", "style"]):
        style = persona.get("communication_style", {})
        return (
            "Communication style: "
            f"{style.get('tone_summary', 'unknown')}. "
            f"Average message length is {style.get('average_words_per_message')} words, "
            f"question rate is {style.get('question_rate')}, and exclamation rate is {style.get('exclamation_rate')}."
        )
    if any(term in q for term in ["person", "personality", "traits", "kind of user"]):
        traits = persona.get("personality_traits", [])[:6]
        facts = persona.get("personal_facts", [])[:5]
        trait_text = "; ".join(f"{t['trait']} ({t['rationale']})" for t in traits) or "not enough trait signal"
        fact_text = "; ".join(f"{f['value']} [msg {f['evidence_message_id']}]" for f in facts) or "no stable facts extracted"
        return f"This user appears {trait_text}. Evidence-backed personal facts include: {fact_text}."
    return None


def _format_persona_items(prefix: str, items: list[dict]) -> str:
    if not items:
        return f"{prefix}: no strong evidence found."
    formatted = "; ".join(f"{item['value']} [msg {item['evidence_message_id']}]" for item in items)
    return f"{prefix}: {formatted}."


def _extractive_answer(query: str, retrieval: dict) -> str:
    topics = retrieval.get("topics", [])
    chunks = retrieval.get("chunks", [])
    if not topics and not chunks:
        return "I could not find relevant evidence in the indexed conversations."

    parts = []
    if topics:
        parts.append("Relevant topic checkpoints: " + " ".join(t["summary"] for t in topics[:2]))
    if chunks:
        chunk_lines = []
        for chunk in chunks[:3]:
            chunk_lines.append(
                f"messages {chunk['start_message_id']}-{chunk['end_message_id']}: {chunk['text'].replace(chr(10), ' / ')[:420]}"
            )
        parts.append("Relevant message evidence: " + " ".join(chunk_lines))
    return " ".join(parts)
