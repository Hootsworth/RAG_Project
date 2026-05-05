from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

RAG_STOP_WORDS = list(ENGLISH_STOP_WORDS | {"user", "users", "message", "messages", "conversation", "conversations"})


def add_pagerank_scores(records: list[dict], text_key: str = "text", top_n: int = 8, damping: float = 0.85) -> list[dict]:
    if not records:
        return records
    texts = [record.get(text_key, "") for record in records]
    vectorizer = TfidfVectorizer(stop_words=RAG_STOP_WORDS, ngram_range=(1, 2), max_features=50000)
    matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(matrix)
    np.fill_diagonal(similarities, 0.0)

    adjacency = np.zeros_like(similarities)
    for row_index, row in enumerate(similarities):
        if row.size == 0:
            continue
        neighbors = np.argsort(row)[::-1][:top_n]
        for col_index in neighbors:
            if row[col_index] > 0:
                adjacency[row_index, col_index] = row[col_index]

    scores = _pagerank(adjacency, damping=damping)
    if scores.max() > scores.min():
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        normalized = np.ones_like(scores)

    enriched = []
    for record, score in zip(records, normalized):
        item = dict(record)
        item["pagerank"] = round(float(score), 6)
        enriched.append(item)
    return enriched


def _pagerank(adjacency: np.ndarray, damping: float = 0.85, iterations: int = 80, tolerance: float = 1e-8) -> np.ndarray:
    node_count = adjacency.shape[0]
    if node_count == 0:
        return np.array([])

    row_sums = adjacency.sum(axis=1)
    transition = np.zeros_like(adjacency, dtype=float)
    for row_index, total in enumerate(row_sums):
        if total > 0:
            transition[row_index] = adjacency[row_index] / total
        else:
            transition[row_index] = np.full(node_count, 1.0 / node_count)

    scores = np.full(node_count, 1.0 / node_count)
    teleport = np.full(node_count, (1.0 - damping) / node_count)
    for _ in range(iterations):
        updated = teleport + damping * transition.T.dot(scores)
        if np.linalg.norm(updated - scores, ord=1) < tolerance:
            scores = updated
            break
        scores = updated
    return scores
