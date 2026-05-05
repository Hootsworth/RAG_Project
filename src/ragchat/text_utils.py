from __future__ import annotations

import re
from collections import Counter


STOPWORDS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "but", "by", "can", "could",
    "did", "do", "does", "doing", "for", "from", "get", "go", "going",
    "good", "got", "great", "had", "has", "have", "he", "her", "here",
    "him", "his", "how", "i", "if", "in", "is", "it", "its", "just",
    "i'm", "i've", "i'll", "you're", "that's", "it's", "like", "me", "my", "no", "not", "of", "on", "or", "our", "out",
    "really", "she", "so", "some", "that", "the", "their", "them", "then",
    "there", "they", "this", "to", "too", "up", "very", "was", "we",
    "well", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "yeah", "yes", "you", "your", "thanks", "thank",
}


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokens(text: str) -> list[str]:
    return [t.lower().strip("'") for t in TOKEN_RE.findall(text or "")]


def keywords(text: str, limit: int | None = None) -> list[str]:
    words = [t for t in tokens(text) if len(t) > 2 and t not in STOPWORDS]
    counts = Counter(words)
    result = [word for word, _ in counts.most_common(limit)]
    return result


def keyword_counter(text: str) -> Counter:
    return Counter(keywords(text))


def cosine_counts(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = sum(v * v for v in a.values()) ** 0.5
    norm_b = sum(v * v for v in b.values()) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def truncate(text: str, chars: int = 220) -> str:
    text = normalize(text)
    if len(text) <= chars:
        return text
    return text[: chars - 3].rstrip() + "..."
