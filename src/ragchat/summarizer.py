from __future__ import annotations

from collections import Counter, defaultdict

from .data_loader import Message
from .text_utils import keywords, truncate


def summarize_messages(messages: list[Message], title: str = "Segment") -> str:
    if not messages:
        return f"{title}: empty segment."

    text = " ".join(m.text for m in messages)
    top_terms = keywords(text, limit=8)
    by_speaker: dict[str, list[Message]] = defaultdict(list)
    for message in messages:
        by_speaker[message.speaker].append(message)

    facts: list[str] = []
    for speaker in sorted(by_speaker):
        speaker_messages = by_speaker[speaker]
        representative = _representative_messages(speaker_messages, top_terms, limit=2)
        if representative:
            joined = " / ".join(truncate(m.text, 120) for m in representative)
            facts.append(f"{speaker} mentions {joined}")

    days = sorted({m.day for m in messages})
    day_label = f"day {days[0]}" if len(days) == 1 else f"days {days[0]}-{days[-1]}"
    topic_phrase = ", ".join(top_terms[:5]) if top_terms else "general conversation"
    evidence = "; ".join(facts[:3])
    if evidence:
        return f"{title} covers {topic_phrase} across {day_label}. {evidence}."
    return f"{title} covers {topic_phrase} across {day_label}."


def _representative_messages(messages: list[Message], top_terms: list[str], limit: int) -> list[Message]:
    terms = set(top_terms)
    scored = []
    for message in messages:
        words = set(keywords(message.text))
        score = len(words & terms)
        score += min(len(message.text) / 180, 1.0)
        scored.append((score, message.id, message))
    scored.sort(reverse=True)
    return [message for _, _, message in scored[:limit]]


def top_keyword_counts(messages: list[Message], limit: int = 12) -> list[dict]:
    counts = Counter()
    for message in messages:
        counts.update(keywords(message.text))
    return [{"term": term, "count": count} for term, count in counts.most_common(limit)]
