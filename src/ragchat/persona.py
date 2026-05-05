from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

from .data_loader import Message
from .text_utils import keywords, normalize, tokens, truncate


FACT_PATTERNS = [
    re.compile(r"\bI(?:'m| am)\s+(?:a|an)\s+([^.!?]{3,90})", re.I),
    re.compile(r"\bI(?:'m| am)\s+studying\s+([^.!?]{3,90})", re.I),
    re.compile(r"\bI(?:'m| am)\s+moving\s+to\s+([^.!?]{3,90})", re.I),
    re.compile(r"\bI have\s+([^.!?]{3,90})", re.I),
    re.compile(r"\bI live(?: in| at)?\s+([^.!?]{3,90})", re.I),
    re.compile(r"\bI moved(?: to| from)?\s+([^.!?]{3,90})", re.I),
    re.compile(r"\bmy (?:name is|job is|favorite [a-z ]+ is)\s+([^.!?]{3,90})", re.I),
]

HABIT_PATTERNS = [
    re.compile(r"\bI (?:usually|always|often|sometimes|never)\s+([^.!?]{3,90})", re.I),
    re.compile(r"\bI (?:love|like|enjoy)\s+(?:to\s+)?([^.!?]{3,90})", re.I),
    re.compile(r"\bI (?:cook|eat|run|walk|read|study|work|sleep|wake)\b([^.!?]{0,90})", re.I),
]

EMOJI_RE = re.compile(
    "[\U0001f300-\U0001f5ff\U0001f600-\U0001f64f\U0001f680-\U0001f6ff\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff]"
)


@dataclass
class EvidenceItem:
    value: str
    evidence_message_id: int
    evidence: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_persona(messages: list[Message], speaker: str = "User 1") -> dict:
    user_messages = [m for m in messages if m.speaker.lower() == speaker.lower()]
    facts = _extract_items(user_messages, FACT_PATTERNS, limit=40)
    habits = _extract_items(user_messages, HABIT_PATTERNS, limit=35)
    traits = _infer_traits(user_messages)
    communication = _communication_style(user_messages)
    interests = _interests(user_messages)

    return {
        "target_speaker": speaker,
        "message_count": len(user_messages),
        "habits": [h.to_dict() for h in habits],
        "personal_facts": [f.to_dict() for f in facts],
        "personality_traits": traits,
        "communication_style": communication,
        "interests": interests,
        "note": "Persona fields are rule-based and include evidence message IDs. Low-signal fields are omitted instead of guessed.",
    }


def _extract_items(messages: list[Message], patterns: list[re.Pattern], limit: int) -> list[EvidenceItem]:
    seen: set[str] = set()
    items: list[EvidenceItem] = []
    for message in messages:
        for pattern in patterns:
            for match in pattern.finditer(message.text):
                value = normalize(match.group(1))
                value = re.sub(r"\s+(and|but|so)$", "", value, flags=re.I).strip(" ,")
                if len(value) < 3 or _too_generic(value):
                    continue
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                items.append(EvidenceItem(value=value, evidence_message_id=message.id, evidence=truncate(message.text, 180)))
                if len(items) >= limit:
                    return items
    return items


def _too_generic(value: str) -> bool:
    generic = {
        "doing well",
        "doing good",
        "good",
        "great",
        "fine",
        "well",
        "not sure",
        "sure",
        "pretty well",
        "doing pretty well",
        "doing well too",
        "doing well, thanks for asking",
        "doing alright",
        "doing great",
        "a lot",
        "day",
        "job",
        "lucky",
    }
    cleaned = value.lower().strip(" .!")
    vague_starts = ("sure ", "glad ", "sorry ", "also ", "more of ")
    return cleaned in generic or cleaned.startswith(vague_starts) or len(cleaned.split()) > 14


def _infer_traits(messages: list[Message]) -> list[dict]:
    if not messages:
        return []
    text = " ".join(m.text for m in messages)
    total = len(messages)
    signals = [
        (
            "curious",
            sum("?" in m.text for m in messages),
            "asks questions frequently",
        ),
        (
            "enthusiastic",
            sum("!" in m.text for m in messages),
            "uses exclamation marks and upbeat reactions",
        ),
        (
            "polite/appreciative",
            len(re.findall(r"\b(thank|thanks|appreciate|nice talking)\b", text, re.I)),
            "uses thanks, appreciation, or courteous closings",
        ),
        (
            "empathetic",
            len(re.findall(r"\b(sorry to hear|miss|hope|glad|wonderful|adorable)\b", text, re.I)),
            "responds with support or warmth",
        ),
    ]
    traits = []
    for trait, count, rationale in signals:
        if count:
            confidence = min(0.95, 0.35 + count / max(total, 1) * 2.5)
            traits.append({"trait": trait, "signal_count": count, "confidence": round(confidence, 2), "rationale": rationale})
    return sorted(traits, key=lambda item: item["signal_count"], reverse=True)


def _communication_style(messages: list[Message]) -> dict:
    if not messages:
        return {}
    lengths = [len(tokens(m.text)) for m in messages]
    question_rate = sum("?" in m.text for m in messages) / len(messages)
    exclamation_rate = sum("!" in m.text for m in messages) / len(messages)
    emoji_count = sum(len(EMOJI_RE.findall(m.text)) for m in messages)
    contraction_count = sum(len(re.findall(r"\b\w+'\w+\b", m.text)) for m in messages)
    greetings = sum(bool(re.search(r"\b(hi|hello|hey)\b", m.text, re.I)) for m in messages)
    return {
        "average_words_per_message": round(sum(lengths) / len(lengths), 2),
        "short_message_ratio": round(sum(length <= 8 for length in lengths) / len(lengths), 2),
        "question_rate": round(question_rate, 2),
        "exclamation_rate": round(exclamation_rate, 2),
        "emoji_count": emoji_count,
        "contraction_count": contraction_count,
        "greeting_count": greetings,
        "tone_summary": _tone_summary(question_rate, exclamation_rate, emoji_count, contraction_count),
    }


def _tone_summary(question_rate: float, exclamation_rate: float, emoji_count: int, contraction_count: int) -> str:
    parts = []
    if question_rate > 0.25:
        parts.append("question-led")
    if exclamation_rate > 0.2:
        parts.append("upbeat")
    if contraction_count:
        parts.append("casual")
    if emoji_count:
        parts.append("emoji-using")
    return ", ".join(parts) if parts else "plain and direct"


def _interests(messages: list[Message]) -> list[dict]:
    counts = Counter()
    for message in messages:
        counts.update(keywords(message.text))
    banned = {"doing", "well", "today", "thing", "things", "something", "someone"}
    return [{"term": term, "count": count} for term, count in counts.most_common(30) if term not in banned][:15]
