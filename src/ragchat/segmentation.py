from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from .data_loader import Message
from .summarizer import summarize_messages, top_keyword_counts
from .text_utils import cosine_counts, keyword_counter


@dataclass
class TopicCheckpoint:
    id: int
    start_message_id: int
    end_message_id: int
    start_day: int
    end_day: int
    message_count: int
    summary: str
    keywords: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HundredCheckpoint:
    id: int
    start_message_id: int
    end_message_id: int
    message_count: int
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_topic_checkpoints(messages: list[Message]) -> list[TopicCheckpoint]:
    if not messages:
        return []

    checkpoints: list[TopicCheckpoint] = []
    segment: list[Message] = []
    segment_counter: Counter = Counter()
    weak_shift_streak = 0

    for message in messages:
        message_counter = keyword_counter(message.text)
        day_changed = bool(segment and message.day != segment[-1].day)
        similarity = cosine_counts(segment_counter, message_counter)
        enough_context = len(segment) >= 6
        likely_shift = enough_context and similarity < 0.055 and len(message_counter) >= 2

        if day_changed:
            previous_day_messages = [m for m in segment if m.day == segment[-1].day]
            previous_day_counter = Counter()
            for old in previous_day_messages[-6:]:
                previous_day_counter.update(keyword_counter(old.text))
            day_similarity = cosine_counts(previous_day_counter or segment_counter, message_counter)
            likely_shift = likely_shift or day_similarity < 0.12

        weak_shift_streak = weak_shift_streak + 1 if likely_shift else 0
        should_split = bool(segment and (day_changed or weak_shift_streak >= 2) and len(segment) >= 4)

        if should_split:
            checkpoints.append(_make_topic_checkpoint(len(checkpoints) + 1, segment))
            segment = []
            segment_counter = Counter()
            weak_shift_streak = 0

        segment.append(message)
        segment_counter.update(message_counter)

    if segment:
        checkpoints.append(_make_topic_checkpoint(len(checkpoints) + 1, segment))
    return checkpoints


def build_hundred_checkpoints(messages: list[Message], size: int = 100) -> list[HundredCheckpoint]:
    checkpoints: list[HundredCheckpoint] = []
    for index in range(0, len(messages), size):
        chunk = messages[index : index + size]
        checkpoints.append(
            HundredCheckpoint(
                id=len(checkpoints) + 1,
                start_message_id=chunk[0].id,
                end_message_id=chunk[-1].id,
                message_count=len(chunk),
                summary=summarize_messages(chunk, title=f"100-message checkpoint {len(checkpoints) + 1}"),
            )
        )
    return checkpoints


def _make_topic_checkpoint(topic_id: int, segment: list[Message]) -> TopicCheckpoint:
    return TopicCheckpoint(
        id=topic_id,
        start_message_id=segment[0].id,
        end_message_id=segment[-1].id,
        start_day=segment[0].day,
        end_day=segment[-1].day,
        message_count=len(segment),
        summary=summarize_messages(segment, title=f"Topic {topic_id}"),
        keywords=top_keyword_counts(segment),
    )
