from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .text_utils import normalize


MESSAGE_RE = re.compile(r"^(User\s+\d+):\s*(.*)$")


@dataclass
class Message:
    id: int
    day: int
    day_message_index: int
    speaker: str
    text: str

    def as_text(self) -> str:
        return f"{self.speaker}: {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def load_conversation_rows(csv_path: str | Path) -> list[str]:
    rows: list[str] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            cell = normalize(row[0])
            if cell:
                rows.append(row[0])
    return rows


def parse_messages(csv_path: str | Path) -> list[Message]:
    conversations = load_conversation_rows(csv_path)
    messages: list[Message] = []
    next_id = 1
    for day, conversation in enumerate(conversations, start=1):
        current_speaker: str | None = None
        current_text: list[str] = []
        day_index = 0

        def flush() -> None:
            nonlocal next_id, day_index, current_speaker, current_text
            if current_speaker and current_text:
                day_index += 1
                messages.append(
                    Message(
                        id=next_id,
                        day=day,
                        day_message_index=day_index,
                        speaker=current_speaker,
                        text=normalize(" ".join(current_text)),
                    )
                )
                next_id += 1
            current_speaker = None
            current_text = []

        for raw_line in conversation.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = MESSAGE_RE.match(line)
            if match:
                flush()
                current_speaker = match.group(1)
                current_text = [match.group(2).strip()]
            elif current_speaker:
                current_text.append(line)
        flush()
    return messages
