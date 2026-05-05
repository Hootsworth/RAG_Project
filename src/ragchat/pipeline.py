from __future__ import annotations

import json
from pathlib import Path

import joblib

from .data_loader import parse_messages
from .persona import build_persona
from .retriever import build_message_chunks, fit_index
from .segmentation import build_hundred_checkpoints, build_topic_checkpoints


def build_artifacts(csv_path: str | Path, out_dir: str | Path = "artifacts", target_speaker: str = "User 1") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    messages = parse_messages(csv_path)
    topics = build_topic_checkpoints(messages)
    hundreds = build_hundred_checkpoints(messages)
    chunks = build_message_chunks(messages)
    persona = build_persona(messages, speaker=target_speaker)

    _write_json(out / "messages.json", [m.to_dict() for m in messages])
    _write_json(out / "topic_checkpoints.json", [t.to_dict() for t in topics])
    _write_json(out / "hundred_checkpoints.json", [h.to_dict() for h in hundreds])
    _write_json(out / "message_chunks.json", chunks)
    _write_json(out / "persona.json", persona)

    topic_records = [t.to_dict() | {"text": t.summary} for t in topics]
    hundred_records = [h.to_dict() | {"text": h.summary} for h in hundreds]
    joblib.dump(fit_index(topic_records, "text"), out / "topic_index.joblib")
    joblib.dump(fit_index(chunks, "text"), out / "chunk_index.joblib")
    joblib.dump(fit_index(hundred_records, "text"), out / "hundred_index.joblib")

    manifest = {
        "csv_path": str(csv_path),
        "message_count": len(messages),
        "topic_checkpoint_count": len(topics),
        "hundred_checkpoint_count": len(hundreds),
        "message_chunk_count": len(chunks),
        "target_speaker": target_speaker,
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
