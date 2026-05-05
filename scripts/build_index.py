#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ragchat.pipeline import build_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chronological RAG and persona artifacts.")
    parser.add_argument("--csv", default="data/conversations.csv", help="Single-column CSV where each row is one day's conversation.")
    parser.add_argument("--out", default="artifacts", help="Artifact output directory.")
    parser.add_argument("--speaker", default="User 1", help="Speaker to build persona for.")
    args = parser.parse_args()
    manifest = build_artifacts(args.csv, args.out, args.speaker)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
