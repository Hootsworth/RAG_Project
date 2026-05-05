#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ragchat.retriever import RagRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local conversation RAG chatbot.")
    parser.add_argument("question")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    bot = RagRetriever(args.artifacts)
    print(json.dumps(bot.answer(args.question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
