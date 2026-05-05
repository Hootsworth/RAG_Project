from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .pipeline import build_artifacts
from .retriever import RagRetriever


ARTIFACT_DIR = Path("artifacts")
CSV_PATH = Path("data/conversations.csv")

app = Flask(__name__, template_folder="../../templates", static_folder="../../static")
_bot: RagRetriever | None = None


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def get_bot() -> RagRetriever:
    global _bot
    if _bot is None:
        if not (ARTIFACT_DIR / "manifest.json").exists():
            build_artifacts(CSV_PATH, ARTIFACT_DIR)
        _bot = RagRetriever(ARTIFACT_DIR)
    return _bot


@app.get("/")
def index():
    bot = get_bot()
    manifest_path = ARTIFACT_DIR / "manifest.json"
    manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else "{}"
    return render_template("index.html", persona=bot.persona, manifest=manifest)


@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400
    return jsonify(get_bot().answer(question))


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
