from flask import Flask, request, jsonify
from transformers import pipeline
import torch
import os
import threading

app = Flask(__name__)

MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MAX_TEXT_LEN = 512
MAX_BATCH_SIZE = 50

_classifier = None
_model_lock = threading.Lock()


def get_classifier():
    global _classifier
    if _classifier is None:
        with _model_lock:
            if _classifier is None:
                _classifier = pipeline(
                    "text-classification",
                    model=MODEL_ID,
                    truncation=True,
                    max_length=128,
                    torch_dtype=torch.float16,
                )
    return _classifier


def _warmup():
    get_classifier()

threading.Thread(target=_warmup, daemon=True).start()


def classify(text: str) -> dict:
    result = get_classifier()(text)[0]
    return {
        "text": text,
        "label": result["label"].lower(),
        "score": round(result["score"], 4),
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_ready": _classifier is not None})


@app.route("/sentiment", methods=["POST"])
def sentiment():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400
    if len(text) > MAX_TEXT_LEN:
        return jsonify({"error": f"Text exceeds {MAX_TEXT_LEN} character limit"}), 400

    return jsonify(classify(text))


@app.route("/batch", methods=["POST"])
def batch():
    data = request.get_json(silent=True)
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' array in request body"}), 400

    texts = data["texts"]
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be an array of strings"}), 400
    if not texts:
        return jsonify({"error": "'texts' array cannot be empty"}), 400
    if len(texts) > MAX_BATCH_SIZE:
        return jsonify({"error": f"Batch size exceeds {MAX_BATCH_SIZE} items"}), 400

    results = []
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            return jsonify({"error": f"Item at index {i} is not a string"}), 400
        cleaned = t.strip()
        if not cleaned:
            results.append({"text": t, "error": "empty"})
            continue
        if len(cleaned) > MAX_TEXT_LEN:
            results.append({"text": t[:50] + "...", "error": "too_long"})
            continue
        results.append(classify(cleaned))

    return jsonify({"count": len(results), "results": results})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
