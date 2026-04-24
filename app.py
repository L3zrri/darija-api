from flask import Flask, request, jsonify
from transformers import pipeline
import os
import threading

app = Flask(__name__)

MODEL_ID = "BenhamdaneNawfal/sentiment-analysis-darija"
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
                )
    return _classifier


# Warm up model on startup in background so first request isn't slow
def _warmup():
    get_classifier()

threading.Thread(target=_warmup, daemon=True).start()


@app.route("/health", methods=["GET"])
def health():
    ready = _classifier is not None
    return jsonify({"status": "ok", "model_ready": ready})


@app.route("/sentiment", methods=["POST"])
def sentiment():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400
    if len(text) > 512:
        return jsonify({"error": "Text exceeds 512 character limit"}), 400

    clf = get_classifier()
    result = clf(text)[0]

    label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}
    label = label_map.get(result["label"], result["label"].lower())

    return jsonify({
        "text": text,
        "label": label,
        "score": round(result["score"], 4),
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
