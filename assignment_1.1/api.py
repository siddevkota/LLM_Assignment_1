from flask import Flask, request, jsonify
from nlp_preprocessing import preprocess, compare_lemmas_stems

app = Flask(__name__)


@app.route("/preprocess", methods=["POST"])
def preprocess_text():
    """
    POST JSON {"text": "..."}
    Returns JSON with tokens, lemmas, stems, pos_tags, entities.
    """
    data = request.get_json(force=True)
    text = data.get("text", "")
    return jsonify(preprocess(text))


@app.route("/compare", methods=["POST"])
def compare():
    """
    POST JSON {"words": ["running", "flies", ...]}
    Returns JSON list of {"word","lemma","stem"}.
    """
    data = request.get_json(force=True)
    words = data.get("words", [])
    return jsonify(compare_lemmas_stems(words))


if __name__ == "__main__":
    # start with `python api.py`
    app.run(host="0.0.0.0", port=5000, debug=True)
