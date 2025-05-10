from flask import Flask, request, jsonify
from embedding_preprocessing import get_embedding, nearest_neighbors

app = Flask(__name__)


@app.route("/embed", methods=["POST"])
def embed():
    """
    Expects JSON: {"word": "..."}
    Returns: {"word": word, "embedding": [...]} or error.
    """
    data = request.get_json(force=True)
    word = data.get("word", "")
    try:
        vec = get_embedding(word)
        return jsonify({"word": word, "embedding": vec})
    except KeyError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/neighbors", methods=["POST"])
def neighbors():
    """
    Expects JSON: {"word": "...", "topn": N}
    Returns: {"word": word, "neighbors": [{"word": w, "similarity": sim}, ...]}
    """
    data = request.get_json(force=True)
    word = data.get("word", "")
    topn = int(data.get("topn", 5))
    try:
        sims = nearest_neighbors(word, topn)
        neighbors_list = [{"word": w, "similarity": float(sim)} for w, sim in sims]
        return jsonify({"word": word, "neighbors": neighbors_list})
    except KeyError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
