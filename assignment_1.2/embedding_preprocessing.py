import gensim.downloader as api

# Load GloVe embeddings (50-dimensional)
model = api.load("glove-wiki-gigaword-50")


def get_embedding(word: str) -> list[float]:
    """
    Return the GloVe embedding for a given word.
    Raises KeyError if the word is not in the vocabulary.
    """
    if word in model:
        return model[word].tolist()
    else:
        raise KeyError(f"Word '{word}' not in vocabulary.")


def nearest_neighbors(word: str, topn: int = 5) -> list[tuple[str, float]]:
    """
    Return a list of (neighbor_word, similarity) tuples.
    """
    if word not in model:
        raise KeyError(f"Word '{word}' not in vocabulary.")
    return model.most_similar(word, topn=topn)
