import spacy
from nltk.stem import PorterStemmer
from nltk import download


#download("punkt")


nlp = spacy.load("en_core_web_sm")


stemmer = PorterStemmer()


def tokenize(text: str):
    """Return list of tokens."""
    return [token.text for token in nlp(text)]


def lemmatize(text: str):
    """Return list of lemmas."""
    return [token.lemma_ for token in nlp(text)]


def stem(text: str):
    """Return list of stems via PorterStemmer."""
    return [stemmer.stem(token.text) for token in nlp(text)]


def pos_tag(text: str):
    """Return list of (token, POS) tuples."""
    return [(token.text, token.pos_) for token in nlp(text)]


def ner(text: str):
    """Return list of (entity, label) tuples."""
    return [(ent.text, ent.label_) for ent in nlp(text).ents]


def preprocess(text: str):
    """Run all preprocessing steps."""
    return {
        "tokens": tokenize(text),
        "lemmas": lemmatize(text),
        "stems": stem(text),
        "pos_tags": pos_tag(text),
        "entities": ner(text),
    }


def compare_lemmas_stems(words: list[str]):
    """
    Given a list of words, return their lemma vs. stem.
    """
    comparisons = []
    for w in words:
        doc = nlp(w)
        lemma = doc[0].lemma_
        stem_ = stemmer.stem(w)
        comparisons.append({"word": w, "lemma": lemma, "stem": stem_})
    return comparisons
