"""TF-IDF vectorizer helpers for resume and job text."""
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2)):
    """Returns an initialized TF-IDF vectorizer with standard stop words."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True
    )
