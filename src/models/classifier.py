"""Supervised resume category classifier (train / predict / save / load)."""
import joblib
from sklearn.linear_model import LogisticRegression
from src import config

def train_classifier(X_train, y_train):
    """Train a Logistic Regression classifier on TF-IDF features."""
    clf = LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

def save_model(model, vectorizer, filepath):
    joblib.dump({"model": model, "vectorizer": vectorizer}, filepath)

def load_model(filepath):
    data = joblib.load(filepath)
    return data["model"], data["vectorizer"]

def predict_category(model, vectorizer, text):
    """Predict category for a single text."""
    X = vectorizer.transform([text])
    return model.predict(X)[0]
