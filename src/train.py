"""Script to train and save all models."""
import pandas as pd
from src import config
from src.features.text_features import build_tfidf_vectorizer
from src.models.classifier import train_classifier, save_model
from src.models.recommender import JobRecommender

def main():
    print("Loading clean data...")
    resumes_df = pd.read_csv(config.RESUMES_CLEAN_CSV)
    jobs_df = pd.read_csv(config.JOBS_CLEAN_CSV)
    
    # Fill NaN just in case
    resumes_df["text"] = resumes_df["text"].fillna("")
    jobs_df["text"] = jobs_df["text"].fillna("")
    
    # 1. Train Classifier
    print("Building TF-IDF for resumes...")
    resume_vectorizer = build_tfidf_vectorizer(max_features=5000)
    X_res = resume_vectorizer.fit_transform(resumes_df["text"])
    y_res = resumes_df["Category"]
    
    print("Training Logistic Regression classifier...")
    clf = train_classifier(X_res, y_res)
    
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    classifier_path = config.MODELS_DIR / "resume_classifier.pkl"
    save_model(clf, resume_vectorizer, classifier_path)
    print(f"Saved classifier -> {classifier_path}")
    
    # 2. Train Recommender
    print("Building Job Recommender...")
    job_vectorizer = build_tfidf_vectorizer(max_features=10000)
    recommender = JobRecommender()
    recommender.fit(jobs_df, job_vectorizer)
    
    recommender_path = config.MODELS_DIR / "job_recommender.pkl"
    recommender.save(recommender_path)
    print(f"Saved recommender -> {recommender_path}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
