"""Unsupervised job recommender (cosine-similarity job ranking)."""
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src import config

class JobRecommender:
    def __init__(self, vectorizer=None, tfidf_matrix=None, job_corpus=None):
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.job_corpus = job_corpus
    
    def fit(self, jobs_df, vectorizer):
        """Fit the recommender on the job corpus."""
        self.job_corpus = jobs_df.reset_index(drop=True)
        self.vectorizer = vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(self.job_corpus["text"])
        
    def recommend(self, resume_text, top_n=5):
        """Recommend top N jobs for a given resume text."""
        resume_vec = self.vectorizer.transform([resume_text])
        sim_scores = cosine_similarity(resume_vec, self.tfidf_matrix).flatten()
        top_indices = sim_scores.argsort()[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            job = self.job_corpus.iloc[idx].to_dict()
            job["similarity"] = float(sim_scores[idx])
            results.append(job)
        return results

    def save(self, filepath):
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
