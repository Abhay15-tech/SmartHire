"""Script to train and save all models, including K-Means Clustering."""
import pandas as pd
from src import config
from src.features.text_features import build_tfidf_vectorizer
from src.models.classifier import train_classifier, save_model
from src.models.recommender import JobRecommender
from src.models.clustering import JobClustering

def main():
    print("Loading clean data...")
    resumes_df = pd.read_csv(config.RESUMES_CLEAN_CSV)
    jobs_df = pd.read_csv(config.JOBS_CLEAN_CSV)
    
    # Fill NaN just in case
    resumes_df["text"] = resumes_df["text"].fillna("")
    jobs_df["text"] = jobs_df["text"].fillna("")
    
    # 1. Train Resume Classifier
    print("\n=== Training Resume Classifier ===")
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
    
    # 2. Fit K-Means Clustering on Jobs
    print("\n=== Training Job Clustering ===")
    job_vectorizer = build_tfidf_vectorizer(max_features=5000)
    X_jobs = job_vectorizer.fit_transform(jobs_df["text"])
    
    print("Training K-Means (K=8)...")
    clustering = JobClustering(n_clusters=8)
    clustering.fit(X_jobs)
    
    # Label the clusters
    cluster_labels = clustering.label_clusters(job_vectorizer)
    print("Identified Cluster Role Families:")
    for cid, label in cluster_labels.items():
        print(f"  Cluster {cid}: {label}")
        
    clustering_path = config.MODELS_DIR / "job_clustering.pkl"
    clustering.save(clustering_path)
    print(f"Saved clustering model -> {clustering_path}")
    
    # Assign clusters to jobs and update jobs CSV
    print("Assigning cluster labels to job corpus...")
    cluster_ids = clustering.predict(X_jobs)
    jobs_df["cluster_id"] = cluster_ids
    jobs_df["cluster_label"] = [cluster_labels[cid] for cid in cluster_ids]
    
    # Save back to processed jobs clean file with cluster labels
    jobs_df.to_csv(config.JOBS_CLEAN_CSV, index=False)
    print(f"Updated job clean file with cluster assignments -> {config.JOBS_CLEAN_CSV}")
    
    # 3. Train Recommender (using the same vectorized matrix)
    print("\n=== Training Job Recommender ===")
    recommender = JobRecommender()
    recommender.fit(jobs_df, job_vectorizer)
    
    recommender_path = config.MODELS_DIR / "job_recommender.pkl"
    recommender.save(recommender_path)
    print(f"Saved recommender -> {recommender_path}")
    
    # Generate visualization
    print("\n=== Generating Cluster Projection Plot ===")
    reports_dir = config.PROJECT_ROOT / "reports" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_path = reports_dir / "job_clusters.png"
    clustering.plot_clusters(X_jobs, cluster_ids, plot_path)
    print(f"Saved cluster scatter plot -> {plot_path}")
    
    print("\nAll training and visualization tasks complete!")

if __name__ == "__main__":
    main()
