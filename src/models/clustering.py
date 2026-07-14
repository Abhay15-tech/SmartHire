"""Job/Role Clustering using K-Means and PCA visualization."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from src import config


class JobClustering:
    def __init__(self, n_clusters=8, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=1024)
        self.cluster_labels = {}  # Map cluster ID to human-readable term/label

    def fit(self, tfidf_matrix):
        """Fit MiniBatchKMeans clustering on the sparse TF-IDF matrix."""
        self.kmeans.fit(tfidf_matrix)
        return self

    def predict(self, tfidf_matrix):
        """Predict cluster IDs for given TF-IDF vectors."""
        return self.kmeans.predict(tfidf_matrix)

    def get_top_terms_per_cluster(self, vectorizer, top_n=8):
        """Return the top TF-IDF words for each cluster centroid to define the 'Role Family'."""
        feature_names = np.array(vectorizer.get_feature_names_out())
        centroids = self.kmeans.cluster_centers_
        top_terms = {}
        for cluster_id in range(self.n_clusters):
            ordered_terms = centroids[cluster_id].argsort()[::-1]
            top_terms[cluster_id] = [feature_names[ind] for ind in ordered_terms[:top_n]]
        return top_terms

    def label_clusters(self, vectorizer):
        """Automatically label clusters with their top 3 TF-IDF terms."""
        top_terms = self.get_top_terms_per_cluster(vectorizer, top_n=3)
        for cluster_id, terms in top_terms.items():
            self.cluster_labels[cluster_id] = " / ".join(terms).title()
        return self.cluster_labels

    def plot_clusters(self, tfidf_matrix, labels, output_path, sample_size=5000):
        """Reduces dimensionality using PCA and saves a 2D scatter plot of job clusters."""
        np.random.seed(self.random_state)
        indices = np.random.choice(tfidf_matrix.shape[0], min(sample_size, tfidf_matrix.shape[0]), replace=False)
        dense_sample = tfidf_matrix[indices].toarray()
        sample_labels = labels[indices]

        # Reduce to 2 dimensions
        pca = PCA(n_components=2, random_state=self.random_state)
        coords = pca.fit_transform(dense_sample)

        plt.figure(figsize=(10, 8), facecolor='#060818')
        ax = plt.axes()
        ax.set_facecolor('#060818')
        
        # Plot each cluster with its own color
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=sample_labels, cmap='plasma', alpha=0.6, s=15)
        
        # Legend styling
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
        ax.add_artist(legend1)
        
        plt.title("Job Role Families (PCA 2D Cluster Projection)", color='white', fontsize=14)
        plt.xlabel("PCA 1", color='white')
        plt.ylabel("PCA 2", color='white')
        plt.tick_params(colors='white')
        
        # Save output
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor='#060818')
        plt.close()

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
