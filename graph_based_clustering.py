# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

# Step 2: Prepare your dataset
# Let's create a synthetic dataset (two interleaving half circles)
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# Step 3: Visualize the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title("Dataset: Two Interleaving Half Circles")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Step 4: Apply Spectral Clustering
n_clusters = 2  # Number of clusters
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
labels = spectral_clustering.fit_predict(X)

# Step 5: Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
plt.title("Spectral Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Step 6: Evaluate the clustering
silhouette_avg = silhouette_score(X, labels)
print(f'Silhouette Score: {silhouette_avg:.2f}')
