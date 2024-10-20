# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Step 2: Prepare your dataset
# Let's create a synthetic dataset (two interleaving half circles)
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# Normalize the dataset
X = StandardScaler().fit_transform(X)

# Step 3: Apply DBSCAN
# Define DBSCAN parameters: eps (maximum distance between two samples for one to be considered as in the neighborhood) and min_samples (number of samples in a neighborhood for a point to be considered as a core point)
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# Step 4: Visualize the clusters
plt.figure(figsize=(8, 6))
# Plot points; points labeled as -1 are considered noise
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
plt.title("DBSCAN Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Step 5: Evaluate the clustering
# Count number of clusters and noise points
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise
n_noise = list(labels).count(-1)

print(f'Estimated number of clusters: {n_clusters}')
print(f'Estimated number of noise points: {n_noise}')
