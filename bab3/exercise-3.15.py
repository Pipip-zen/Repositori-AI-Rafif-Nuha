# Modifikasi Example 3.20: K-means Clustering dengan penambahan kelompok titik data ketiga
from sklearn.cluster import KMeans
import numpy as np

X = np.array([
    [1, 2, 3], [1, 4, 2], [1, 0, 3],       # Grup 1
    [10, 2, 4], [9, 4, 3], [11, 0, 2],    # Grup 2
    [20, 20, 20], [21, 19, 22], [19, 21, 20] # Grup 3 (Grup baru)
])

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print("Labels:\n", kmeans.labels_)
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Prediction for new point:\n", kmeans.predict([[20, 20, 21]]))