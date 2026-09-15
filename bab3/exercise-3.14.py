# Modifikasi Example 3.20: K-means Clustering dengan penambahan dua titik data pada setiap grup
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2, 3], [1, 4, 2], [1, 0, 3], [2, 3, 2], [0, 1, 4], \
              [10, 2, 4], [9, 4, 3], [11, 0, 2], [10, 5, 3], [12, 1, 2]])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print("Labels:\n", kmeans.labels_)
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Prediction for new point:\n", kmeans.predict([[12, 3, 1]]))