# Modifikasi Example 3.20: K-means Clustering dengan make_blobs
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Membangkitkan data tiruan menggunakan make_blobs
X, y = make_blobs(
    n_samples=300, centers=2, n_features=3, random_state=0
)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print("Labels:\n", kmeans.labels_)
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Prediction for new point:\n", kmeans.predict([[12, 3, 1]]))