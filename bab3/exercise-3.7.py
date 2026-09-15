# Modifikasi Example 3.10: Principal Component Analysis pada Breast Cancer Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, decomposition

# Load Breast Cancer data
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# Plot Original Data (menggunakan 2 fitur pertama)
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.title("Original Breast Cancer Data")
plt.show()

# Perform PCA
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X1 = pca.transform(X)

# Plot PCA data
plt.figure(2)
plt.scatter(X1[:, 0], X1[:, 1], c=y)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA Breast Cancer Data")
plt.show()