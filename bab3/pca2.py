#Example 3.10 Principal Component Analysis Iris
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
#Load Iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target
#Plot Iris data
plt.figure(1)
plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel('sepals length')
plt.ylabel('sepals width')
plt.title('Original Data')
plt.show()

#Perform PCA
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X1 = pca.transform(X)

#Plot PCA data
plt.figure(2)
plt.scatter(X1[:, 0], X1[:, 1], c=y)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA Data')
plt.show()