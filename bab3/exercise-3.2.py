# Modifikasi Example 3.2 menggunakan fitur petal (length dan width)
from sklearn import svm, datasets

iris = datasets.load_iris()
# Take the third and fourth features: Petal length and Petal width
X = iris.data[:, 2:]
y = iris.target  # 0: Setosa, 1: Versicolour, 2: Virginica
print(y)

clf = svm.SVC()
clf.fit(X, y)

# Predict the flower for a given Petal length and width
p = clf.predict([[1.4, 0.2]])
print(p)