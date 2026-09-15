#Example 3.3 Python SVM Iris Classifications
from sklearn import svm, datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

clf = svm.SVC()
clf.fit(X, y)

#Predict the flower for a given Sepal length and width
p = clf.predict([[5.4, 3.2]])
print(p)