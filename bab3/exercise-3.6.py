# Modifikasi Example 3.8: LinearDiscriminantAnalysis dengan 2000 sampel dan 6 fitur
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X, y = make_classification(
    n_samples=2000,
    n_features=6,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)
print(X)
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0, 0, 0]]))