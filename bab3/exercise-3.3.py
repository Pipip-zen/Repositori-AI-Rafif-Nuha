# Modifikasi Example 3.4 untuk menampilkan scatter plot dari dua fitur pertama
from matplotlib import pyplot
import pandas as pd
from sklearn import svm

url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
df = pd.read_csv(url)

print(df.shape)
print(df.head(10))
print(df.tail(10))
print(df.describe())

# count NAN
print(df.isna().sum().sum())

# drop NAN values
df = df.dropna()
print(df.groupby("species").size())

# histograms
df.hist()
pyplot.show()

# Scatter plot khusus untuk dua fitur pertama (sepal_length vs sepal_width)
X = df.values[:, :2]
s = df["species"]
d = dict([(y, x) for x, y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]

pyplot.figure()
pyplot.scatter(X[:, 0], X[:, 1], c=y)
pyplot.xlabel("sepal_length")
pyplot.ylabel("sepal_width")
pyplot.title("Scatter Plot of Sepal Length vs Sepal Width")
pyplot.show()

clf = svm.SVC()
clf.fit(X, y)

# Predict the flower for a given Sepal length and width
p = clf.predict([[5.4, 3.2]])
print(p)