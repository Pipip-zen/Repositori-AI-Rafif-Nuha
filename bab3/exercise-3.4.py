# Modifikasi Example 3.6 untuk memplot histogram fitur tertentu
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer = load_breast_cancer()
X = cancer.data  # All of the features
y = cancer.target  # All of the labels

# Konversi data ke dalam DataFrame Pandas dan beri nama kolom sesuai feature_names
df = pd.DataFrame(X, columns=cancer.feature_names)

# Memplot histogram untuk fitur radius, area (size), texture, dan smoothness (menggunakan fitur mean)
selected_features = ["mean radius", "mean area", "mean texture", "mean smoothness"]
df[selected_features].hist(bins=20, figsize=(10, 8))
plt.suptitle("Histogram of Radius, Area, Texture, and Smoothness")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20
)
clf = SVC()
clf.fit(X_train, y_train)

# Prediction
y_predict = clf.predict(X_test)

# Print Confusion Matrix and Classification Report
from sklearn.metrics import classification_report, confusion_matrix

cm = np.array(confusion_matrix(y_test, y_predict, labels=[1, 0]))
confusion = pd.DataFrame(
    cm,
    index=["is_cancer", "is_healthy"],
    columns=["predicted_cancer", "predicted_healthy"],
)
print(confusion)
print(classification_report(y_test, y_predict))