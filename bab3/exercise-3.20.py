# Modifikasi Example 3.28: Perbandingan classifier pada Breast Cancer Dataset
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVC": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=1),
    "Random Forest": RandomForestClassifier(random_state=1),
    "Extra Trees": ExtraTreesClassifier(random_state=1),
}

results = []
for name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    results.append({
        "Classifier": name,
        "Accuracy": classifier.score(X_test, y_test),
    })

models = pd.DataFrame(results).set_index("Classifier").sort_values(
    "Accuracy", ascending=False
)
print(models)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(models.index, models["Accuracy"], marker="o")
plt.xticks(rotation=90)
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.title("LazyClassifier Performance on Breast Cancer Dataset")
plt.grid(True)
plt.tight_layout()
plt.show()