# Example 3.27 PyCaret-style model comparison
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


iris = load_iris()
models = {
	"Logistic Regression": make_pipeline(
		StandardScaler(), LogisticRegression(max_iter=1000)
	),
	"SVM": make_pipeline(StandardScaler(), SVC()),
	"K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier()),
	"Naive Bayes": GaussianNB(),
	"Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = []
for name, model in models.items():
	scores = cross_val_score(model, iris.data, iris.target, cv=5, scoring="accuracy")
	results.append((name, scores.mean()))
	print(f"{name}: accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")

best_model, best_score = max(results, key=lambda result: result[1])
print(f"Best model: {best_model} (accuracy = {best_score:.3f})")