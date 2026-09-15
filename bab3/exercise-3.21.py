# Modifikasi Example 3.28: Perbandingan regressor pada Diabetes Dataset
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)
regressors = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge()),
    "SVR": make_pipeline(StandardScaler(), SVR()),
    "Random Forest": RandomForestRegressor(random_state=1),
    "Extra Trees": ExtraTreesRegressor(random_state=1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=1),
}

results = []
for name, regressor in regressors.items():
    regressor.fit(X_train, y_train)
    results.append({
        "Regressor": name,
        "R-Squared": regressor.score(X_test, y_test),
    })

models = pd.DataFrame(results).set_index("Regressor").sort_values(
    "R-Squared", ascending=False
)
print(models)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(models.index, models["R-Squared"], marker="o")
plt.xticks(rotation=90)
plt.xlabel("Regressor")
plt.ylabel("R-Squared")
plt.title("LazyRegressor Performance on Diabetes Dataset")
plt.grid(True)
plt.tight_layout()
plt.show()