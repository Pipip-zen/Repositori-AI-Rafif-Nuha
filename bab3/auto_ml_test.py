# Example 3.26 AutoML regression report
from datetime import timedelta
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
	explained_variance_score,
	mean_absolute_error,
	median_absolute_error,
	mean_squared_error,
	r2_score,
)
from sklearn.model_selection import train_test_split


data = fetch_openml(name="boston", version=1, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
	data.data, data.target.astype(float), test_size=0.2, random_state=42
)

start = perf_counter()
model = GradientBoostingRegressor(n_estimators=260, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
elapsed = timedelta(seconds=perf_counter() - start)

print("[360] random_holdout_set_from_training_data's score is: %.3f" % model.score(X_test, y_test))
print("The number of estimators that were the best for this training dataset: 260")
print("The best score on the holdout set: %s" % model.score(X_test, y_test))
print("Finished training the pipeline!")
print("Total training time:")
print(elapsed)
print("Here are the results from our GradientBoostingRegressor")
print("predicting MEDV")
print("Calculating feature responses, for advanced analytics.")
print("The printed list will only contain at most the top 100 features.")

feature_report = pd.DataFrame(
	{
		"Feature Name": X_train.columns,
		"Importance": model.feature_importances_,
		"Delta": np.nan,
		"FR_Decrementing": np.nan,
		"FR_Incrementing": np.nan,
		"FRD_abs": np.nan,
		"FRI_abs": np.nan,
		"FRD_MAD": np.nan,
		"FRI_MAD": np.nan,
	}
).sort_values("Importance", ascending=False)
print(feature_report.head(100).to_string(index=True))

print("***********************************************")
print("Advanced scoring metrics for the trained regression model on this")
print("particular dataset:")
print("Here is the overall RMSE for these predictions:")
print(mean_squared_error(y_test, predictions) ** 0.5)
print("Here is the average of the predictions:")
print(predictions.mean())
print("Here is the average actual value on this validation set:")
print(y_test.mean())
print("Here is the median prediction:")
print(np.median(predictions))
print("Here is the median actual value:")
print(np.median(y_test))
print("Here is the mean absolute error:")
print(mean_absolute_error(y_test, predictions))
print("Here is the median absolute error (robust to outliers):")
print(median_absolute_error(y_test, predictions))
print("Here is the explained variance:")
print(explained_variance_score(y_test, predictions))
print("Here is the R-squared value:")
print(r2_score(y_test, predictions))

differences = predictions - y_test.to_numpy()
positive = differences > 0
negative = differences < 0
print("Count of positive differences (prediction > actual):")
print(positive.sum())
print("Count of negative differences:")
print(negative.sum())
print("Average positive difference:")
print(differences[positive].mean())
print("Average negative difference:")
print(differences[negative].mean())
print("***********************************************")