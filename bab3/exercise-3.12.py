# Modifikasi Example 3.18: Multiple Linear Regression pada Linnerrud Dataset
from sklearn import linear_model
from sklearn.datasets import load_linnerud

# Memuat dataset Linnerud
X, y = load_linnerud(return_X_y=True)
# Mengambil kolom target pertama (misalnya: 'Chins') untuk regresi linear berganda
y_single = y[:, 0]

reg = linear_model.LinearRegression()
reg.fit(X, y_single)
print("Coefficients: \n", reg.coef_)
print("Intercept: \n", reg.intercept_)
# Melakukan prediksi dengan data masukan baru berdimensi 3
pred = reg.predict([[180, 80, 70]])
print("Prediction: \n", pred)