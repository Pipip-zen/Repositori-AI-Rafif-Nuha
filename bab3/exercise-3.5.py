# Modifikasi Example 3.7: Train, Save (joblib), Load, dan Predict
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib

X, y = load_iris(return_X_y=True)
print(X)

# 1. Melatih model
clf = GaussianNB()
clf.fit(X, y)

# 2. Menyimpan model ke file (Serialization)
joblib.dump(clf, 'model.pkl')
print("Model berhasil disimpan ke model.pkl")

# 3. Memuat kembali model dari file (Deserialization)
clf2 = joblib.load('model.pkl')
print("Model berhasil dimuat kembali dari file")

# 4. Membuat prediksi dengan model yang dimuat
p = clf2.predict([[5.0, 3.4, 1.5, 0.4]])
print(p)