# Example 4.2 Modified: Multiple Layer Perceptron with 3 Inputs
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.neural_network import MLPClassifier

# Data dengan 3 input (3 fitur) dan 8 sampel kombinasi logika OR
X = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
]
y = [0, 1, 1, 1, 1, 1, 1, 1]

clf = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
)
clf.fit(X, y)

print(clf.predict([[2.0, 2.0, 2.0], [-1.0, -2.0, -1.0]]))
print([coef.shape for coef in clf.coefs_])