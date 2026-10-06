# Example 4.1 Modified: Neuron for Logical OR Operation
from numpy import array, dot, exp, random

# Set up parameters with standard OR truth table inputs and outputs
inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = array([[0, 1, 1, 1]]).T

random.seed(1)
weights = 2 * random.random((2, 1)) - 1


# Define a Single Neuron function
def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output


# Train the Neuron (50,000 iterations)
for iteration in range(50000):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    weights += dot(inputs.T, (outputs - output) * output * (1 - output))

# Test the Neuron with all possible OR inputs
print("Hasil Pengujian Logika OR:")
print("Input [0, 0] -> Output:", neuron(array([0, 0]), weights))
print("Input [0, 1] -> Output:", neuron(array([0, 1]), weights))
print("Input [1, 0] -> Output:", neuron(array([1, 0]), weights))
print("Input [1, 1] -> Output:", neuron(array([1, 1]), weights))