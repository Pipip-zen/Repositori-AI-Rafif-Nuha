# Modifikasi Example 3.15: Linear Regression dengan penambahan data dan elemen visualisasi plot
import matplotlib.pyplot as plt
from scipy import stats

# Menambahkan lebih banyak titik data pada x dan y
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [3, 5, 5, 6, 7, 8, 9, 11, 12, 14]

slope, intercept, r, p, std_err = stats.linregress(x, y)
print("slope: ", slope)
print("intercept: ", intercept)


def myfunc(x):
  return slope * x + intercept


mymodel = list(map(myfunc, x))

# Plotting dengan label, judul, legenda, dan grid
plt.scatter(x, y, color="blue", label="Data Asli")
plt.plot(x, mymodel, color="red", label="Garis Regresi Linear")

plt.xlabel("Nilai X")
plt.ylabel("Nilai Y")
plt.title("Grafik Analisis Regresi Linear")
plt.legend()
plt.grid(True)

plt.show()