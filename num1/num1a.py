import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x**3)

def accurate_derivative(x):
    return 3 * x**2 * np.cos(x**3)

def approximated_derivative(f, x, h):
    return (f(x + h) - f(x)) / h

x = 0.2

exact = accurate_derivative(x)

h_values = np.logspace(-10, -1, num=500)

errors_float32 = []
errors_float64 = []

for h in h_values:
    h_float32 = np.float32(h)
    errors_float32.append(abs(approximated_derivative(f, np.float32(x), h_float32)-np.float32(exact)))

for h in h_values:
    h_float64 = np.float64(h)
    errors_float64.append(abs(approximated_derivative(f, np.float64(x), h_float64) - np.float64(exact)))

plt.figure(figsize=(10, 6))

plt.loglog(h_values, errors_float32, label="Derivative approximation error float32", color='red')
plt.loglog(h_values, errors_float64, label="Derivative approximation error float64", color='blue')

plt.xlabel("h (log scale)")
plt.ylabel("Error (log scale)")
plt.savefig("num1a.png")
plt.legend()
plt.grid(True, "both", ls="--")
plt.show()