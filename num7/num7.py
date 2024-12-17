import numpy as np
import matplotlib.pyplot as plt

def y(x):
    return 1 / (1 + 10 * x**2)

def generate_points(n):
    x = np.linspace(-1, 1, n + 1)
    y_vals = y(x)
    return x, y_vals

def cubic_spline(x_points, y_points, x_eval):
    """
    Implementacja interpolacji funkcją sklejaną stopnia trzeciego.
    Uwzględnia warunki naturalne s''(x_0) = s''(x_n) = 0.

    :param x_points: Węzły interpolacji (tablica 1D)
    :param y_points: Wartości funkcji w węzłach (tablica 1D)
    :param x_eval: Punkty, w których wyliczamy wartość interpolacji (tablica 1D)
    :return: Wartości interpolacji w punktach x_eval (tablica 1D)
    """
    n = len(x_points) - 1  # Liczba przedziałów
    h = np.diff(x_points)  # Długości przedziałów
    alpha = np.zeros(n + 1)

    # Obliczenie wektora alpha
    for i in range(1, n):
        alpha[i] = (3 / h[i] * (y_points[i + 1] - y_points[i]) -
                    3 / h[i - 1] * (y_points[i] - y_points[i - 1]))

    # Tworzenie macierzy trójdiagonalnej
    l = np.ones(n + 1)  # Główna przekątna
    mu = np.zeros(n)    # Przekątna powyżej głównej
    z = np.zeros(n + 1)  # Prawa strona równania

    l[0] = 1  # Warunek brzegowy s''(x_0) = 0
    for i in range(1, n):
        l[i] = 2 * (x_points[i + 1] - x_points[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    l[n] = 1  # Warunek brzegowy s''(x_n) = 0

    # Rozwiązanie układu równań dla współczynników c
    c = np.zeros(n + 1)
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]

    # Obliczenie współczynników b i d
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = (y_points[i + 1] - y_points[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    # Obliczenie wartości interpolacji w punktach x_eval
    y_eval = np.zeros_like(x_eval)
    for i, x in enumerate(x_eval):
        # Znajdź odpowiedni przedział
        for j in range(n):
            if x_points[j] <= x <= x_points[j + 1]:
                dx = x - x_points[j]
                y_eval[i] = (y_points[j] +
                             b[j] * dx +
                             c[j] * dx**2 +
                             d[j] * dx**3)
                break

    return y_eval

def lagrange_interpolation(x_points, y_points, x_eval):
    """
    Implementacja interpolacji wielomianowej metodą Lagrange'a.

    :param x_points: Węzły interpolacji (tablica 1D)
    :param y_points: Wartości funkcji w węzłach (tablica 1D)
    :param x_eval: Punkty, w których wyliczamy wartość interpolacji (tablica 1D)
    :return: Wartości interpolacji w punktach x_eval (tablica 1D)
    """
    n = len(x_points)
    y_eval = np.zeros_like(x_eval)

    for i in range(n):
        L = np.ones_like(x_eval)
        for j in range(n):
            if i != j:
                L *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
        y_eval += y_points[i] * L

    return y_eval

def plot_results(n):
    x_nodes, y_nodes = generate_points(n)

    # Funkcja rzeczywista i punkty
    x_fine = np.linspace(-1, 1, 500)
    y_fine = y(x_fine)

    # Interpolacja funkcją sklejaną
    y_spline = cubic_spline(x_nodes, y_nodes, x_fine)

    # Interpolacja wielomianowa
    y_lagrange = lagrange_interpolation(x_nodes, y_nodes, x_fine)

    # Wykres interpolacji
    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, y_fine, label="Funkcja $y(x)$", color='black', linewidth=2)
    plt.plot(x_fine, y_spline, label="Interpolacja sklejana $s(x)$", linestyle='--', color='blue')
    plt.plot(x_fine, y_lagrange, label="Interpolacja wielomianowa $W_n(x)$", linestyle='--', color='green')
    plt.scatter(x_nodes, y_nodes, color='red', label="Węzły interpolacji")
    plt.title(f"Porównanie interpolacji dla n={n}")
    plt.legend()
    plt.grid()
    plt.show()

    # Różnice interpolacji
    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, y_fine - y_spline, label="$y(x) - s(x)$", color='purple')
    plt.plot(x_fine, y_fine - y_lagrange, label="$y(x) - W_n(x)$", color='orange')
    plt.title(f"Różnice interpolacji dla n={n}")
    plt.legend()
    plt.grid()
    plt.show()

# Test dla n = 10
plot_results(10)
