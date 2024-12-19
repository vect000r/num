import numpy as np
import matplotlib.pyplot as plt

def y(x):
    return 1 / (1 + 10 * x**2)

def generate_points(n: int) -> tuple:
    x = np.linspace(-1, 1, n + 1)
    y_vals = y(x)
    return x, y_vals

def cubic_spline(x_points: np.ndarray, y_points: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    Implementacja interpolacji funkcja sklejana stopnia trzeciego.
    Uwzglednia warunki naturalne s''(x_0) = s''(x_n) = 0.

    :param x_points: Wezly interpolacji (tablica 1D)
    :param y_points: Wartosci funkcji w wezlach (tablica 1D)
    :param x_eval: Punkty, w ktorych wyliczamy wartosc interpolacji (tablica 1D)
    :return: Wartosci interpolacji w punktach x_eval (tablica 1D)
    """
    num_intervals = len(x_points) - 1  # Liczba przedzialow
    interval_lengths = np.diff(x_points)  # Dlugosci przedzialow
    alpha = np.zeros(num_intervals + 1)

    # Obliczenie wektora alpha
    for i in range(1, num_intervals):
        alpha[i] = (3 / interval_lengths[i] * (y_points[i + 1] - y_points[i]) -
                    3 / interval_lengths[i - 1] * (y_points[i] - y_points[i - 1]))

    # Tworzenie macierzy trojdiagonalnej
    main_diag = np.ones(num_intervals + 1)  # Glowna przekatna
    upper_diag = np.zeros(num_intervals)    # Przekatna powyzej glownej
    rhs = np.zeros(num_intervals + 1)  # Prawa strona rownania

    main_diag[0] = 0  # Warunek brzegowy s''(x_0) = 0
    for i in range(1, num_intervals):
        main_diag[i] = 2 * (x_points[i + 1] - x_points[i - 1]) - interval_lengths[i - 1] * upper_diag[i - 1]
        upper_diag[i] = interval_lengths[i] / main_diag[i]
        rhs[i] = (alpha[i] - interval_lengths[i - 1] * rhs[i - 1]) / main_diag[i]
    main_diag[num_intervals] = 0  # Warunek brzegowy s''(x_n) = 0

    # Rozwiazanie ukladu rownan dla wspolczynnikow c
    c_coeffs = np.zeros(num_intervals + 1)
    for j in range(num_intervals - 1, -1, -1):
        c_coeffs[j] = rhs[j] - upper_diag[j] * c_coeffs[j + 1]

    # Obliczenie wspolczynnikow b i d
    b_coeffs = np.zeros(num_intervals)
    d_coeffs = np.zeros(num_intervals)
    for i in range(num_intervals):
        b_coeffs[i] = (y_points[i + 1] - y_points[i]) / interval_lengths[i] - interval_lengths[i] * (c_coeffs[i + 1] + 2 * c_coeffs[i]) / 3
        d_coeffs[i] = (c_coeffs[i + 1] - c_coeffs[i]) / (3 * interval_lengths[i])

    # Obliczenie wartosci interpolacji w punktach x_eval
    y_eval = np.zeros_like(x_eval)
    for i, x in enumerate(x_eval):
        # Znajdz odpowiedni przedzial
        for j in range(num_intervals):
            if x_points[j] <= x <= x_points[j + 1]:
                dx = x - x_points[j]
                y_eval[i] = (y_points[j] +
                             b_coeffs[j] * dx +
                             c_coeffs[j] * dx**2 +
                             d_coeffs[j] * dx**3)
                break

    return y_eval

def lagrange_interpolation(x_points: np.ndarray, y_points: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    Implementacja interpolacji wielomianowej metoda Lagrange'a.

    :param x_points: Wezly interpolacji (tablica 1D)
    :param y_points: Wartosci funkcji w wezlach (tablica 1D)
    :param x_eval: Punkty, w ktorych wyliczamy wartosc interpolacji (tablica 1D)
    :return: Wartosci interpolacji w punktach x_eval (tablica 1D)
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

def plot_results(n: int) -> None:
    x_nodes, y_nodes = generate_points(n)

    # Funkcja rzeczywista i punkty
    x_fine = np.linspace(-1, 1, 500)
    y_fine = y(x_fine)

    # Interpolacja funkcja sklejana
    y_spline = cubic_spline(x_nodes, y_nodes, x_fine)

    # Interpolacja wielomianowa
    y_lagrange = lagrange_interpolation(x_nodes, y_nodes, x_fine)

    # Wykres interpolacji
    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, y_fine, label="Funkcja $y(x)$", color='black', linewidth=2)
    plt.plot(x_fine, y_spline, label="Interpolacja sklejana $s(x)$", linestyle='--', color='blue')
    plt.plot(x_fine, y_lagrange, label="Interpolacja wielomianowa $W_n(x)$", linestyle='--', color='green')
    plt.scatter(x_nodes, y_nodes, color='red', label="Wezly interpolacji")
    plt.title(f"Porownanie interpolacji dla n={n}")
    plt.legend()
    plt.grid()
    plt.savefig("interpolation.svg", dpi=300)
    plt.show()

    # Roznice interpolacji
    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, y_fine - y_spline, label="$y(x) - s(x)$", color='purple')
    plt.plot(x_fine, y_fine - y_lagrange, label="$y(x) - W_n(x)$", color='orange')
    plt.title(f"Roznice interpolacji dla n={n}")
    plt.legend()
    plt.grid()
    plt.savefig("interpolationdiff.svg", dpi=300)
    plt.show()

# Test dla n = 10
plot_results(5)
