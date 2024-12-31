import numpy as np
from scipy.linalg import solve, inv
import matplotlib.pyplot as plt

def generate_test_data(n_points, sigma):
    """
    Generuje dane testowe ze znanymi parametrami i szumem gaussowskim
    Args:
        n_points: liczba punktow pomiarowych
        sigma: odchylenie standardowe szumu
    Returns: x, y (z szumem), y_true (bez szumu), rzeczywiste parametry
    """
    x = np.linspace(0, 10, n_points)
    a_true = np.array([2.0, 3.0, 0.5])  # rzeczywiste wartosci parametrow
    y_true = a_true[0] + a_true[1]*x + a_true[2]*x**2
    y = y_true + np.random.normal(0, sigma, n_points)
    return x, y, y_true, a_true

def create_matrix_A(x, m):
    """
    Tworzy macierz ukladu A, gdzie A[i,j] = x[i]^j
    Args:
        x: punkty pomiarowe
        m: liczba parametrow
    Returns: macierz A
    """
    A = np.zeros((len(x), m))
    for j in range(m):
        A[:, j] = x**j
    return A

def least_squares_fit(x, y, m, G):
    """
    Wykonuje dopasowanie metoda najmniejszych kwadratow z macierza kowariancji
    Args:
        x: punkty pomiarowe
        y: wartosci pomiarowe
        m: liczba parametrow
        G: macierz kowariancji
    Returns: parametry, kowariancja parametrow, reszty, forma kwadratowa
    """
    A = create_matrix_A(x, m)
    G_inv = inv(G)
    ATG_inv_A = A.T @ G_inv @ A
    ATG_inv_y = A.T @ G_inv @ y
    a = solve(ATG_inv_A, ATG_inv_y)  # rozwiazanie rownan normalnych
    C_a = inv(ATG_inv_A)  # macierz kowariancji parametrow
    y_fit = A @ a
    xi = y - y_fit  # reszty
    Q = 0.5 * (xi.T @ G_inv @ xi)  # forma kwadratowa
    return a, C_a, xi, Q

def plot_results(x, y, y_true, y_fit, n, sigma):
    """
    Rysuje dane pomiarowe i dopasowana funkcje
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Dane pomiarowe y', alpha=0.5)
    plt.plot(x, y_true, 'r-', label='F(x) rzeczywista')
    plt.plot(x, y_fit, 'g--', label='F(x) dopasowana')
    plt.legend()
    plt.title(f'n={n}, σ={sigma}')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    filename = f'fit_n{n}_sigma{sigma}.svg'
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.show()

def analyze_parameters(n_values, sigma_values):
    """
    Analizuje wyniki dopasowania dla roznych wartosci n i sigma
    Args:
        n_values: lista testowanych wielkosci probek
        sigma_values: lista testowanych poziomow szumu
    """
    m = 3  # liczba parametrow
    np.random.seed(42)  # dla powtarzalnosci wynikow
    
    for n in n_values:
        for sigma in sigma_values:
            print(f"\nAnaliza dla n={n}, σ={sigma}")
            x, y, y_true, a_true = generate_test_data(n, sigma)
            G = sigma**2 * np.eye(n)  # diagonalna macierz kowariancji
            a, C_a, xi, Q = least_squares_fit(x, y, m, G)
            
            # Wyswietl parametry z niepewnosciami i porownaniem do rzeczywistych wartosci
            print("\nWartosci wspolczynnikow aj:")
            for j, (aj, aj_true) in enumerate(zip(a, a_true)):
                error = np.sqrt(C_a[j,j])
                diff = aj - aj_true
                print(f"a_{j} = {aj:.4f} ± {error:.4f} (rzecz: {aj_true:.4f}, roznica: {diff:.4f})")
            
            print(f"Q = {Q:.4f}")
            
            y_fit = create_matrix_A(x, m) @ a
            plot_results(x, y, y_true, y_fit, n, sigma)


def main():
    # Testowanie roznych wielkosci probek i poziomow szumu
    n_values = [10, 20, 50]  # liczba punktow
    sigma_values = [0.1, 0.5, 1.0]  # poziomy szumu
    analyze_parameters(n_values, sigma_values)

if __name__ == "__main__":
    main()