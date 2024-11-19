import numpy as np
import matplotlib.pyplot as plt
import time

def create_matrix(N):
    matrix = np.ones((N, N))
    np.fill_diagonal(matrix, 5)
    np.fill_diagonal(matrix[:-1,1:], 3)
    return matrix

def create_b(N):
    return np.full(N, 2)  

def create_banded_matrix(N):
    A = []
    A.append([4] * N)
    A.append([2] * (N-1) + [0])
    return A

def backward_substitution(M, b, N):
    # Backward substitution do wyznaczenia wektorów z i q
    z = [0] * N
    q = [0] * N

    z[N - 1] = b[N - 1] / M[0][N - 1]
    q[N - 1] = 1 / M[0][N - 1]

    for i in range(N - 2, -1, -1):
        z[i] = (b[i] - M[1][i] * z[i + 1]) / M[0][i]
        q[i] = (0 - M[1][i] * q[i + 1]) / M[0][i]

    return z, q

def sherman_morrison(N):
    b = create_b(N)

    # Tworzenie macierzy pasmowej
    M = create_banded_matrix(N)

    # Rozwiązywanie równań
    z, q = backward_substitution(M, b, N)

    # Wyliczanie końcowego wyniku
    # w = z - ((v.T @ z) / (1 + v.T @ q)) * q
    vT_z = sum(z)  
    vT_q = sum(q)  

    w = [z[i] - ((vT_z / (1 + vT_q)) * q[i]) for i in range(N)]

    return w

def check(N):
    A = create_matrix(N)
    b = create_b(N)
    numpy_solution = np.linalg.solve(A, b)
    my_solution = sherman_morrison(N)

    absolute_err = abs(numpy_solution - my_solution)

    return absolute_err

def graph_time():
    N_values = [x for x in range(10, 120)]
    times = []
    numpy_times = []

    # Czasy dla metody własnej
    for N in N_values:
        start = time.time()
        sherman_morrison(N)
        end = time.time()
        times.append(end - start)
    
    # Czasy dla metody numpy
    for N in N_values:
        A = create_matrix(N)
        b = create_b(N)
        start = time.time()
        np.linalg.solve(A, b)
        end = time.time()
        numpy_times.append(end - start)

    plt.figure(figsize=(12, 8))
    plt.plot(N_values, times, 'b-o', label='Custom Method', markersize=6, linewidth=1.5)
    plt.plot(N_values, numpy_times, 'r-s', label='Numpy Method', markersize=6, linewidth=1.5)
    plt.xlabel('Matrix Size (N)', fontsize=14)
    plt.ylabel('Execution Time [s]', fontsize=14)
    plt.title('Comparison of Execution Time: Custom vs Numpy Methods', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(fontsize=12)
    plt.savefig('execution_time.svg', dpi=300)
    plt.show()

print(f"Absolute error: {check(120)}")
graph_time()




