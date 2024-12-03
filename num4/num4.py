import numpy as np
import matplotlib.pyplot as plt
import time

def create_matrix(N: int) -> list:
    matrix = np.ones((N, N))
    np.fill_diagonal(matrix, 5)
    np.fill_diagonal(matrix[:-1,1:], 3)
    return matrix

def create_b(N: int) -> np.ndarray:
    return np.full(N, 2)  

def create_banded_matrix(N: int) -> list:
    A = []
    A.append([5] * N)
    A.append([3] * (N-1) + [1])
    return A

def backward_substitution(M: list, b: np.ndarray, N: int) -> tuple:
    z = np.zeros(N)
    q = np.zeros(N)
    
    # More careful handling of last element
    z[N-1] = b[N-1] / M[0][N-1]
    q[N-1] = -1.0 / M[0][N-1]  
    
    # Backward substitution with more careful numeric handling
    for i in range(N-2, -1, -1):
        # Add small epsilon to prevent division by near-zero
        denom = M[0][i] + np.finfo(float).eps
        z[i] = (b[i] - M[1][i] * z[i+1]) / denom
        q[i] = (-1.0 - M[1][i] * q[i+1]) / denom
    
    return z, q

def sherman_morrison(N: int) -> np.ndarray:
    b = create_b(N)
    B = create_banded_matrix(N)
    z, q = backward_substitution(B, b, N)
    
    # More flexible vector choice
    u = np.ones(N)
    
    # More robust correction calculation
    vt_z = np.dot(u, z)
    vt_q = np.dot(u, q)
    
    # Added numerical stability checks
    if abs(1 + vt_q) < np.finfo(float).eps:
        alpha = 0
    else:
        alpha = vt_z / (1 + vt_q)
    
    x = z - alpha * q
    
    return x

def check(N: int) -> np.ndarray:
    A = create_matrix(N)
    b = create_b(N)
    numpy_solution = np.linalg.solve(A, b)
    my_solution = sherman_morrison(N)
    print(my_solution)
    absolute_err = abs(numpy_solution - my_solution)

    return absolute_err

def graph_time():
    N_values = [x for x in range(10, 120)]
    times = []
    numpy_times = []

    # Czasy dla metody wlasnej
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