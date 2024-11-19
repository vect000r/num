import numpy as np
import matplotlib.pyplot as plt
import time

def create_matrix(N):
    matrix = np.ones((N, N))
    np.fill_diagonal(matrix, 5)
    np.fill_diagonal(matrix[:-1,1:], 3)
    return matrix

def create_b(N):
    return np.full((N, 1), 2)  

def sherman_morrison(N):
    b = create_b(N)  
    A = create_matrix(N)  
    
    
    # Rozkład macierzy A na B + uv^T
    u = np.ones((N, 1))
    v = np.ones((N, 1))

    # Macierz B 
    B = np.zeros((N, N))
    np.fill_diagonal(B, 4)  # główna przekątna
    np.fill_diagonal(B[:-1,1:], 2)  # nad przekątną

    # Obliczanie odwrotności B (B^-1)
    B_inv = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            value = (-1.0) ** (j - i) / (4.0 ** (j - i + 1))
            B_inv[i, j] = value

    # Wzór Shermana-Morrisona
    z = B_inv @ b
    vt_B_inv_u = v.T @ B_inv @ u  # skalar
    w = B_inv @ u
    

    return z - ((v.T @ z) * w) / (1 + vt_B_inv_u)

def check(N):
    A = create_matrix(N)
    b = create_b(N)
    numpy_solution = np.linalg.solve(A, b)
    my_solution = sherman_morrison(N)

    absolute_err = abs(numpy_solution - my_solution)

    return absolute_err

def graph_time():
    N_values = [x for x in range(120)]
    times = []

    for N in N_values:
        start = time.time()
        x = sherman_morrison(N)
        end = time.time()
        result = end - start
        times.append(result)
    

    plt.figure(figsize=(10, 6))
    plt.plot(N_values, times, 'b-', label='Custom method', marker = 'o')
    plt.xlabel('Matrix size (N)')
    plt.ylabel('Time taken [s]')
    plt.title('Comparison of time taken to compute the result between a custom method and numpy method')
    plt.grid(True)
    plt.legend()
    plt.savefig('execution_time.png')
    plt.show()

graph_time()

