import numpy as np
import matplotlib.pyplot as plt
import time
import num3


def create_matrix(N):
    matrix = np.ones((N, N))
    np.fill_diagonal(matrix, 5)
    np.fill_diagonal(matrix[:-1,1:], 3)
    return matrix

def create_b(N):
    return np.full((N, 1), 2)  

def create_banded_matrix(N):
    A = []
    A.append([0] + [2] * N)
    A.append([4] * N)
    return A

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
    
    #B_banded = create_banded_matrix(N)
    #num3.LU(B_banded, b, N)
    z = num3.solveA(B, b, N)
    q = num3.solveA(B, u, N)
    
    w = z -((v.T @ z) / (1 + (v.T @ q))) * q
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

    for N in N_values:
        start = time.time()
        sherman_morrison(N)
        end = time.time()
        result = end - start
        times.append(result)
    
    for N in N_values:
        A = create_matrix(N)
        b = create_b(N)
        
        
        start = time.time()
        np.linalg.solve(A, b)
        end = time.time()
        numpy_times.append(end - start)



    plt.figure(figsize=(10, 6))
    plt.plot(N_values, times, 'b-', label='Custom method', marker = 'o')
    plt.plot(N_values, numpy_times, 'r-', label='Numpy method', marker = 'o')
    plt.xlabel('Matrix size (N)')
    plt.ylabel('Time taken [s]')
    plt.title('Comparison of time taken to compute the result between a custom method and numpy method')
    plt.grid(True)
    plt.legend()
    plt.savefig('num4/execution_time.png')
    plt.show()

print(f"Absolute error: {check(120)}")
graph_time()

