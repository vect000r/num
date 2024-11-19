from functools import reduce
import time
import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt

def createMatrix(n: int) -> list:
    A = []
    A.append([0] + [0.3] * (n-1))
    A.append([1.01] * n)
    A.append([0.2 / i for i in range(1, n)] + [0])
    A.append([0.15 / i**3 for i in range(1, n - 1)] + [0] + [0])
    return A

def createX(n: int) -> list:
    x = list(range(1, n + 1))
    return x

def LU(A: list, x: list, n: int):
    
    i = 0
    
    # LU factorization

    for i in range(1, n-2):
        A[0][i] = A[0][i] / A[1][i - 1]
        A[1][i] = A[1][i] - A[0][i] * A[2][i - 1]
        A[2][i] = A[2][i] - A[0][i] * A[3][i - 1]

    A[0][n-2] = A[0][n-2] / A[1][n-3]
    A[1][n-2] = A[1][n-2] - A[0][n-2] * A[2][n-3]
    A[2][n-2] = A[2][n-2] - A[0][n-2] * A[3][n-3]

    A[0][n-1] = A[0][n-1] / A[1][n-2]
    A[1][n-1] = A[1][n-1] - A[0][n-1] * A[2][n-2]

    
def solveA(A: list, x: list, n: int) -> float:
    
    # Forward substitution
    
    for i in range(1, n):
        x[i] = x[i] - A[0][i] * x[i - 1]
    
    # Back substitution
    
    x[n - 1] = x[n - 1] / A[1][n - 1]
    x[n - 2] = (x[n - 2] - A[2][n - 2] * x[n - 1]) / A[1][n - 2]

    for i in range(n - 3, -1, -1):
        x[i] = (x[i] - A[3][i] * x[i + 2] - A[2][i] * x[i + 1]) / A[1][i]

    return x

def graphTimes(values: list):
    times = []
    numpy_times = []    
    for value in values:
        A = createMatrix(value)
        x = createX(value)
        
        start = time.time()
        LU(A, x, value)
        solveA(A, x, value)
        end = time.time()
        times.append(end - start)

        start = time.time()
        A_np = np.array(A)
        x_np = np.array(x)
        linalg.lu(A_np)
        end = time.time()
        numpy_times.append(end - start)

        
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(values, times, 'b-', label='Custom method', marker = 'o')
    plt.plot(values, numpy_times, 'r-', label='Numpy method', marker = 'o')
    plt.xlabel('Matrix size (N)')
    plt.ylabel('Time taken [s]')
    plt.title('Comparison of time taken to compute the result between a custom method and numpy method')
    plt.grid(True)
    plt.legend()
    plt.savefig('execution_time.png')
    plt.show()


values = [x for x in range(3,303)]
graphTimes(values)

A = createMatrix(300)
x = createX(300)

LU(A, x, 300)
solution = solveA(A, x, 300)
determinant = reduce(lambda x, y: x * y, A[1])
print(determinant)
print(solution)