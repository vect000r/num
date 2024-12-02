import random
import numpy as np
import matplotlib.pyplot as plt

def create_matrix():
    matrix = []
    matrix.append([9, 2, 0, 0])
    matrix.append([2, 4, 1, 0])
    matrix.append([0, 1, 3, 1])
    matrix.append([0, 0, 1, 2])
    return matrix

def create_x():
    x = random.sample(range(500), 4)
    return x

def find_eigen(A, x0, iter_num, tol=10e-6):
    x = x0 / np.linalg.norm(x0)
    lambda_dom = 0
    iterations = []
    errors = []

    for iter in range(1, iter_num + 1):
        x_next = np.dot(A, x)
        x_next_norm = np.linalg.norm(x_next)
        x_next /= x_next_norm 
        lambda_next = np.dot(x_next.T, np.dot(A, x_next))

        error = np.abs(lambda_next - lambda_dom)
        iterations.append(iter)
        errors.append(error)

        if error < tol:
            return lambda_next, x_next, iterations, errors

        x = x_next
        lambda_dom = lambda_next
        
    raise ValueError("Metoda nie jest zbieżna w podanej liczbie iteracji")

def check(A, x0):
    lambda_1, x_1, iterations, errors = find_eigen(A, x0, 1000)
    evalue, evect = np.linalg.eig(A)
    
    absolute_err = np.abs(lambda_1 - evalue[0])
    return absolute_err

def graph(A, x0):
    lambda_1, x_1, iterations, errors = find_eigen(A, x0, 1000)
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, errors, label=f'Metoda potęgowa')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence.svg', dpi=300)
    plt.show()


A = create_matrix()
x0 = create_x()
graph(A, x0)

