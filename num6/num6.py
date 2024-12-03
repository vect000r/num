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

def qr_no_shift(A, tol=10e-9, iter=100):
    M = A.copy()
    eigval = []
    lower_triangle_norms = []

    for it in range(iter):
        Q, R = np.linalg.qr(M)
        M = R @ Q
        
        diagonal = np.diag(M)
        eigval.append(diagonal)

        lower_triangle_norm = np.linalg.norm(np.tril(M, k=-1))
        lower_triangle_norms.append(lower_triangle_norm)

        off_diag = np.sqrt(np.sum(M**2) - np.sum(diagonal**2))
        if off_diag < tol:
            break
    return np.array(eigval), M, lower_triangle_norms
        
def check_power(A, x0):
    lambda_1, x_1, iterations, errors = find_eigen(A, x0, 1000)
    evalue, evect = np.linalg.eig(A)
    
    absolute_err = np.abs(lambda_1 - evalue[0])
    return absolute_err

def check_qr(A):
    eigvals, M, lower_triangle_norms = qr_no_shift(A)
    evalue, evect = np.linalg.eig(A)
    
    absolute_err = np.abs(eigvals[len(eigvals) - 1] - evalue)
    return absolute_err

def graph_power_convergence(A, x0):
    lambda_1, x_1, iterations, errors = find_eigen(A, x0, 1000)
    print(lambda_1)
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, errors)
    plt.yscale('log')
    plt.xlabel('Iteracja')
    plt.ylabel('Błąd (log)')
    plt.title('Zbieżność metody potęgowej')
    plt.grid(True)
    plt.savefig('convergencepower.svg', dpi=300)
    plt.show()

def graph_norms(lower_triangle_norms):
    plt.figure(figsize=(12, 6))
    plt.plot(lower_triangle_norms, label='Norma dolnej trójkątnej części')
    plt.yscale('log')
    plt.xlabel('Liczba iteracji')
    plt.ylabel('Norma dolnej trójkątnej części (log)')
    plt.title('Zbieżność macierzy do postaci trójkątnej górnej')
    plt.legend()
    plt.grid(True)
    plt.savefig('norms.svg', dpi=300)
    plt.show()

def graph_qr_convergence(eigenvalues):
    plt.figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        plt.plot(eigenvalues[:, i], label=f'λ{i+1}')
    plt.yscale('log')
    plt.xlabel('Liczba iteracji')
    plt.ylabel('Wartości własne (log)')
    plt.title('Zbieżność wartości własnych w algorytmie QR')
    plt.legend()
    plt.grid(True)
    plt.savefig('convergenceqr.svg', dpi=300)
    plt.show()

A = create_matrix()
x0 = create_x()
eigvals, M, lower_triangle_norms = qr_no_shift(A)
graph_power_convergence(A, x0)
graph_norms(lower_triangle_norms)
graph_qr_convergence(eigvals)
