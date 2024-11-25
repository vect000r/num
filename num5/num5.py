import numpy as np
import matplotlib.pyplot as plt

def create_matrix(N: int, d: float) -> np.ndarray:
    A = np.zeros((N, N))
    np.fill_diagonal(A, d)
    
    for i in range(N-1):
        A[i, i+1] = 0.5
        A[i+1, i] = 0.5
    
    for i in range(N-2):
        A[i, i+2] = 0.1
        A[i+2, i] = 0.1
    
    return A

def create_banded_matrix(N: int, d: float) -> np.ndarray:
    matrix = np.zeros((5, N))
    matrix[0, 2:] = 0.1    
    matrix[1, 1:] = 0.5    
    matrix[2, :] = d       
    matrix[3, :-1] = 0.5   
    matrix[4, :-2] = 0.1   
    
    return matrix

def create_b(N: int) -> list:
    return list(range(1, N + 1))


def gauss_seidel_method(A: np.ndarray, b: list, max_iter: int =1000 , tol: float = 1e-6):
        N = len(b)
        x = np.zeros(N)
        iterations = []
        errors = []
        
        second_sub = A[0]  
        first_sub = A[1]   
        main_diag = A[2]   
        first_super = A[3] 
        second_super = A[4] 
        
        for iter in range(max_iter):
            x_old = x.copy()
            
            for i in range(N):
                sum_term = 0.0
                
                # Składnik z drugiej podprzekątnej (0.1)
                if i >= 2:
                    sum_term += second_sub[i] * x[i-2]
                
                # Składnik z pierwszej podprzekątnej (0.5)
                if i >= 1:
                    sum_term += first_sub[i] * x[i-1]
                
                # Składnik z pierwszej nadprzekątnej (0.5)
                if i < N-1:
                    sum_term += first_super[i] * x_old[i+1]
                
                # Składnik z drugiej nadprzekątnej (0.1)
                if i < N-2:
                    sum_term += second_super[i] * x_old[i+2]
                
                # Aktualizacja x[i]
                x[i] = (b[i] - sum_term) / main_diag[i]
            
            # Obliczenie błędu 
            error = np.linalg.norm(x - x_old)
            iterations.append(iter)
            errors.append(error)
            
            # Sprawdzenie zbieżności
            if error < tol:
                break
        
        return x, iterations, errors

def jacobi_method(A: np.ndarray, b: list, max_iter: int = 1000, tol: float = 1e-6):
    N = len(b)
    x = np.zeros(N)  
    iterations = []
    errors = []
    
    second_sub = A[0]   
    first_sub = A[1]    
    main_diag = A[2]    
    first_super = A[3]  
    second_super = A[4] 
    
    for iter in range(max_iter):
        x_new = np.zeros(N)
        
        # Obliczenie nowych wartości
        for i in range(N):
            sum_term = 0.0
            
            # Składnik z drugiej podprzekątnej (0.1)
            if i >= 2:
                sum_term += second_sub[i] * x[i-2]
            
            # Składnik z pierwszej podprzekątnej (0.5)
            if i >= 1:
                sum_term += first_sub[i] * x[i-1]
            
            # Składnik z pierwszej nadprzekątnej (0.5)
            if i < N-1:
                sum_term += first_super[i] * x[i+1]
            
            # Składnik z drugiej nadprzekątnej (0.1)
            if i < N-2:
                sum_term += second_super[i] * x[i+2]
            
            # Obliczenie nowej wartości x[i]
            x_new[i] = (b[i] - sum_term) / main_diag[i]
        
        # Obliczenie błędu
        error = np.linalg.norm(x_new - x)
        iterations.append(iter)
        errors.append(error)
        
        # Sprawdzenie zbieżności
        if error < tol:
            break
            
        # Aktualizacja x
        x = x_new.copy()
    
    return x, iterations, errors

def check(A, b, x_jacobi, x_gs):
    numpy_result = np.linalg.solve(A, b)
    abe_jacobi = abs(numpy_result - x_jacobi)
    abe_gs = abs(numpy_result - x_gs)
    return abe_jacobi, abe_gs

def compare(x_jacobi, x_gs):
    return abs(x_jacobi - x_gs)   

def graph(N, d_values):
    b = create_b(200)
    plt.figure(figsize=(12, 6))

    for d in d_values:
        A = create_banded_matrix(N, d)
        
        x_jacobi, iter_j, errors_j = jacobi_method(A, b)
        x_gs, iter_gs, errors_gs = gauss_seidel_method(A, b)

        plt.plot(iter_j, errors_j, label=f'Jacobi (d={d})', linestyle='--')
        plt.plot(iter_gs, errors_gs, label=f'Gauss-Seidel (d={d})')

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence.svg', dpi=300)
    plt.show()

N = 200
d_values = [1.0, 1.5, 2.0, 2.5]
graph(N, d_values)