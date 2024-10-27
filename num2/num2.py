import numpy as np
from numpy.linalg import norm, cond, solve

A1 = np.array([
    [5.8267103432, 1.0419816676, 0.4517861296, -0.2246976350, 0.7150286064],
    [1.0419816676, 5.8150823499, -0.8642832971, 0.6610711416, -0.3874139415],
    [0.4517861296, -0.8642832971, 1.5136472691, -0.8512078774, 0.6771688230],
    [-0.2246976350, 0.6610711416, -0.8512078774, 5.3014166511, 0.5228116055],
    [0.7150286064, -0.3874139415, 0.6771688230, 0.5228116055, 3.5431433879]
])

A2 = np.array([
    [5.4763986379, 1.6846933459, 0.3136661779, -1.0597154562, 0.0083249547],
    [1.6846933459, 4.6359087874, -0.6108766748, 2.1930659258, 0.9091647433],
    [0.3136661779, -0.6108766748, 1.4591897081, -1.1804364456, 0.3985316185],
    [-1.0597154562, 2.1930659258, -1.1804364456, 3.3110327980, -1.1617171573],
    [0.0083249547, 0.9091647433, 0.3985316185, -1.1617171573, 2.1174700695]
])

b = np.array([-2.8634904630, -4.8216733374, -4.2958468309, -0.0877703331, -2.0223464006])

X1 = solve(A1, b)
X2 = solve(A2, b)
print(f"Rozwiązanie bazowe macierzy A1:\n{X1}")
print(f"Rozwiązanie bazowe macierzy A2:\n{X2}")

def generate_perturbation(size, magnitude = 1e-6):
    db = np.random.normal(0, magnitude, size)
    db = db * (magnitude / norm(db)) # skalowanie do pożądanej normy
    return db

perturbation = b + generate_perturbation(len(b))

dX1 = solve(A1, perturbation)
dX2 = solve(A2, perturbation)

print(f"Rozwiązanie zaburzone macierzy A1:\n{dX1}")
print(f"Rozwiązanie zaburzone macierzy A2:\n{dX2}")


differenceX1 = abs(X1 - dX1)
differenceX2 = abs(X2 - dX2)
print(f"Różnica pomiędzy rozwiązaniem bazowym a zaburzonym A1:\n{differenceX1}")
print(f"Różnica pomiędzy rozwiązaniem bazowym a zaburzonym A2:\n{differenceX2}")

condA1 = cond(A1)
condA2 = cond(A2)

print(f"Wskaźnik uwarunkowania dla macierzy A1:\n{condA1}")
print(f"Wskaźnik uwarunkowania dla macierzy A2:\n{condA2}")