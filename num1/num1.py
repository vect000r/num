import math
import matplotlib.pyplot as plt

def calculate_derivative(f, x, h):
    return (f(x + h) - f(x)) / h

def f(x):
    return math.sin(x**3)


x = 1
h = 0.00001

approximated_derivative = calculate_derivative(f, x, h)

accurate_derivative = 3*(x**2)*(math.cos(x**3))

h_values = [1.0,0.625,0.391,0.244,0.153,0.0954,0.0596,0.0373,0.0233,0.0146,
0.0091,0.00569,0.00356,0.00222,0.00139,0.000868,0.000543,0.000339,
0.000212,0.000133,0.0000829,0.0000518,0.0000324,0.0000202,0.0000126,
0.00000791,0.00000494,0.00000309,0.00000193,0.00000121,0.000000754,
0.000000471,0.000000295,0.000000184,0.000000115,0.000000072,0.000000045,
0.0000000281,0.0000000176,0.000000011,0.00000000687,0.00000000429,
0.00000000268,0.00000000168,0.00000000105,0.000000000655,0.000000000409,
0.000000000256,0.00000000016,0.0000000001]
errors = []


for i in h_values:
    approximated_derivative = calculate_derivative(f, x, i)
    calculation_error= abs(approximated_derivative-accurate_derivative)
    errors.append(calculation_error)

fig, ax = plt.subplots()
ax.plot([math.log10(val) for val in h_values], [math.log10(err) for err in errors])
ax.set(xlabel ='value of h', ylabel='error', title = 'derivative calculation error')
ax.grid()

fig.savefig("error.png")
plt.show()
