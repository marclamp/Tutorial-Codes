# Tutorial 6 - Ordinary Differential Equations
# Initial Value Problem

import numpy as np
import matplotlib.pyplot as plt

# f(x_i, y_i)
def dy_dx(x, y):
    return 0.7*y

# Forward Euler
# or First Order Truncation
def forward_euler(h, x, y):
    fxy = dy_dx(x, y)
    return y + h * fxy

# Runge-Kutta second order (RK2)
# also Second Order Truncation
# Heun's Method
def rk2_heun(h, x, y):
    k1 = dy_dx(x, y)
    k2 = dy_dx(x+h, y+h)
    return y + h/2 * (k1 + k2)

# Midpoint Method
def rk2_midpoint(h, x, y):
    return y + h * dy_dx(x+h/2, y+h/2 * dy_dx(x, y))

def rk4(h, x, y):
    k1 = dy_dx(x, y)
    k2 = dy_dx(x + h/2, y + h/2 * k1)
    k3 = dy_dx(x + h/2, y + h/2 * k2)
    k4 = dy_dx(x + h, y + h * k3)
    return y + h * (k1 + 2*k2 + 2*k3 + k4)/6

def mse(exact, computed):
    N = len(computed)
    mse = np.sum((computed - exact)**2) / N
    return mse

# Given exact equation and initial value

initial_value = 1   # y(0) = 1
h = 0.1             # delta_x
x_axis = np.arange(0, 3, h)
y_exact = np.exp(0.7 * x_axis)

y_euler = [initial_value]
y_heun = [initial_value]
y_midpoint = [initial_value]
y_rk4 = [initial_value]

for i in range(len(x_axis)-1):
    x = x_axis[i]
    
    y_prev_euler = y_euler[-1]
    y_next_euler = forward_euler(h, x, y_prev_euler)
    y_euler.append(y_next_euler)
    
    y_prev_heun = y_heun[-1]
    y_next_heun = rk2_heun(h, x, y_prev_heun)
    y_heun.append(y_next_heun)
    
    y_prev_mid = y_midpoint[-1]
    y_next_mid = rk2_midpoint(h, x, y_prev_mid)
    y_midpoint.append(y_next_mid)
    
    y_prev_rk4 = y_rk4[-1]
    y_next_rk4 = rk4(h, x, y_prev_rk4)
    y_rk4.append(y_next_rk4)
    
# Compute Mean Square Errors (MSE)
mse_euler = mse(y_exact, y_euler)
mse_heun = mse(y_exact, y_heun)
mse_midpoint = mse(y_exact, y_midpoint)
mse_rk4 = mse(y_exact, y_rk4)

print('Mean Square Error (MSE) of Forward Euler Method is', mse_euler)
print('Mean Square Error (MSE) of RK2 (Heun) Method is', mse_heun)
print('Mean Square Error (MSE) of RK2 (Midpoint) Method is', mse_midpoint)
print('Mean Square Error (MSE) of RK4 Method is', mse_rk4)

# Plot 
plt.figure()
plt.plot(x_axis, y_exact, '-', label='Exact')
plt.plot(x_axis, y_euler, 'x-', label='Forward Euler')
plt.plot(x_axis, y_heun, 'x-', label='RK2 (Heun)')
plt.plot(x_axis, y_midpoint, 'x-', label='RK2 (Midpoint)')
plt.plot(x_axis, y_rk4, 'x-', label='RK4')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc='upper left')
plt.show()
    