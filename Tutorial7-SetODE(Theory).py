# Tutorial 7 - Set of Ordinary Differential Equations

# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Function Definitions
def dy_dx(x, y):
    return np.array([dy1_dx(x,y), dy2_dx(x,y)])

def dy1_dx(x, y):
    return - x * y[1]

def dy2_dx(x, y):
    return 1.5 * y[0] - 5 * y[1]


#########################

# Forward Euler
# or First Order Truncation
def forward_euler(h, x, y, model):
    dy = model(x, y)
    return y + h * dy

# Runge-Kutta second order (RK2)
# also Second Order Truncation
# Heun's Method
def rk2_heun(h, x, y, model):
    k1 = model(x, y)
    k2 = model(x+h, y+h*k1)
    return y + h/2 * (k1 + k2)

# Midpoint Method
def rk2_midpoint(h, x, y, model):
    return y + h * model(x + h/2, y+ h/2 * model(x, y))

def rk4(h, x, y, model):
    k1 = model(x, y)
    k2 = model(x + h/2, y + h/2 * k1)
    k3 = model(x + h/2, y + h/2 * k2)
    k4 = model(x + h, y + h * k3)
    return y + h * (k1 + 2*k2 + 2*k3 + k4)/6

#########################

def mse(y_exact, y_computed):
    N = len(y_computed)
    mse = np.sum((y_computed - y_exact)**2) / N
    return mse
    

# Given exact equation and initial value
def compute_ode(initial_value=0, 
                h=0.1, 
                duration=100, 
                model=dy_dx, 
                sol=None):
    
    x_axis = np.arange(0, duration, h)
    
    y_euler = [initial_value]
    y_heun = [initial_value]
    y_midpoint = [initial_value]
    y_rk4 = [initial_value]
    
    for i in range(len(x_axis)-1):
        x = x_axis[i]
        
        y_prev_euler = y_euler[-1]
        y_next_euler = forward_euler(h, x, y_prev_euler, model)
        y_euler.append(y_next_euler)
        
        y_prev_heun = y_heun[-1]
        y_next_heun = rk2_heun(h, x, y_prev_heun, model)
        y_heun.append(y_next_heun)
        
        y_prev_mid = y_midpoint[-1]
        y_next_mid = rk2_midpoint(h, x, y_prev_mid, model)
        y_midpoint.append(y_next_mid)
        
        y_prev_rk4 = y_rk4[-1]
        y_next_rk4 = rk4(h, x, y_prev_rk4, model)
        y_rk4.append(y_next_rk4)
    
    if sol != None:
        
        # Compute the solution
        solution = sol(x_axis)
        
        # Compute Mean Square Errors (MSE)
        mse_euler = mse(solution, y_euler)
        mse_heun = mse(solution, y_heun)
        mse_midpoint = mse(solution, y_midpoint)
        mse_rk4 = mse(solution, y_rk4)
        
        print('Mean Square Error (MSE) of Forward Euler Method is', mse_euler)
        print('Mean Square Error (MSE) of RK2 (Heun) Method is', mse_heun)
        print('Mean Square Error (MSE) of RK2 (Midpoint) Method is', mse_midpoint)
        print('Mean Square Error (MSE) of RK4 Method is', mse_rk4)
    

###
# compute_ode(initial_value= np.array([[1],[0]]),
#             h= 0.1,
#             duration= 3,
#             model= dy_dx)
###