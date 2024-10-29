# Tutorial 8 - Second Order ODEs
# IVP (Initial Value Problem)
# BVP (Boundary Value Problem)
# Gist: Conversion to Set of ODEs

# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Second Order IVP
# define auxiliary variables

def f(x,y):
    return 2*x + np.log(y[0]) + 3*y[1]

def d2y_dx2(x, y):
    return np.array([dy1_dx(x,y), dy2_dx(x,y)])

def dy1_dx(x, y):
    return y[1]

def dy2_dx(x, y):
    return f(x,y)

#########################

# Forward Euler
# or First Order Truncation
def forward_euler(h, x, y, model):
    dy = model(x, y)
    return y + h * dy

def mse(y_exact, y_computed):
    N = len(y_computed)
    mse = np.sum((y_computed - y_exact)**2) / N
    return mse

#########################

# Given exact equation and initial value
def compute_ode(initial_value=0, 
                h=0.1, 
                duration=100, 
                model=None, 
                sol=None):
    
    x_axis = np.arange(0, duration, h)
    
    y_euler = [initial_value]
    
    for i in range(len(x_axis)-1):
        x = x_axis[i]
        
        y_prev_euler = y_euler[-1]
        y_next_euler = forward_euler(h, x, y_prev_euler, model)
        y_euler.append(y_next_euler)
            
    if sol != None:
        
        # Compute the solution
        solution = sol(x_axis)
        
        # Compute Mean Square Errors (MSE)
        mse_euler = mse(solution, y_euler)
        
        print('Mean Square Error (MSE) of Forward Euler Method is', mse_euler)
    
    return np.array(y_euler)

# Test Run (2nd Order IVP)
solu = compute_ode(initial_value= [3, 5],
                   h= 0.2,
                   duration= 5,
                   model= d2y_dx2)
