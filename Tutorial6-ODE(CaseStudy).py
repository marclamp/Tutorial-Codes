# Tutorial 6 - Ordinary Differential Equations
# Lower Back Pain (LBP) in Bus Drivers [Initial Value Problem]

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Define known constants
E = 5       # MPa
eta = 1.9   # MPa * s (η)
sigma_0 = 6 # MPa (σ_0)
f = 0.1     # Hz

omega = 2 * np.pi * f

# Function Definitions

# computation of the exact solution
def e_exact(t):
    return (sigma_0 / (E**2 + eta**2 * omega**2)) * \
        (omega * eta * np.exp(-E*t/eta) + E*np.sin(omega*t) - \
         (omega*eta*np.cos(omega*t)))

# helper function σ(t)
def sigma_t(t):
    return sigma_0 * np.sin(omega * t)

# Kelvin model, dϵ(t)/dt 
def kelvin(t, et):
    return - (E/eta) * et + sigma_t(t)/eta

# Non-Linear Kelvin Model, Enc = E + 3 * ϵ^2
def nlkm(t, et):
    Enc = E + 3 * et**2
    return - (Enc/eta) * et + sigma_t(t)/eta

### Optional Exercise, SLS and SNLS ###

# helper function dσ(t)
def d_sigma(t):
    return sigma_0 * omega * np.cos(omega * t)

# Standard Linear Solid Model
def sls(t, et, E1=E, E2=10):
    return (1/(E1+E2) * d_sigma(t)) + \
        (E2/(E1+E2) * (sigma_t(t)/eta - (E1/eta) * et))

# Standard Non-Linear Solid Model
# Enc = E + 3 * ϵ^2
def snls(t, et):
    return sls(t, et, E1=E+3*et**2, E2=10+3*et**2)

#########################

# Forward Euler
# or First Order Truncation
def forward_euler(dt, t, et, model):
    det = model(t, et)
    return et + dt * det

# Runge-Kutta second order (RK2)
# also Second Order Truncation
# Heun's Method
def rk2_heun(dt, t, et, model):
    k1 = model(t, et)
    k2 = model(t+dt, et+dt*k1)
    return et + dt/2 * (k1 + k2)

# Midpoint Method
def rk2_midpoint(dt, t, et, model):
    return et + dt * model(t+dt/2, et+dt/2 * model(t, et))

def rk4(dt, t, et, model):
    k1 = model(t, et)
    k2 = model(t + dt/2, et + dt/2 * k1)
    k3 = model(t + dt/2, et + dt/2 * k2)
    k4 = model(t + dt, et + dt * k3)
    return et + dt * (k1 + 2*k2 + 2*k3 + k4)/6

#########################

def mse(et_exact, et_computed):
    N = len(et_computed)
    mse = np.sum((et_computed - et_exact)**2) / N
    return mse
    
# function to extract the last cycle
# returns the last cycle start and end indicies
def last_cycle(et):
    y = np.array(et)
    peaks,_ = find_peaks(y)
    peaks_neg,_ = find_peaks(-y)
    return peaks_neg[-2], peaks_neg[-1], peaks[-1] - peaks_neg[-2]

def hysteresis_area(deformation, stress, idx_max):
    stress_hys = stress + np.abs(np.min(stress))
    al = aul = 0
    
    for i in range(idx_max):
        al += (stress_hys[i+1] + stress_hys[i]) * \
            np.abs(deformation[i+1] - deformation[i])/2
    
    for i in range(idx_max, len(stress)-1):
        aul += (stress_hys[i+1] + stress_hys[i]) * \
            np.abs(deformation[i+1] - deformation[i])/2
       
    area = (al - aul) / al
    return area

# Given exact equation and initial value
def compute_ode(initial_value=0, 
                dt=0.1, 
                duration=100, 
                model=kelvin, 
                sol=None,
                hysteresis=True):
    
    t_axis = np.arange(0, duration, dt)
    
    et_euler = [initial_value]
    et_heun = [initial_value]
    et_midpoint = [initial_value]
    et_rk4 = [initial_value]
    
    for i in range(len(t_axis)-1):
        t = t_axis[i]
        
        et_prev_euler = et_euler[-1]
        et_next_euler = forward_euler(dt, t, et_prev_euler, model)
        et_euler.append(et_next_euler)
        
        et_prev_heun = et_heun[-1]
        et_next_heun = rk2_heun(dt, t, et_prev_heun, model)
        et_heun.append(et_next_heun)
        
        et_prev_mid = et_midpoint[-1]
        et_next_mid = rk2_midpoint(dt, t, et_prev_mid, model)
        et_midpoint.append(et_next_mid)
        
        et_prev_rk4 = et_rk4[-1]
        et_next_rk4 = rk4(dt, t, et_prev_rk4, model)
        et_rk4.append(et_next_rk4)
    
    if sol != None:
        
        # Compute the solution
        solution = sol(t_axis)
        
        # Compute Mean Square Errors (MSE)
        mse_euler = mse(solution, et_euler)
        mse_heun = mse(solution, et_heun)
        mse_midpoint = mse(solution, et_midpoint)
        mse_rk4 = mse(solution, et_rk4)
        
        print('Mean Square Error (MSE) of Forward Euler Method is', mse_euler)
        print('Mean Square Error (MSE) of RK2 (Heun) Method is', mse_heun)
        print('Mean Square Error (MSE) of RK2 (Midpoint) Method is', mse_midpoint)
        print('Mean Square Error (MSE) of RK4 Method is', mse_rk4)
    
    if hysteresis:
        
        # Find Last Cycle
        lc_start, lc_end, idx_max = last_cycle(et_euler)
        
        # Calculate λ (Area under Curve)
        sigma_t_axis = sigma_t(t_axis)
        lamda_euler = hysteresis_area(et_euler[lc_start:lc_end], sigma_t_axis[lc_start:lc_end], idx_max)
        lamda_heun = hysteresis_area(et_heun[lc_start:lc_end], sigma_t_axis[lc_start:lc_end], idx_max)
        lamda_midpoint = hysteresis_area(et_midpoint[lc_start:lc_end], sigma_t_axis[lc_start:lc_end], idx_max)
        lamda_rk4 = hysteresis_area(et_rk4[lc_start:lc_end], sigma_t_axis[lc_start:lc_end], idx_max)
        
        print("Hysteresis parameter, λ, Forward Euler Method is", lamda_euler)
        print("Hysteresis parameter, λ, RK2 Heun's Method is", lamda_heun)
        print("Hysteresis parameter, λ, RK2 Midpoint Method is", lamda_midpoint)
        print("Hysteresis parameter, λ, RK4 Method is", lamda_rk4)
        print()
        
        # Plot
        plt.figure()
        plt.plot(t_axis[lc_start:lc_end], et_euler[lc_start:lc_end], 'x-', label='Forward Euler')
        plt.plot(t_axis[lc_start:lc_end], et_heun[lc_start:lc_end], 'x-', label='RK2 (Heun)')
        plt.plot(t_axis[lc_start:lc_end], et_midpoint[lc_start:lc_end], 'x-', label='RK2 (Midpoint)')
        plt.plot(t_axis[lc_start:lc_end], et_rk4[lc_start:lc_end], 'x-', label='RK4')
        plt.ylabel('Deformation, ϵ')
        plt.xlabel('time (s)')
        plt.legend(loc='best')
        plt.title(model.__name__)
        plt.show()
        
        # Hysteresis plot (ϵ on the x axis, and σ on the y axis)
        plt.figure()
        plt.plot(et_euler[lc_start:lc_end], sigma_t_axis[lc_start:lc_end], 'x-', label='Forward Euler')
        plt.plot(et_heun[lc_start:lc_end], sigma_t_axis[lc_start:lc_end], 'x-', label='RK2 (Heun)')
        plt.plot(et_midpoint[lc_start:lc_end], sigma_t_axis[lc_start:lc_end], 'x-', label='RK2 (Midpoint)')
        plt.plot(et_rk4[lc_start:lc_end], sigma_t_axis[lc_start:lc_end], 'x-', label='RK4')
        plt.xlabel('Deformation, ϵ')
        plt.ylabel('Stress, σ (MPa)')
        plt.legend(loc='best')
        plt.title(model.__name__)
        plt.show()

print("Linear Kelvin Model")
compute_ode(sol=e_exact, model=kelvin)

# print("Non-Linear Kelvin Model")
# compute_ode(model=nlkm)

# print("Standard Linear Solid Model")
# compute_ode(model=sls)

# print("Standard Non-Linear Solid Model")
# compute_ode(model=snls)

###
def dy_dx(x, y):
    return 0.7*y
def ex(x):
    return np.exp(0.7 * x)

print("Test code, y = e^0.7x")
compute_ode(initial_value= 1,
            duration= 3,
            model= dy_dx, 
            sol= ex,
            hysteresis= False)
###

def fx(x, y):
    # first derivative 
    return 7

def exact(x):
    #return exact solution computation (if any)
    return 7 * x

print('For midterm')
compute_ode(initial_value=0,
            model= fx,
            sol= exact,         # comment out if none
            hysteresis= False)

###