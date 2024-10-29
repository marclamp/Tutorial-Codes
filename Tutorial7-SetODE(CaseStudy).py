# Tutorial 7 - Set of Ordinary Differential Equations
# COVID-19 Pandemic Models
# SIR Model
# S - Susceptible individuals
# I - Infected individuals
# R - Recovered individuals

# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Declare known variables
beta = 0.35         # 1/day ("infective" rate)
gamma = 0.2         # 1/day (recovery rate)

n = 4e6             # individuals (total population)
i_init = 1          # individuals (infected)
r_init = 0          # individuals (recovered)

s_init = n - i_init - r_init


# Refined variables
v = 5000            # vaccination rate (day 60), for 60 days
beta_drop = 0.05    # after lockdown (day 90), for 30 days

vac_day = 120       # day at which vaccinations start
vac_dur = 60        # duration at which vaccination occur
lock_day = 90       # day at which lockdown protocols start
lock_dur = 30       # duration of lockdown

# Function Definitions

# Basic Model (SIR)
# y is an array arranged as [s i r]
# returns the evaluation of the RHS 
# of the SIR Model
def d_dt(x, y, func_args=None):
    s = y[0]
    i = y[1]
    r = y[2]
    beta = func_args['beta']
    gamma = func_args['gamma']
    n = func_args['n']
    
    ds_dt = -beta * (s*i/n)
    di_dt = beta * (s*i/n) - gamma*i
    dr_dt = gamma*i
    
    return np.array([ds_dt, di_dt, dr_dt])

# Refined Model
def d_dt_refined(x, y, func_args=None):
    s = y[0]
    i = y[1]
    r = y[2]
    beta = func_args['beta']
    gamma = func_args['gamma']
    n = func_args['n']
    v = 0
    
    if x > vac_day and x < (vac_day + vac_dur):
        v = func_args['v']
    
    if x > lock_day and x < (lock_day + lock_dur):
        beta = func_args['beta_drop']
        
    ds_dt = -beta * (s*i/n) - v
    di_dt = beta * (s*i/n) - gamma*i
    dr_dt = gamma*i + v
    
    return np.array([ds_dt, di_dt, dr_dt])

#########################
# Reused code from previous tutorial
# only change is the inclusion of func_args
# ToDo: include func_args in previous code

# Forward Euler
# or First Order Truncation
def forward_euler(h, x, y, model, func_args=None):
    dy = model(x, y, func_args)
    return y + h * dy

# Runge-Kutta second order (RK2)
# also Second Order Truncation
# Heun's Method
def rk2_heun(h, x, y, model, func_args=None):
    k1 = model(x, y, func_args)
    k2 = model(x+h, y+h*k1, func_args)
    return y + h/2 * (k1 + k2)

# Midpoint Method
def rk2_midpoint(h, x, y, model, func_args=None):
    return y + h * model(x + h/2, y+ h/2 * model(x, y, func_args), func_args)

def rk4(h, x, y, model, func_args=None):
    k1 = model(x, y, func_args)
    k2 = model(x + h/2, y + h/2 * k1, func_args)
    k3 = model(x + h/2, y + h/2 * k2, func_args)
    k4 = model(x + h, y + h * k3, func_args)
    return y + h * (k1 + 2*k2 + 2*k3 + k4)/6

#########################

# def mse(y_exact, y_computed):
#     N = len(y_computed)
#     mse = np.sum((y_computed - y_exact)**2) / N
#     return mse 

# Given exact equation and initial value
def compute_ode(initial_value, 
                h, 
                duration, 
                model=None, 
                func_args=None,
                sol=None,
                plot=False):
    
    x_axis = np.arange(0, duration, h)
    
    y_euler = [initial_value]
    y_heun = [initial_value]
    y_midpoint = [initial_value]
    y_rk4 = [initial_value]
    
    for i in range(len(x_axis)-1):
        x = x_axis[i]
        
        y_prev_euler = y_euler[-1]
        y_next_euler = forward_euler(h, x, y_prev_euler, model, func_args)
        y_euler.append(y_next_euler)
        
        y_prev_heun = y_heun[-1]
        y_next_heun = rk2_heun(h, x, y_prev_heun, model, func_args)
        y_heun.append(y_next_heun)
        
        y_prev_mid = y_midpoint[-1]
        y_next_mid = rk2_midpoint(h, x, y_prev_mid, model, func_args)
        y_midpoint.append(y_next_mid)
        
        y_prev_rk4 = y_rk4[-1]
        y_next_rk4 = rk4(h, x, y_prev_rk4, model, func_args)
        y_rk4.append(y_next_rk4)
    
    if plot:
        y_euler = np.array(y_euler)
        y_heun = np.array(y_heun)
        y_midpoint = np.array(y_midpoint)
        y_rk4 = np.array(y_rk4)
        
        plt.figure()
        plt.plot(x_axis, y_euler[:,0], '-', label= 'Suceptible')
        plt.plot(x_axis, y_euler[:,1], '-', label= 'Infected')
        plt.plot(x_axis, y_euler[:,2], '-', label= 'Recovered')
        plt.legend(loc= 'best')
        plt.title('Forward Euler (Beta: ' + str(beta) +
                  ') [' + model.__name__ + ']')
        
        plt.figure()
        plt.plot(x_axis, y_heun[:,0], '-', label= 'Suceptible')
        plt.plot(x_axis, y_heun[:,1], '-', label= 'Infected')
        plt.plot(x_axis, y_heun[:,2], '-', label= 'Recovered')
        plt.legend(loc= 'best')
        plt.title('RK2 Heun (Beta: ' + str(beta) +
                  ') [' + model.__name__ + ']')
    
    return y_euler



###
# Control Variables (h (step) and duration)
h = 1
duration = 250
compute_ode(initial_value= np.array([[s_init],[i_init], [r_init]]),
            h= h,
            duration= duration,
            model= d_dt,
            func_args= {
                'beta': beta,
                'gamma': gamma,
                'n': n
                },
            plot=True)

# # Change beta to high number
# # The system has become unstable. 
# # Stiffer: Huge difference between parameters
# # The solution is not usable.
# beta_2 = 53
# compute_ode(initial_value= np.array([[s_init],[i_init], [r_init]]),
#             h= 1,
#             duration= 250,
#             model= d_dt,
#             func_args= {
#                 'beta': beta_2,
#                 'gamma': gamma,
#                 'n': n
#                 },
#             plot=True)

# Use refined model
compute_ode(initial_value= np.array([[s_init],[i_init], [r_init]]),
            h= h,
            duration= duration,
            model= d_dt_refined,
            func_args= {
                'beta': beta,
                'gamma': gamma,
                'n': n,
                'v': v,
                'beta_drop' : beta_drop
                },
            plot=True)


##############################

# Realistic Scenario
# Unknown pandemic, hence beta and gamma is not known

def obj_fun(x):
    beta_comp = x[0]
    gamma_comp = x[1]
    infected_init = infected_numbers[0]
    
    dur = len(infected_numbers)
    
    y_euler = compute_ode(
                initial_value= np.array([[s_init],[infected_init],[r_init]]),
                h= step,
                duration= dur,
                model= d_dt,
                func_args= {
                    'beta': beta_comp,
                    'gamma': gamma_comp,
                    'n': n,
                    })

    y_euler = np.array(y_euler)
    infected_computed = y_euler[:,1]
    mse = 0
    N = len(infected_numbers)
    
    for i in range(N):
        mse += ((infected_computed[i*10] - infected_numbers[i])**2)
        
    return mse/N
    

def fit_ODE(data_filename):
    # obtain data
    infected_numbers = np.genfromtxt(data_filename)
    x_axis_data = np.arange(len(infected_numbers))
    data = np.stack((x_axis_data, infected_numbers), axis=1)
    infected_init = infected_numbers[0]
        
    # initial guess for simplex
    sp_init_guess = np.array([0.2, 0.1])
    
    # use simplex method to obtain beta & gamma
    res = minimize(fun= obj_fun, 
                   x0= sp_init_guess, 
                   method='nelder-mead')
    
    beta_fitted = res['x'][0]
    gamma_fitted = res['x'][1]
    
    # after computing beta and gamma, calculate the best fit curve
    sir_fitted = compute_ode(
                initial_value= np.array([[s_init],[infected_init],[r_init]]),
                h= 0.1,
                duration= duration,
                model= d_dt,
                func_args= {
                    'beta': beta_fitted,
                    'gamma': gamma_fitted,
                    'n': n,
                    })
    
    # extract only the infected numbers
    sir_fitted = np.array(sir_fitted)
    infected = sir_fitted[:,1]
    
    x_axis_fitted = np.arange(duration, step=step)
    
    # Plot (Data Collected)
    plt.figure()
    plt.plot(x_axis_data, infected_numbers, 'b.', label= 'Observed')
    plt.plot(x_axis_fitted, infected, '-', label= 'ODE Solution')
    
    plt.legend(loc= 'best')
    plt.ylabel('Number Infected')
    plt.xlabel('Time (Days)')
    
    plt.title('Fitted Curve (Beta: ' + str(beta_fitted) +')')
    return beta_fitted, gamma_fitted
###

step = 0.1
infected_numbers = np.genfromtxt('infected_cases.txt')
beta, gamma = fit_ODE(infected_numbers)



