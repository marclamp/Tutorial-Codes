# Tutorial 4 - Steepest Descent Method (Multiple Variable Optimization)

# Pharmacokinetics Case Study

import numpy as np
import matplotlib.pyplot as plt

# Declare known variables
bw = 84             # kg
v = 20 * bw * 1000  # litres/kg body weight (l -> ml)
f = 0.6             # fraction of ingested dosage
d = 400 * 1e6       # mg -> ng dosage
tolerance = 1e-6    # threshold    
max_iter = 40000    # max number of iterations

# Function definitions

# provides e^(-kt)
def e_k_t(t, k):
    return np.exp(-k * t)

# computes e^(-ke*t) - e^(-ka*t)
def eket_sub_ekat(t, ka, ke):
    return (e_k_t(t, ke) - e_k_t(t, ka))
    
# function for the oral equation
def func_C(t, ka, ke):
    return (f*d/v) * (ka/(ka-ke)) * \
        (eket_sub_ekat(t, ka, ke))

# functions to compute the derivatives of ka and ke
def first_der_ka(t, ka, ke):
    return (f*d/v) * (((ka * t)/(ka-ke)) * e_k_t(t, ka) + \
        eket_sub_ekat(t, ka, ke) * \
        ((1/(ka-ke)) - ka/(ka-ke)**2))

def first_der_ke(t, ka, ke):
    return (f*d/v) * (((-ka * t)/(ka-ke)) * e_k_t(t, ke) + \
        eket_sub_ekat(t, ka, ke) * \
        (ka/(ka-ke)**2))

# function for finding gradient of sse
# to be minimised
def mse_grad(ka, ke):
    m = len(time)
    mse_ka = 0
    mse_ke = 0
    
    for i in range(0, m):
        mse_ka += (plasma_conc[i] - func_C(time[i], ka, ke)) * \
            first_der_ka(time[i], ka, ke)
        
        mse_ke += (plasma_conc[i] - func_C(time[i], ka, ke)) * \
            first_der_ke(time[i], ka, ke)
    
    mse_ka = -2/m * mse_ka
    mse_ke = -2/m * mse_ke
    
    grad = np.array([mse_ka, mse_ke])
    magnitude = np.sqrt((grad[0]**2 + grad[1]**2))
   
    return (grad, magnitude)

# function for mean sum of squared errors
def mse(k):
    ka = k[0]
    ke = k[1]
    m = len(time)
    mse = 0
    for i in range(0, m):
        mse += (plasma_conc[i] - func_C(time[i], ka, ke))**2
    
    return 1/m * mse

# function for computing alpha
def fx_plus_alpha_d(a, k, gradient, magnitude):
    return mse(k - a * (gradient / magnitude))

# golden sections method
# adapted to this case, finding minimum
def golden_section(a, b, func, k, grad, mag):   
    
    max_iter = 10000    
    epsilon = 1e-9
    
    # useful number for golden sections
    r = (np.sqrt(5)-1)/2
    
    # two internal points
    x1 = (1-r) * (b-a) + a
    x2 = r * (b-a) + a
    
    # function evaluations
    f_x1 = func(x1, k, grad, mag)   # +ve because we searching for min
    f_x2 = func(x2, k, grad, mag)
    
    for i in range(0, max_iter):
        # choose which part of the bracket to discard
        # A------x1----x2------B
               
        # if f(x1) > f(x2), discard [A x1]
        # A------x1----x2------B
        #         A----x1--x2--B
        if f_x1 >= f_x2:
            a = x1
            x1 = x2
            x2 = r * (b-a) + a              # new point
            f_x1 = f_x2                     # reuse, no need to recompute
            f_x2 = func(x2, k, grad, mag)   # use +ve, as finding min
    
        # if f(x2) > f(x1), discard [x2 B]
        # A------x1----x2------B
        # A--x1--x2----B
        elif f_x2 > f_x1:
            b = x2
            x2 = x1
            x1 = (1-r) * (b-a) + a
            f_x2 = f_x1
            f_x1 = func(x1, k, grad, mag)
        
        # exit golden sections if below epsilon value
        if np.abs(b-a) < epsilon:
           break
    
    minimum = (a+b)/2
    return minimum

def steepest_descent(data,          # data to fit the curve to
                     sd_init_guess, # initial guess for steepest descent algo
                     gs_init_guess):# initial guess for golden sections
    
    # parameter to be optimized
    k = np.array(sd_init_guess)

    
    for i in range(0, max_iter):   
        
        # compute the gradient and magnitude
        grad, mag = mse_grad(k[0], k[1])
        
        # termination criteria
        if mag <= tolerance:
            break
        
        # perform line search
        a = golden_section(gs_init_guess[0], gs_init_guess[1], \
                           fx_plus_alpha_d, k, grad, mag)
        
        # new point, grad/mag = direction
        k = k - a * (grad/mag)
        
        # Tracking of ka and ke
        ka_plot.append(k[0])
        ke_plot.append(k[1])
    
    print(k, ' in ', i+1, ' iterations')  
    return k
    
    
# Solve
# extract data from trial participant
data = np.genfromtxt('single_patient_data.txt', skip_header=1)

# extract x and y axis
time = data[:,0]
plasma_conc = data[:,1]

# choose initial guess
sd_init_guess = [3.0, 5.0]
gs_init_guess = [0, 3]

# Tracking of ka and ke for plotting purposes
ka_plot = [sd_init_guess[0]]
ke_plot = [sd_init_guess[1]]

k = steepest_descent(data, sd_init_guess, gs_init_guess)

# Plotting parameters
x = np.arange(np.min(time), np.max(time), 0.1)
curve = func_C(x, k[0], k[1])

# Plot
plt.title('Drug Concentration (ng/ml) against Time (hr)')
plt.xlabel('Time (hr)')
plt.ylabel('Concentration (ng/ml)')
plt.plot(time, plasma_conc, 'b.')
plt.plot(x, curve, 'r-')

plt.figure()
plt.plot(ka_plot, ke_plot, 'b.-')
plt.xlabel('ka')
plt.ylabel('ke')