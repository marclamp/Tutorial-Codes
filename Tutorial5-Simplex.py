# Tutorial 5 - Simplex (Neldor-Mead) Method (Multiple Variable Optimization)

# Pharmacokinetics Case Study

import numpy as np
import matplotlib.pyplot as plt

# Declare known variables
bw = 84             # kg
v = 20 * bw * 1000  # litres/kg body weight (l -> ml)
f = 0.6             # fraction of ingested dosage
d = 400 * 1e6       # mg -> ng dosage
tolerance = 1e-9    # threshold
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
        
# function for mean sum of squared errors
# to be minimized.
# pass in 1 parameter (array), if there
# are multiple variables to be minimized
def mse(k):
    ka = k[0]
    ke = k[1]
    m = len(time)
    mse = 0
    for i in range(0, m):
        mse += (plasma_conc[i] - func_C(time[i], ka, ke))**2
    
    return 1/m * mse

# helper function to evaluate points
def eval_points(points):
    new_arr = []
    for p in points:
        err = mse(p)
        new_arr.append([err, p])
    return new_arr

def is_converged(points):
    # tolerances
    epsilon_g = 1e-5
    epsilon_c = 1e-5

    # allocate the points
    g = points[0][1]
    w = points[2][1]
    mse_g = points[0][0]
    mse_w = points[2][0]
    
    # geometric convergence
    if np.linalg.norm(g - w) < epsilon_g:
        print('Converged, geometric criterion')
        return True
    
    # cost convergence
    if np.abs(mse_g - mse_w) < epsilon_c:
        print('Converged, cost criterion')
        return True
    
    return False

# Solve

# extract data from trial participant
data = np.genfromtxt('single_patient_data.txt', skip_header=1)

time = data[:,0]
plasma_conc = data[:,1]

# choose initial guess
init_guess = np.array([3, 5])
sp_init_guess = [init_guess,    # points are (ka, ke)
                 np.array([init_guess[0]*1.05, init_guess[1]]), 
                 np.array([init_guess[0], init_guess[1]*1.05])]   

points = eval_points(sp_init_guess)
points.sort()

ka_plot = []
ke_plot = []

simplex_list_track = []

# Neldor-Mead Logic
for i in range(max_iter):
       
    # allocate the points (GAW)
    g = points[0][1]
    a = points[1][1]
    w = points[2][1]
    mse_g = points[0][0]
    mse_a = points[1][0]
    mse_w = points[2][0]
    
    # Plotting purposes
    # plt.plot([g[0], a[0], w[0], g[0]], [g[1], a[1], w[1], g[1]], '.-')
    ka_plot.append(g[0])
    ke_plot.append(g[1])
    
    simplex_list_track.append(np.round([[g[0], g[1]], [a[0], a[1]], [w[0], w[1]]], 3))
    
    # convergence criteria
    if is_converged(points):
        break
    
    # find the middle (M), and its mse
    m = (g + a) / 2
    mse_m = mse(m)
    
    # find the reflection (R), and its mse
    r = 2 * m - w
    mse_r = mse(r)
    
    # Simplex Decision Tree
    # if f(R) <= f(G), try extension
    if mse_r <= mse_g:
        t = 2 * r - m
        mse_t = mse(t)
        
        # if f(T) <= f(R), extend
        # new simplex is TGA
        if mse_t <= mse_r:
            points = [[mse_t, t],
                      [mse_g, g],
                      [mse_a, a]]
            print("Extension")
        # otherwise, new simplex is RGA
        else:
            points = [[mse_r, r],
                      [mse_g, g],
                      [mse_a, a]]
            print("Reflection (1)")
    # otherwise, if f(R) <= f(A)
    # new simplex is RGA
    elif mse_r <= mse_a:
        points = [[mse_r, r],
                  [mse_g, g],
                  [mse_a, a]]
        print("Reflection (2)")
    
    # otherwise, check for contraction
    else:
        # compute contraction points and their mse
        c_out = (m + r) / 2
        c_in = (m + w) / 2
        mse_cout = mse(c_out)
        mse_cin = mse(c_in)
        contraction_points = [[mse_cout, c_out],
                              [mse_cin, c_in]]
        contraction_points.sort()
        
        mse_cp = contraction_points[0][0]
        cp = contraction_points[0][1]
        
        # if f(CP) < f(A), contract
        # new simplex is CGA
        if mse_cp < mse_a:
            points = [[mse_cp, cp],
                      [mse_g, g],
                      [mse_a, a]]
            print("Contraction")
        # otherwise, perform shrink operation
        # new simplex is G,S1,S2
        else:
            # compute new points
            s1 = (g + w) / 2
            s2 = (g + a) / 2
            
            points = [[mse_g, g],
                      [mse(s1), s1],
                      [mse(s2), s2]]
            print("Shrink")
            
k = points[0][1]
print(k, ' in ', i+1, ' iterations')  
x = np.arange(np.min(time), np.max(time), 0.1)
curve = func_C(x, k[0], k[1])

# Plot
plt.figure()
plt.title('Drug Concentration (ng/ml) against Time (hr)')
plt.xlabel('Time (hr)')
plt.ylabel('Concentration (ng/ml)')
plt.plot(time, plasma_conc, 'b.')
plt.plot(x, curve, 'r-')

# Plot Simplex Progression
plt.figure()
plt.xlabel('ka')
plt.ylabel('ke')
plt.plot(ka_plot, ke_plot, '.-')
        
        
        
        
        
        
    

    
    
    
    
    
    
    
    