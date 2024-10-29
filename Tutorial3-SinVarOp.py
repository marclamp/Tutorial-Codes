# Tutorial 3 - Single Variable Optimization

# Biopharma Reactors Case Study

import numpy as np
import matplotlib.pyplot as plt

# Declare known parameters
price = 9300        # $/kg
cost_b = 250        # $/kg
cost_a = 245        # $/kg
hours_op = 8000     # hours
vol = 120e3         # l (litres)
a_in = 8e-6         # mol/l 
k = 0.02            # 1/hr
molweight_b = 0.02  # kg/mol
molweight_a = 0.05  # kg/mol

# Helper constant variables
d1 = (price-cost_b)*hours_op*molweight_b*vol*k*a_in
d2 = a_in*hours_op*molweight_a

# Profit as a function of Q
def q_func(q):
    f_Q = (d1/(1 + (k*vol/q))) - cost_a*(q*d2)**0.6
    return f_Q

# Newton Method

# compute the first and second derivatives
def first_der(q):
    fd_Q = ((d1*k*vol) / (q**2*(1+k*vol/q)**2)) - \
           (0.6*cost_a*d2**0.6) / (q**0.4)
    return fd_Q

def second_der(q):
    sd_Q = d1*k*vol*(-2/(q**3*(1+k*vol/q)**2)) + \
        (2*k*vol / (q**4*(1+k*vol/q)**3)) + \
        (0.6*cost_a*(d2**0.6)*(0.4/q**1.4))
    return sd_Q

def newton_method(initial_guess):
    x_0 = initial_guess # initial guess
    max_iter = 1000 # max number of iterations
    epsilon = 1e-9  # precision
    x_prev = x_0
    
    for i in range(0, max_iter):
        fd_x = first_der(x_prev)
        sd_x = second_der(x_prev)
        x_next = x_prev - (fd_x/sd_x) 
        
        if (np.abs(fd_x) < epsilon):
            break
            
        x_prev = x_next     # update i with i+1
    return x_next

# Golden Sections method
def golden_section(a, b, func, maximum=False):
    max_iter = 1000 # max number of iterations
    epsilon = 1e-9  # precision
       
    # useful number for golden sections
    r = (np.sqrt(5)-1)/2
    
    # two internal points
    x1 = (1-r) * (b-a) + a
    x2 = r * (b-a) + a
    
    # function evaluations
    f_x1 = func(x1)  
    f_x2 = func(x2)
    
    if maximum:
        f_x1 = - f_x1   # -ve because we searching for max
        f_x2 = - f_x2
    
    for i in range(0, max_iter):
        # choose which part of the bracket to discard
        # A------x1----x2------B
        
        if np.abs(b-a) < epsilon:
            break
        
        # if f(x2) > f(x1), discard [x2 B]
        # A------x1----x2------B
        # A--x1--x2----B
        if f_x2 > f_x1:
            b = x2
            x2 = x1
            x1 = (1-r) * (b-a) + a
            f_x2 = f_x1
            f_x1 = func(x1)
            
            if maximum:
                f_x1 = - f_x1   # -ve because we searching for max
        
        # if f(x1) > f(x2), discard [A x1]
        # A------x1----x2------B
        #         A----x1--x2--B
        elif f_x1 > f_x2:
            a = x1
            x1 = x2
            x2 = r * (b-a) + a  # new point
            f_x1 = f_x2         # reuse, no need to recompute
            f_x2 = func(x2)   
            if maximum:
                f_x2 = - f_x2   # use -ve, as finding max
    
    result = (a+b)/2
    return result

# Plot
x_axis = np.arange(0.1, 90e3, 1) # Q
profit = q_func(x_axis)
plt.figure()
plt.plot(x_axis, profit, 'r-')
plt.xlabel('Flow Rate (l/hr)')
plt.ylabel('Annual Profit ($)')
# plt.plot(x_axis, first_der(x_axis), 'b-')

# Part 4
# 3 plots, +30% price, price, -30% price
diff_price = [price*0.7, price, price*1.3]
x_axis = np.arange(0.1, 90e3, 1) # Q
 
plt.figure()
plt.xlabel('Flow Rate (l/hr)')
plt.ylabel('Annual Profit ($)')

for p in diff_price:
    # update helper constant variables
    d1 = (p-cost_b)*hours_op*molweight_b*vol*k*a_in
    
    print('For price at: ' + str(p))
    # initial guess
    x_0 = 10e3
    result_newton = newton_method(x_0)
    
    print('Newton Method')
    print("Optimal Flow Rate:", result_newton)
    print('Profits at optimal flow rate:', q_func(result_newton))
    print("Second derivative:", second_der(result_newton))
    print()

    # initial bracket
    a = 15e3
    b = 50e3
    result_gs = golden_section(a, b, func=q_func, maximum=True)
    
    print('Golden Sections Method')
    print("Optimal Flow Rate:", result_gs)
    print('Profits at optimal flow rate:', q_func(result_gs))
    print()
    
    profit = q_func(x_axis)
    plt.plot(x_axis, profit, label='Market Price ' + str(int(p)) + ' $/kg')

plt.legend(loc='lower right')

