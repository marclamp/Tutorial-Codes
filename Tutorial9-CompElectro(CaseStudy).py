# Tutorial 9 - Computational Electrophysiology

# Cardiac Models and Arrhythmia

# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from tt06_model import init_cond, tt_model, tt_model_stim, \
    tt_model_50block, tt_model_fullblock, tt_model_LQT3

####################################
def solve_and_plot(func,
                   duration,
                   initial_value,
                   method= 'LSODA'):
    
    # Use Scipy built-in fucntion
    result = solve_ivp(fun= func,               # returns the RHS of the eqn
                       t_span= [0, duration],   # 2-member sequence     
                       y0= initial_value,       # initial state
                       method= 'LSODA',         # has automatic stiffness detection
                       max_step = 0.1
                       )
    
    plt.plot(result.t, result.y[0,:], label= func.__name__)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    plt.title('Membrane Potential Over Time')
    plt.legend(loc='upper right')
    
    return result

####################################

### First Tasks
step = 0.001        # ms
time_period = 2000   # ms

initial_value = init_cond

# Solve ODE
basic_res = solve_and_plot(tt_model, time_period, initial_value)

# Include stimulus
stim_res = solve_and_plot(tt_model_stim, time_period, initial_value)

### Second Tasks - Simulate effects of Cisapride

# Compute and show the effects of Cisapride
half_res = solve_and_plot(tt_model_50block, time_period, initial_value)
full_res = solve_and_plot(tt_model_fullblock, time_period, initial_value)

# Third Tasks - Simultate the effets of LQT3 Gene Mutation
lqt_res = solve_and_plot(tt_model_LQT3, time_period, initial_value)