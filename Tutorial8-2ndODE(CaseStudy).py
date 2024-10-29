# Tutorial 8 - Second Order ODEs
# IVP (Initial Value Problem)
# BVP (Boundary Value Problem)
# Gist: Conversion to Set of ODEs

# 3-Point Bending in Sports

# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Define Known Constants
E = 18e5    # GPa (Conversion to (N/m2) then to (N/cm2))
T = 833     # N
I = 2.7     # cm^4
F = 8500    # N
L = 40      # cm
a = 10      # cm

# Define helper functions and variables
mu = F / (E*I*L)
alpha = T / (E*I)

# Boundary Values
y_0 = 0
y_L = 0

####################################
# Reused code from previous tutorial
# only change is the inclusion of func_args

def forward_euler(h, x, y, model, func_args : dict = None):
    """ Forward Euler function 
    (or First Order Truncation)
       ...

        Parameters
        ---------- 
        h : int
            step size (typically: h, dx, dt, ...)
        
        x : float64
            the value of x at i
            
        y : float64 or np.array
            can be y value or ODE values
            
        model : func
            model used to compute the derivative or ODE solutions
            
        func_args : dict
            contains any additional parameters needed 
            (to be passed into the model for computation)
    """
    
    dy = model(x, y, func_args)
    return y + h * dy

def backward_euler(h, x, y, model, func_args=None):
    """ Backward Euler function
    
        Parameters
        ----------
        h : int
            step size (typically: h, dx, dt, ...)
        
        x : float64
            the value of x at i+1
            
        y : np.array
            matrix ODE of y values at i
            
        model : func
            model used to compute the ODE solutions
            model dictates the LHS and RHS
            
        func_args : dict, optional
            contains any additional parameters needed 
            (to be passed into the model for computation
        
    """
    
    return model(h, x, y, func_args)

####################################

def defl_eqn(x):
    return np.where(x > a, 
                    mu * a * (L - x),   # if ture
                    mu * x * (L - a))

def d_dx(x, y, func_args=None):
    """ BVP, where the boundary is y(0) and y(L).
        \n Converting it to IVP, y(0) = y(x0), \
            dy/dx(x0) = is passed in (guessed & iterated)

    \n Equations in the Set of ODE (converted to IVP)
    \n |     y1(i)   |
    \n | y2(i) + eqn |
    \n
    \n where eqn =  | mu * a * (L - x)  if x > a
    \n             \| mu * x * (L - a)  otherwise

    Parameters
    ----------
    x : float64
        the value of x at i.
        
    y : np.array
        array of ODE y values.
        
    func_args : TYPE, optional
        contains any additional parameters needed 
        (to be passed into the model for computation. 
        The default is None.

    Returns
    -------
    np.array
        Array of ODE solutions at i+1

    """
    y1 = y[0]
    y2 = y[1]
        
    return np.array([         y2, 
                     -alpha*y1 + defl_eqn(x)])

# both equations in the Set of ODE (for backward euler)
# x in this case is the i+1
# y is the matrix [y1, y2, ...]
def backward_model(dx, x_next, y, func_args=None):
    # Backward Euler Matrix
    # |1    -dx| \/ |y1(i+1)| _ |     y1(i)   |
    # |α*dx   1| /\ |y2(i+1)| ‾ | y2(i) + eqn |
    #
    # where eqn _ | dx * mu * a * (L - x(i+1))  if x(i+1) > a
    #           ‾ | dx * mu * x(i+1) * (L - a)  otherwise
    
    lhs = np.array([[1,      -dx],
                    [alpha*dx, 1]])
              
    rhs = np.zeros(2)
    rhs[0] = y[0]                       # y1(i)
    rhs[1] = y[1] + dx*defl_eqn(x_next) # y2(i) + eqn
    
    y_sol = np.linalg.solve(lhs, rhs)   # returns column array
    
    return np.transpose(y_sol)
    
# Function that solves 2nd order ODE as an IVP,
# (Y(0) is known and Y'(0) is passed in)
def solve_IVP(x, y_prime, dx,
              model=None,
              method=None,       # can use forward_euler, rk2, rk4  
              func_args=None):  
    
    y = np.zeros((len(x), 2))   # col 1 -> y1, col 2 -> y2
    
    # initial condition
    y[0, 0] = y_0               # y1 = y
    y[0, 1] = y_prime           # y2 = y'
    
    for i in range(len(x)-1):
        
        if method == backward_euler:
            y_next = method(h= dx, 
                            x= x[i+1], 
                            y= y[i], 
                            model= model)
            
        else:
            y_next = method(h= dx, 
                            x= x[i], 
                            y= y[i], 
                            model= model)
         
        y[i+1] = y_next
    
    return y[:, 0], y[0,1], y[-1,0] # returns y1 -> y, beta value, and b value

# computing the next "shoot" 
# β3 = β2 + (B - B2) / slope
# slope = (B2 − B1) / (β2 − β1)
def compute_next_shoot(b, b1, b2, beta1, beta2):
    slope = (b2 - b1) / (beta2 - beta1)
    beta3 = beta2 + (b - b2) / slope
    return beta3

def shooting_method(initial_guess,  # consists of [first, second] guess
                    L,              # total Length (or duration)
                    dx= 0.1,        # step size
                    model= None,    # model for ODE eqns
                    method= None):  # method used to solve (FE, BE, RK2, RK4)
    
    x = np.arange(0, L+dx, dx)
    
    # "shoot" using initial guesses
    y_first, beta1, b1 = solve_IVP(x= x,
                                   y_prime= initial_guess[0],
                                   dx= dx,
                                   model= model,
                                   method= method)

    y_second, beta2, b2 = solve_IVP(x= x,
                                    y_prime= initial_guess[1],
                                    dx= dx,
                                    model= model,
                                    method= method)
   
    # helper variables
    b = y_L
    beta_i_prev = beta1
    beta_i =      beta2
    b_i_prev =    b1
    b_i =         b2
    
    # begin continuous shooting
    while (np.abs(beta_i - y_L) > 1e-6):
        
        # compute new shoot
        beta_new = compute_next_shoot(b, b_i_prev, b_i, beta_i_prev, beta_i)
        
        y_sol, b_sol, beta_sol = solve_IVP(x= x,
                                           y_prime= beta_new,
                                           dx= dx,
                                           model= model,
                                           method= method)
        
        # update i and i+1 variables
        b_i_prev = b_i
        b_i = b_sol
        beta_i_prev = beta_i
        beta_i = beta_sol
    
    plt.figure()
    plt.plot(x, y_sol)
    plt.title('Deflection of Tibia, Method: [' + method.__name__ +']')
    plt.xlabel('Tibia Length (cm)')
    plt.ylabel('Deflection (cm)')
    # plt.axis('scaled')  # to see a realistic deformation
    return
    
def equilibrium_method(h,           # step, dx in this case
                       P,           # coefficient to dy/dx, 0 in this case
                       Q,           # coefficient to y, α in this case
                       F,           # F(x), defl_eqn in this case
                       lb=0,        # lower bound, y_0 in this case
                       ub=0,        # upper bound, y_L in this case
                       ):
    
    # Generate x-axis for discretization
    dx = h
    x = np.arange(0, L+dx, dx)
    
    # Number of grid points
    n = len(x)
        
    # Create tbe finite difference matrix (tridiagonal)
    main_diag = np.ones(n-2) * (-2 + h**2 * Q)
    upp_diag = np.ones(n-3) * (1 - h/2 * P)
    low_diag = np.ones(n-3) * (1 + h/2 * P)
    
    # Construct the tridiagonal matrix (A matrix)
    a_matrix = np.diag(main_diag, 0) + \
               np.diag(upp_diag, 1) + \
               np.diag(low_diag, -1)

    # Right-hand side vector (F) with deflection equation values
    rhs = h**2 * F(x[0:-2])
    
    # Apply Boundary Conditions
    rhs[0] -= lb * (1 - h/2 * P)
    rhs[-1] -= ub * (1 - h/2 * P)
    
    # Solve the linear system
    y_interior = np.linalg.solve(a_matrix, rhs)
    
    # Full solution vector (including boundaries)
    y_sol = np.zeros(n)         # Initialize with zeros
    y_sol[1:-1] = y_interior    # Insert the interior solution
    y_sol[0] = y_0              # Insert y(0), which is 0 in this case
    y_sol[-1] = y_L             # Insert y(L), which is 0 in this case
    
    plt.figure()
    plt.plot(x, y_sol)
    plt.title('Deflection of Tibia, Method: [equilibrium_method]')
    plt.xlabel('Tibia Length (cm)')
    plt.ylabel('Deflection (cm)')
    # plt.axis('scaled')  # to see a realistic deformation
    return y_sol
    
####################################

initial_guess = [7.5, 12.5]
shooting_method(initial_guess, L, 
                model= d_dx, 
                method= forward_euler)

shooting_method(initial_guess, L, 
                model= backward_model, 
                method= backward_euler)

equilibrium_method(h= 0.1, 
                   P= 0, 
                   Q= alpha, 
                   F= defl_eqn)

''' Additional Notes
# Verify the formula for deflection (δ) to calculate E
# only when the load is in the middle of the beam
# and when there is no lateral load (T)
# print(-F*L**3/(48*E*I))    # valid only with a=20, T=0

# Changing the step, h, to 1, 
# and E to soft stiffness, 18e1
# results in unstable solution for forward euler
# but a stable, albeit inaccurate solution for backward euler

E = 18e1
mu = F / (E*I*L)
alpha = T / (E*I)

shooting_method(initial_guess, L, 
                dx= 1,
                model= d_dx, 
                method= forward_euler)

shooting_method(initial_guess, L, 
                dx= 1,
                model= backward_model, 
                method= backward_euler)

equilibrium_method(h= 1, 
                   P= 0, 
                   Q= alpha, 
                   F= defl_eqn)
'''



