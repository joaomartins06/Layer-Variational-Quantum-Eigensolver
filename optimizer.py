import numpy as np
from scipy.optimize import minimize


class COBYLA:
    ''' 
    Implement the COBYLA optimization algorithm
    This is one of the optimisation algorithms used in the paper
    '''
    def __init__(self, max_iter=3000, rhobeg=1.0):
        #3000 was the max iterations on the paper, so this will be our default
        self.max_iter = max_iter
        #trust region parameter for the COBYLA algo 
        self.rhobeg = rhobeg

    def optimise(self, params, cost_fn):
        #optimise uisng COBYLA
        #the cost function will be our hamiltonian
        result = minimize(
            fun=lambda p: cost_fn(p),
            x0=params,
            method='COBYLA',
            options={'maxiter': self.max_iter, 'rhobeg': self.rhobeg}
        )

        return result.x, result.fun
    


class SMO:
    """
    Sequential Minimal Optimization
    Apparently this is a good optimiser for VQEs and is the other optimiser used in the paper 
    """

    def __init__(self, max_iter=3000):
        #same things as before
        self.max_iter = max_iter

    def optimise(self, params, cost_fn):
        params = params.copy()
        n_params = len(params)
        best_energy = cost_fn(params)
        best_params = params.copy()

        for iteration in range(self.max_iter):
            #"choose" which parameter to update 
            k = iteration % n_params

            #store current angle
            theta = params[k]

            #evaluate at three points to fit the sinusoid
            #had to chcek how SMO works
            #we change a certain parameter by 0, pi/2 and pi 
            #and evaluate the cost function at these three points
            #1st eval
            params[k] = theta
            f0 = cost_fn(params)
            #2nd eval
            params[k] = theta + np.pi / 2
            f1 = cost_fn(params)
            #3rd eval
            params[k] = theta + np.pi
            f2 = cost_fn(params)
            #get the necessary quantities
            c = (f0 + f2) / 2
            a = f0 - c
            b = f1 - c
            #and compute the exact minimum of the sinusoid
            theta_min = np.arctan2(b, a) + np.pi

            #update the parameter to the optimal value
            params[k] = theta_min

            #evaluate the cost function at the new parameter value
            current_energy = c - np.sqrt(a**2 + b**2)
            if current_energy < best_energy:
                best_energy = current_energy
                best_params = params.copy()

        return best_params, best_energy