import numpy as np
from scipy._lib.pyprima.common import history
from scipy.optimize import minimize
from tqdm import tqdm

class COBYLA:
    ''' 
    Implement the COBYLA optimization algorithm
    This is one of the optimization algorithms used in the paper
    '''
    
    def __init__(self, max_iter=3000, rhobeg=1.0, verbose=True, record_loss=False):
        #3000 was the max iterations on the paper, so this will be our default
        self.max_iter = max_iter
        #trust region parameter for the COBYLA algo 
        self.rhobeg = rhobeg
        self.verbose = verbose
        self.record_loss = record_loss


    def optimise(self, params, cost_fn):
        #optimise uisng COBYLA
        #the cost function will be our hamiltonian

        self.eval_count = 0
        self.best_so_far = np.inf
        loss_history = []

        # COBYLA does roughly n_params + max_iter * 2 evaluations
        n_params = len(params)
        estimated_total = n_params + self.max_iter * 2

        if self.verbose:
            pbar = tqdm(total=estimated_total, desc="COBYLA")

        def wrapped(p):
            self.eval_count += 1
            val = cost_fn(p)
            loss_history.append(val)
            if val < self.best_so_far:
                self.best_so_far = val
            if self.verbose:
                pbar.update(1)
                pbar.set_postfix({'best_E': f'{self.best_so_far:+.4f}'})
            return val
        
        result = minimize(
            fun=wrapped,
            x0=params,
            method='COBYLA',
            options={'maxiter': self.max_iter, 'rhobeg': self.rhobeg}
        )

        if self.verbose:
            pbar.close()
            print(f"Total evaluations: {self.eval_count}")

        return result.x, result.fun, loss_history
    


class SMO:
    """
    Sequential Minimal Optimization
    Apparently this is a good optimiser for VQEs and is the other optimiser used in the paper 
    """

    def __init__(self, max_iter=3000, record_loss=False):
        #same things as before
        self.max_iter = max_iter
        self.record_loss = record_loss

    def optimise(self, params, cost_fn):
        params = params.copy()
        n_params = len(params)
        best_energy = cost_fn(params)
        best_params = params.copy()
        loss_history = []

        #start f0, though this will be updated each iteration
        f0 = best_energy
        if self.record_loss:
            loss_history = [f0]

        for iteration in range(self.max_iter):
            # choose which parameter to update (sequential)
            k = iteration % n_params
            theta_prev = params[k]

            # evaluate at theta_prev + pi/2 and theta_prev - pi/2
            # f0 at theta_prev is reused from previous iteration
            params[k] = theta_prev + np.pi / 2
            f_plus = cost_fn(params)

            params[k] = theta_prev - np.pi / 2
            f_minus = cost_fn(params)

            # solve for the sinusoid constants
            a3 = (f_plus + f_minus) / 2
            numerator = f_minus - f_plus
            denominator = 2 * f0 - f_plus - f_minus

            # minimum location and value, computed analytically
            theta_star = theta_prev + np.pi - np.arctan2(numerator, denominator)
            L_min = a3 - 0.5 * np.sqrt(numerator**2 + denominator**2)

            # update parameter
            params[k] = theta_star

            #update f0 to the "current" value
            f0 = L_min

            if self.record_loss:
                loss_history.append(f0)

            # track best
            if f0 < best_energy:
                best_energy = f0
                best_params = params.copy()

        return best_params, best_energy, loss_history
    

