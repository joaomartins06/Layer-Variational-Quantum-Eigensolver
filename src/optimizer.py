import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

class COBYLA:
    ''' 
    Implement the COBYLA optimization algorithm
    This is one of the optimization algorithms used in the paper
    '''
    
    def __init__(self, maximize=False, max_iter=3000, rhobeg=1.0, verbose=True, record_loss=False):
        self.maximize = maximize
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
        self.best_so_far = -np.inf if self.maximize else np.inf
        loss_history = []

        with tqdm(total=self.max_iter, desc="COBYLA", disable=not self.verbose) as pbar:

            def wrapped(p):
                self.eval_count += 1
                val = cost_fn(p)
                loss_history.append(val)
                if (self.maximize and val > self.best_so_far) or (not self.maximize and val < self.best_so_far):
                    self.best_so_far = val
                pbar.update(1)
                pbar.set_postfix({'best_E': f'{self.best_so_far:+.4f}'})
                return -val if self.maximize else val

            result = minimize(
                fun=wrapped,
                x0=params,
                method='COBYLA',
                options={'maxiter': self.max_iter, 'rhobeg': self.rhobeg}
            )

        if self.verbose:
            print(f"Total evaluations: {self.eval_count}")

        final_val = -result.fun if self.maximize else result.fun
        return result.x, final_val, loss_history
    


class SMO:
    """
    Sequential Minimal Optimization
    Apparently this is a good optimizer for VQEs and is the other optimizer used in the paper
    """

    def __init__(self, maximize=False, max_iter=3000, verbose=True, record_loss=False):
        #same things as before
        self.maximize = maximize
        self.max_iter = max_iter
        self.verbose = verbose
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

        with tqdm(total=self.max_iter, desc="SMO", disable=not self.verbose) as pbar:

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

                # extrema location and value, computed analytically
                if self.maximize:
                    theta_star = theta_prev - np.arctan2(numerator, denominator)
                    L_ext = a3 + 0.5 * np.sqrt(numerator ** 2 + denominator ** 2)
                else:
                    theta_star = theta_prev + np.pi - np.arctan2(numerator, denominator)
                    L_ext = a3 - 0.5 * np.sqrt(numerator ** 2 + denominator ** 2)

                # update parameter
                params[k] = theta_star

                #update f0 to the "current" value
                f0 = L_ext

                if self.record_loss:
                    loss_history.append(f0)

                # track best
                if (self.maximize and f0 > best_energy) or (not self.maximize and f0 < best_energy):
                    best_energy = f0
                    best_params = params.copy()

                pbar.update(1)
                pbar.set_postfix({'best_E': f'{best_energy:+.4f}'})

        return best_params, best_energy, loss_history
    

