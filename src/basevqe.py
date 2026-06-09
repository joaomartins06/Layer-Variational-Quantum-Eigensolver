import numpy as np
from .ansatze import Ansatz


class BaseVQE:
    '''
    Fixed-ansatz VQE baseline used in Section VI-B of Liu et al. (2022).

    Unlike L-VQE, the ansatz is built upfront to its full depth (all n_layers added
    at once) and all parameters are optimized in a single run with the full budget
    (n_layers * k_per_layer + k_final iterations). Layer-0 parameters are randomly
    initialized; deeper-layer parameters start at zero.

    With finite sampling, performance degrades as depth increases because the
    optimizer must navigate a large, poorly initialized landscape from scratch.
    L-VQE avoids this by growing the circuit gradually with warm-started parameters.
    '''

    def __init__(self, problem, simulator, optimizer_class, seed=None, n_layers=2, k_per_layer=200, k_final=3000,
                 use_sampling=False, n_samples=2000, record_loss=False):

        # define the problem
        # in the paper, it is k communities
        self.problem = problem

        # define the simulator, how we evaluate the hamiltonian
        self.simulator = simulator

        # choose the optimizer class
        # in the paper is COBYLA or SMO
        # we can extend to other classes if we want to
        self.optimizer_class = optimizer_class

        self.seed = seed

        # define the number of layers, iterations per layer and final iterations
        # the default valus were taken from the paper
        self.n_layers = n_layers
        self.k_per_layer = k_per_layer
        self.k_final = k_final
        # choose if we compute things analytically or by sampling
        self.use_sampling = use_sampling
        self.n_samples = n_samples

        self.record_loss = record_loss

        # track the results
        self.history = {
            'layer': [],
            'energy': [],
            'approx_ratio': [],
            'optimizer_loss': []
        }

    def _cost_fn(self, ansatz):
        # defining the cost function depending on the simulator
        if self.use_sampling:
            return lambda params: self.simulator.sample_expectation(params, ansatz, self.problem,
                                                                    self.n_samples)
        else:
            return lambda params: self.simulator.expectation(params, ansatz, self.problem)

    def _record(self, layer, energy, loss_history):
        # record results
        ratio = self.problem.get_approximation_ratio(energy)
        self.history['layer'].append(layer)
        self.history['energy'].append(energy)
        self.history['approx_ratio'].append(ratio)
        self.history['optimizer_loss'].append(loss_history)
        print(f"layer {layer}: energy={energy:+.4f}, approx_ratio={ratio:+.4f}")

    def run(self):
        # execute the algo
        print(f"Starting Base-VQE: {self.n_layers} layers, "
              f"{self.k_per_layer} iter/layer, {self.k_final} final iter")
        print(f"Mode: {'finite sampling' if self.use_sampling else 'exact expectation'}")

        ansatz = Ansatz(self.problem.num_qubits, seed=self.seed)

        for _ in range(1, self.n_layers + 1):
            # add a layer
            ansatz.add_layer()

        cost_fn = self._cost_fn(ansatz)
        optimizer = self.optimizer_class(
            maximize=self.problem.maximize,
            max_iter=(self.n_layers * self.k_per_layer) + self.k_final,
            record_loss=self.record_loss
        )
        best_params, best_energy, loss_history = optimizer.optimise(ansatz.params.copy(), cost_fn)
        ansatz.params = best_params
        self._record(self.n_layers, best_energy, loss_history)

        return {'final_energy': best_energy,
                'final_approx_ratio': self.problem.get_approximation_ratio(best_energy),
                'final_params': best_params,
                'final_ansatz': ansatz,
                'history': self.history}

