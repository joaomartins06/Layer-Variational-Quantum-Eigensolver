import numpy as np
from .ansatze import Ansatz


class LayerVQE:
    ''' 
    Now we join everything.
    '''

    def __init__(self, problem, simulator, optimizer_class, n_layers=2, k_per_layer=200, k_final=3000,
                 use_sampling=False, n_samples=2000, record_loss=False):
        
        #define the problem
        #in the paper, it is k communities
        self.problem = problem

        #define the simulator, how we evaluate the hamiltonian
        self.simulator = simulator
        
        #choose the simulator class
        #in the paper is COBYLA or SMO
        #we can extend to other classes if we want to
        self.optimizer_class = optimizer_class

        #define the number of layers, iterations per layer and final iterations
        #the default valus were taken from the paper
        self.n_layers = n_layers
        self.k_per_layer = k_per_layer
        self.k_final = k_final
        #choose if we compute things analytically or by sampling
        self.use_sampling = use_sampling
        self.n_samples = n_samples

        self.record_loss = record_loss

        #track the results
        self.history = {
            'layer': [],
            'energy': [],
            'approx_ratio': [],
            'optimizer_loss': []
            }


    def _cost_fn(self, ansatz):
        #defining the cost function depending on the simulator
        if self.use_sampling:
            return lambda params: self.simulator.sample_expectation(params, ansatz, self.problem, 
                                                                    self.n_samples)
        else:
            return lambda params: self.simulator.expectation(params, ansatz, self.problem)


    def _record(self, layer, energy, loss_history):
        #record results
        ratio = self.problem.get_approximation_ratio(energy)
        self.history['layer'].append(layer)
        self.history['energy'].append(energy)
        self.history['approx_ratio'].append(ratio)
        self.history['optimizer_loss'].append(loss_history)
        print(f"layer {layer}: energy={energy:+.4f}, approx_ratio={ratio:+.4f}")


    def run(self):
        #execute the algo
        print(f"Starting L-VQE: {self.n_layers} layers, "
              f"{self.k_per_layer} iter/layer, {self.k_final} final iter")
        print(f"Mode: {'finite sampling' if self.use_sampling else 'exact expectation'}")

        #layer 0
        ansatz = Ansatz(self.problem.num_qubits)
        cost_fn = self._cost_fn(ansatz)

        print(f"\nLayer 0: \n")
        optimizer = self.optimizer_class(max_iter=self.k_per_layer, record_loss=self.record_loss)
        best_params, best_energy, loss_history = optimizer.optimise(ansatz.params.copy(), cost_fn)
        ansatz.params = best_params
        self._record(0, best_energy, loss_history)

        #layer l > 0
        #the range is from 1 to n_layers + 1 because we have a set of iterations for each layer and
        #a final set of iterations in the end /just following the suggested algorithm)
        for l in range(1, self.n_layers + 1):
            #add a layer
            ansatz.add_layer()
            #update the cost function
            cost_fn = self._cost_fn(ansatz)

            #use intermediate iterations for all but the last layer
            if l < self.n_layers:
                n_iter = self.k_per_layer
                print(f"\nLayer {l} — {n_iter} iterations (before convergence)")

            #final set of iterations
            else:
                n_iter = self.k_final
                print(f"\nFinal layer — {n_iter} iterations (final)")

            #optimize this layer accordingly
            optimizer = self.optimizer_class(max_iter=n_iter, record_loss=self.record_loss)
            best_params, best_energy, loss_history = optimizer.optimise(ansatz.params.copy(), cost_fn)

            ansatz.params = best_params
            self._record(l, best_energy, loss_history)

        return {'final_energy': best_energy,
                'final_approx_ratio': self.problem.get_approximation_ratio(best_energy),
                'final_params': best_params,
                'final_ansatz': ansatz,
                'history': self.history}

