import numpy as np
import networkx as nx
from itertools import product as iterproduct
from .problem import Problem, HamiltonianType
from typing import Optional


class MaxCut(Problem):
    """
     Implementation of the Max-Cut combinatorial optimization problem.
     """

    def __init__(self, graph: nx.Graph, seed: Optional[int] = None):
        super().__init__(graph, maximize=True, seed=seed)

        # we have one qubit per graph node
        self.num_qubits = self.num_nodes - 1

    def _get_best_known_value(self):
        if self.num_nodes <= 25:
            return self._brute_force_max_cut()
        else:
            return self._gw_optimum()

    def _brute_force_max_cut(self):
        #brute force exact Max-Cut for small graphs (N <= 25)

        best_cut = 0.0
        best_assignment = None

        for assignment in iterproduct([0, 1], repeat=self.num_nodes):
            current_cut = self.evaluate(list(assignment))

            if current_cut > best_cut:
                best_cut = current_cut
                best_assignment = assignment

        return best_cut

    def _gw_optimum(self):
        #gets the best cut value obtained by the Goemans-Williamson optimizer

        from qiskit_optimization.applications import Maxcut
        from qiskit_optimization.algorithms import GoemansWilliamsonOptimizer

        maxcut = Maxcut(self.graph)
        qp = maxcut.to_quadratic_program()

        # num_cuts determines how many random hyperplane projections to try.
        # higher numbers increase the chance of finding the best approximation.
        gw_optimizer = GoemansWilliamsonOptimizer(num_cuts=20, seed=self.seed)

        result = gw_optimizer.solve(qp)

        return result.fval

    def _build_hamiltonian(self) -> HamiltonianType:
        terms = []
        coeff = +0.5
        for u, v in self.graph.edges():
            if u == 0:
                terms.append((coeff, [v-1]))
            elif v == 0:
                terms.append((coeff, [u-1]))
            else:
                terms.append((coeff, sorted([u-1, v-1])))
        return terms

    def evaluate(self, assignment):
        cut = 0
        for u, v in self.graph.edges():
            if assignment[u] != assignment[v]:
                cut += 1.0
        return cut

    def get_approximation_ratio(self, energy: float) -> float:
        cut_value = self.energy_to_cut(energy)
        return cut_value / self.best_known_value

    def energy_to_cut(self, energy: float) -> float:
        return (0.5 * self.num_edges) - energy

    def cut_to_energy(self, cut: float) -> float:
        return (0.5 * self.num_edges) - cut