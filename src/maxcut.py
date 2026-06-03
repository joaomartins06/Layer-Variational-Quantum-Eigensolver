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

    @property
    def num_qubits(self) -> int:
        # one qubit per graph node, minus one because we remove the problem symmetry
        return self.num_nodes - 1

    def _get_best_known_value(self):
        """Return the best known value *as a cut value*, not as an energy"""
        if self.num_nodes <= 20:
            return self._brute_force_max_cut()
        else:
            return self._gw_optimum()

    def _brute_force_max_cut(self):
        """For small graphs (N <= 20), Max-Cut can be solved exactly by brute force."""

        best_cut = 0.0
        for assignment in iterproduct([0, 1], repeat=self.num_nodes):
            current_cut = self.evaluate(list(assignment))
            if current_cut > best_cut:
                best_cut = current_cut
        return best_cut

    def _gw_optimum(self):
        """For larger graphs (N > 20), we use the Goemans-Williamson (GW) optimizer to approximately solve Max-Cut.
        GW is guaranteed to find a solution with approximation ratio >= 0.87856 in polynomial time."""

        # The Qiskit Optimization framework provides neat easy-to-use tools to solve Max-Cut with GW.
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
        """
        Builds Hamiltonian terms for Max-Cut, in its spin variables formulation.
        Originally, the Hamiltonian should be H = 0.5  * sum_{(u,v) in E} (I - Z_u Z_v),
        but we can drop the constant energy term 0.5 * |E| * I,
        hence we just construct H = -0.5 * sum_{(u,v) in E} Z_u Z_v
        """

        terms = []
        coeff = -0.5
        for u, v in self.graph.edges():
            # following Amaro et al. (2022), break problem symmetry to save one qubit
            if u == 0:
                terms.append((coeff, [v-1]))
            elif v == 0:
                terms.append((coeff, [u-1]))
            else:
                terms.append((coeff, sorted([u-1, v-1])))
        return terms

    def evaluate(self, assignment) -> float:
        """
        Evaluate the cut encoded by the bitstring assignment by computing the corresponding cut value.
        """

        cut = 0
        for u, v in self.graph.edges():
            if assignment[u] != assignment[v]:
                cut += 1.0
        return cut

    def get_approximation_ratio(self, energy: float) -> float:
        # for the approximation ratio, we use the cut values
        # because that is how self.best_known_value is expressed
        cut_value = self.energy_to_cut(energy)
        return cut_value / self.best_known_value

    def energy_to_cut(self, energy: float) -> float:
        """Converts an Ising energy into the corresponding cut value,
         taking into account the dropped constant term in the Hamiltonian."""
        return energy + (0.5 * self.num_edges)

    def cut_to_energy(self, cut: float) -> float:
        """Converts a cut value back into the corresponding Ising energy,
        taking into account the dropped constant term in the Hamiltonian."""
        return cut - (0.5 * self.num_edges)