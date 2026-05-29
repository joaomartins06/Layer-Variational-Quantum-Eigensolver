from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, Tuple, Union, Optional
import networkx as nx

HamiltonianType = List[Tuple[float, List[int]]]

class Problem(ABC):
    """
    Base class for combinatorial optimization (CO) problems, as presented in the Layer-VQE paper.
    For now, restricted to CO on finite unweighted graphs.
    """

    def __init__(self, graph: nx.Graph, maximize: bool = False, seed: Optional[int] = None):
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()
        self.num_qubits = None
        self.maximize = maximize  # True for a maximization problem, False for minimization
        self.seed = seed

    @cached_property
    def best_known_value(self) -> Union[float, None]:
        return self._get_best_known_value()

    @cached_property
    def hamiltonian_terms(self) -> HamiltonianType:
        return self._build_hamiltonian()

    @abstractmethod
    def _get_best_known_value(self) -> Union[float, None]:
        pass

    @abstractmethod
    def _build_hamiltonian(self) -> HamiltonianType:
        """Builds the terms of the Hamiltonian encoding the problem.
        Expected format : [(coeff, [qubit_1, qubit_2, ...]), ...]"""
        pass

    @abstractmethod
    def evaluate(self, assignment: List[int]) -> float:
        pass

    @abstractmethod
    def get_approximation_ratio(self, energy: float) -> float:
        pass