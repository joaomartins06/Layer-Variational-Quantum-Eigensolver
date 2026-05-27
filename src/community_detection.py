import numpy as np
import networkx as nx
from .problem import Problem, HamiltonianType
from itertools import product as iterproduct
from collections import defaultdict
from typing import Optional

class CommunityDetection(Problem):
    """
     Implementation of the k-Community Detection problem, as described in the Layer-VQE paper.
    """

    def __init__(self, graph: nx.Graph, k: int, seed: Optional[int] = None):
        super().__init__(graph, maximize=False, seed=seed)
        self.k = k

        #number of bits/qubits to encode k communities
        self.N = int(np.ceil(np.log2(k)))
        self.num_qubits = self.num_nodes * self.N

        # modularity matrix B_{u,v} = A_{u,v} - d_u*d_v / (2m)
        #taken from the paper

        #get adjacency matrix and degree vector
        A = nx.to_numpy_array(graph)
        #get the degree of each node
        degrees = np.array([d for _, d in graph.degree()])
        #compute the modularity matrix B
        self.B = A - np.outer(degrees, degrees) / (2 * self.num_edges)

    def _get_best_known_value(self):
        if self.num_nodes <= 12:
            return -self._brute_force_optimum()
        else:
            return -self._louvain_optimum()

    def _brute_force_optimum(self) -> float:
        #just brute force all possible assignments of nodes to communities and compute the modularity
        best = -np.inf
        for assignment in iterproduct(range(self.k), repeat=self.num_nodes):
            Q = self.evaluate(list(assignment))
            if Q > best:
                best = Q
        return best

    def _louvain_optimum(self) -> float:
        #gets the best modularity found by the louvain algorithm
        from networkx.algorithms.community import louvain_communities
        best = -np.inf
        nodes_list = list(self.graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes_list)}
        for _ in range(20):
            communities = louvain_communities(self.graph, seed=self.seed)
            # convert to assignment list
            assignment = [0] * self.num_nodes
            for comm_idx, comm in enumerate(communities):
                for node in comm:
                    assignment[node_to_idx[node]] = comm_idx
            Q = self.evaluate(assignment)
            if Q > best:
                best = Q

        return best

    def _build_hamiltonian(self) -> HamiltonianType:
        
        grouped = defaultdict(float)
        prefactor = -1.0 / (2 * self.num_edges * (2 ** self.N))

        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                b_uv = self.B[u, v]
                if abs(b_uv) < 1e-12:
                    continue
                coeff_base = prefactor * b_uv
                
                for S in iterproduct([0, 1], repeat=self.N):
                    qubits = []
                    for j, include in enumerate(S):
                        if include:
                            qubits.append(self._qubit_index(j, u))
                            qubits.append(self._qubit_index(j, v))
                    # use tuple as dict key (lists not hashable)
                    # also: Z*Z = I, so duplicate qubits cancel
                    # reduce by removing pairs
                    reduced = tuple(sorted(self._reduce_z_squared(qubits)))
                    grouped[reduced] += coeff_base

        # filter out near-zero coefficients
        return [
            (coeff, list(qubits))
            for qubits, coeff in grouped.items()
            if abs(coeff) > 1e-12
        ]

    def _qubit_index(self, bit: int, node: int):
        return bit * self.num_nodes + node

    def _reduce_z_squared(self, qubits):
        """
        Z * Z = I, so any qubit appearing an even number of times cancels.
        Returns only qubits appearing an odd number of times.
        """
        from collections import Counter
        counts = Counter(qubits)
        return [q for q, c in counts.items() if c % 2 == 1]

    def evaluate(self, assignment):
        Q = 0.0
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if assignment[u] == assignment[v]:
                    Q += self.B[u, v]
        return Q / (2 * self.num_edges)

    def get_approximation_ratio(self, energy: float) -> float:
        return energy / self.best_known_value