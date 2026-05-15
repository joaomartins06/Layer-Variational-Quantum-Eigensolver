import numpy as np
import networkx as nx
from itertools import product as iterproduct



class CommunityDetection:
    """
     Defines the community detection problem and has some methods to compute 
     the Hamiltonian and best known value.
    """

    def __init__(self, graph: nx.Graph, k: int):
        self.graph = graph
        self.k = k
        self.n_nodes = graph.number_of_nodes()
        self.m = graph.number_of_edges()
        #number of bits/qubits to encode k communities
        self.N = int(np.ceil(np.log2(k)))
        self.n_qubits = self.n_nodes * self.N

        # modularity matrix B_{u,v} = A_{u,v} - d_u*d_v / (2m)
        #taken from the paper

        #get adjacency matrix and degree vector
        A = nx.to_numpy_array(graph)
        #get the degree of each node
        degrees = np.array([d for _, d in graph.degree()])
        #compute the modularity matrix B
        self.B = A - np.outer(degrees, degrees) / (2 * self.m)

        # build Hamiltonian once and cache it
        self.terms = self._build_hamiltonian()


    def _qubit_index(self, bit: int, node: int):
        return bit * self.n_nodes + node


    def _build_hamiltonian(self):
 
        terms = []
        #denominator in the hamiltonian, takes into account the 1/2 in the product
        prefactor = -1.0 / (2 * self.m * (2 ** self.N))

        for u in range(self.n_nodes):
            for v in range(self.n_nodes):
                b_uv = self.B[u, v]
                # skip negligible terms in case they are not really 0
                if abs(b_uv) < 1e-12:
                    continue
                
                coeff_base = prefactor * b_uv
                #gets all combinations of bits for the communities depending on the encoding 
                #this is the loop from j= 1 to N in the paper
                for j in iterproduct([0, 1], repeat=self.N):

                    #qubits where Z acts in this particular term
                    qubits = []

                    for s, bit in enumerate(j):
                        #bit=0 means this position contributes ident
                        #bit=1 means this position contributes ZZ
                        if bit:
                            qubits.append(self._qubit_index(s, u))
                            qubits.append(self._qubit_index(s, v))

                    terms.append((coeff_base, sorted(qubits)))

        return terms
    

    def _modularity(self, assignment):
        Q = 0.0
        for u in range(self.n_nodes):
            for v in range(self.n_nodes):
                if assignment[u] == assignment[v]:
                    Q += self.B[u, v]
        return Q / (2 * self.m)


    def best_known_value(self):

        if self.n_nodes <= 12:
            return self._brute_force_optimum()
        else:
            return self._louvain_optimum()


    def _brute_force_optimum(self):
        #just brute force all possible assignments of nodes to communities and compute the modularity
        best = -np.inf
        for assignment in iterproduct(range(self.k), repeat=self.n_nodes):
            Q = self._modularity(assignment)
            if Q > best:
                best = Q
        return best


    def _louvain_optimum(self):
        #gets the best modularity found by the louvain algorithm
        from networkx.algorithms.community import louvain_communities
        best = -np.inf
        for _ in range(20):
            communities = louvain_communities(self.graph, seed=None)
            # convert to assignment list
            assignment = [0] * self.n_nodes
            nodes = list(self.graph.nodes())
            for comm_idx, comm in enumerate(communities):
                for node in comm:
                    assignment[nodes.index(node)] = comm_idx
            Q = self._modularity(tuple(assignment))
            if Q > best:
                best = Q

        return best


    def set_best_known_value(self, value: float):
        self._best_known_value = value


    def approximation_ratio(self, energy: float, best_known: float) -> float:
        return -energy / best_known