import numpy as np
import quimb as qu
import quimb.tensor as qtn
from collections import Counter

from src.community_detection import CommunityDetection
from src.maxcut import MaxCut
import autoray as ar


class QuimbSimulator:

    def __init__(self, use_gpu: bool = False, max_bond: int = None):
        self.max_bond = max_bond


    def _build_circuit(self, params, ansatz):
        #notice this uses an object of the class Ansatz
        #initialize the circuit
        circ = qtn.CircuitMPS(ansatz.n_qubits)
        for gate in ansatz.get_gates(params):
            #add a rotation
            if gate[0] == 'Ry':
                _, qubit, angle = gate
                circ.apply_gate('RY', angle, qubit)
            #add a CNOT
            elif gate[0] == 'CNOT':
                _, control, target = gate
                circ.apply_gate('CNOT', control, target)
        #get the full circuit
        return circ


    def _pauli_z_string(self, qubits):
        #get a string of Z operatores
        operator = qu.pauli('Z')
        for _ in range(len(qubits) - 1):
            operator = operator & qu.pauli('Z')
        return operator


    def expectation(self, params, ansatz, problem):
        #computing the expectation ANALITICALLY value according to the hamiltonian defined in the problem
        with ar.backend_like('numpy'):
            #initialize the circuit
            circ = self._build_circuit(params, ansatz)

            #problem is an object of the class CommunityDetection
            #.hamiltonian_terms is the list of terms in the hamiltonian, each term is a tuple (coefficient, qubits)
            identity_energy = sum(coeff for coeff, qubits in problem.hamiltonian_terms if not qubits)
            operator_terms = [(c, q) for c, q in problem.hamiltonian_terms if q]

            energy = identity_energy
            for coeff, qubits in operator_terms:
                #get the expectation value for this term
                #notice that the hamiltonian is a sum of pauli strings, so we compute the expectation value
                #of each term separately and sum them up
                operator = self._pauli_z_string(qubits)
                energy += coeff * circ.local_expectation(operator, qubits)

        #get the real part (it should be real, but there is always some numerical error)
        return float(energy.real)


    def sample_expectation(self, params, ansatz, problem, n_samples=2000):
        #computing the expectation value by sampling bitstrings from the circuit
        #in a vectorized manner
        #initialize the circuit
        circ = self._build_circuit(params, ansatz)
        counts = Counter(circ.sample(n_samples))

        bitstrings = np.array([[int(b) for b in bs] for bs in counts.keys()])
        weights = np.array(list(counts.values()), dtype=float) / n_samples

        energy = sum(coeff for coeff, qubits in problem.hamiltonian_terms if not qubits)
        for coeff, qubits in problem.hamiltonian_terms:
            if not qubits:
                continue
            eigenvalues = np.prod(1 - 2 * bitstrings[:, qubits], axis=1)
            energy += coeff * float(np.dot(weights, eigenvalues))

        return energy

    def get_most_frequent_assignments(self, params, ansatz, problem, n_samples=2000):
        # TODO: rewrite

        circ = self._build_circuit(params, ansatz)

        counts = Counter(circ.sample(n_samples))

        bitstrings = []

        if isinstance(problem, MaxCut):
            for bitstring, count in counts.most_common(5):
                measured_bits = [int(b) for b in bitstring]
                full_assignment = [0] + measured_bits
                probability = (count / n_samples) * 100
                bitstrings.append((full_assignment, probability))

        elif isinstance(problem, CommunityDetection):
            n, N = problem.num_nodes, problem.N
            for bitstring, count in counts.most_common(5):
                bits = [int(b) for b in bitstring]
                assignment = [
                    sum(bits[j * n + v] * (2 ** j) for j in range(N))
                    for v in range(n)
                ]
                probability = (count / n_samples) * 100
                bitstrings.append((assignment, probability))

        else:
            raise NotImplementedError

        return bitstrings


