import numpy as np
import quimb as qu
import quimb.tensor as qtn
from collections import Counter
from .maxcut import MaxCut
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
        #this is the realistic approach
        #initialize the circuit
        circ = self._build_circuit(params, ansatz)

        counts = Counter(circ.sample(n_samples))

        # estimate the energy from the samples
        energy = 0.0
        for bitstring, count in counts.items():
            #parse the string in an array of bits
            x = np.array([int(b) for b in bitstring])
            #evaluate the hamiltonian for a certain bitstring
            energy += (count / n_samples) * self._evaluate_hamiltonian(x, problem)

        return energy

    def _evaluate_hamiltonian(self, x, problem):
        #aux funtion to compute the energy for a given bitstring x
        energy = 0.0

        for coeff, qubits in problem.hamiltonian_terms:
            if len(qubits) == 0:
                energy += coeff

            else:
                #compute the parity of the bits in the qubits of this term
                eigenvalue = 1.0
                for q in qubits:
                    # +1 if x[q]=0, -1 if x[q]=1
                    eigenvalue *= (1 - 2 * x[q])  # +1 if x[q]=0, -1 if x[q]=1

                energy += coeff * eigenvalue

        return energy

    def get_most_frequent_assignments(self, params, ansatz, problem, n_samples=2000):
        # TODO: rewrite

        circ = self._build_circuit(params, ansatz)

        counts = Counter(circ.sample(n_samples))

        bitstrings = []

        if not(isinstance(problem, MaxCut)):
            raise NotImplementedError
        else:
            for bitstring, count in counts.most_common(5):
                measured_bits = [int(b) for b in bitstring]
                full_assignment = [0] + measured_bits

                probability = (count / n_samples) * 100
                bitstrings.append((full_assignment, probability))

            return bitstrings
