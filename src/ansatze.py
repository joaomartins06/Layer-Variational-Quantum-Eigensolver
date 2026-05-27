import numpy as np


class Ansatz:
    ''' 
    Define the ansatz circuit to be optimised.
    This class creates an abstraction from how we compute things
    For smaller graphs and for a noisy case, we will be using a circuit.
    For bigger graphs, we will be using a tensor network (there is a limit of around 25 qubits for the circuit)
    '''

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_layers = 0

        #pre-compute the CNOT pairs for the layers > 0
        self.even_odd_pairs = [(i, i+1) for i in range(0, n_qubits-1, 2)]
        self.odd_even_pairs = [(i, i+1) for i in range(1, n_qubits-1, 2)]

        #randomly initialize parameters for the first layer
        self.params = np.random.uniform(0, 2*np.pi, size=n_qubits)


    def add_layer(self):
        #easy to check that, according to the paper's construction, each layer adds 4*n_qubits - 4 rotations
        n_new = 4 * self.n_qubits - 4

        #following the paper, the newly added parameters are initialized as zero
        self.params = np.concatenate([self.params, np.zeros(n_new)])

        self.n_layers += 1


    def param_count(self):
        return len(self.params)


    def get_gates(self, params: np.ndarray) -> list:
        #get the list of gates
        #for the rotations, an element is ('Ry', qubit_index, rotation_angle)
        #for the CNOTs, an element is ('CNOT', control_qubit_index, target_qubit_index)
        #this will be used later

        gates = []
        idx = 0 

        #layer 0
        for q in range(self.n_qubits):
            gates.append(('Ry', q, params[idx]))
            idx += 1

        #layer > 0
        for _ in range(self.n_layers):
            
            #start with a series of CNOTs then Ry on all qubits
            #this is the even-odd part
            for control, target in self.even_odd_pairs:
                gates.append(('CNOT', control, target))
            for q1, q2 in self.even_odd_pairs:
                gates.append(('Ry', q1, params[idx]))
                gates.append(('Ry', q2, params[idx+1]))
                idx += 2
            #repeat the even-odd part
            for control, target in self.even_odd_pairs:
                gates.append(('CNOT', control, target))
            for q1, q2 in self.even_odd_pairs:
                gates.append(('Ry', q1, params[idx]))
                gates.append(('Ry', q2, params[idx+1]))
                idx += 2

            #same thing, but now for the odd-even part
            for control, target in self.odd_even_pairs:
                gates.append(('CNOT', control, target))
            for q1, q2 in self.odd_even_pairs:
                gates.append(('Ry', q1, params[idx]))
                gates.append(('Ry', q2, params[idx+1]))
                idx += 2
            #repeat the odd-even part
            for control, target in self.odd_even_pairs:
                gates.append(('CNOT', control, target))
            for q1, q2 in self.odd_even_pairs:
                gates.append(('Ry', q1, params[idx]))
                gates.append(('Ry', q2, params[idx+1]))
                idx += 2

        return gates