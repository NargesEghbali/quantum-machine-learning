#!/usr/bin/env python
# coding: utf-8



import cirq
import numpy as np
import math
from swap_test import SwapTest




class Euclidean_distance:
    def __init__(self, input_state_dim,prepare_input_states=False,copies=10000):
        self.prepare_input_states = prepare_input_states
        self.input_state_dim = input_state_dim
        self.copies = copies
        self.nq = 0
        self.control_qubit = cirq.LineQubit(0)
        self.nq += 1
        self.num_qubits_per_state= int(np.log2(self.input_state_dim))
        self.state_store_qubits = [cirq.LineQubit(i) for i in range(self.nq,self.nq +self.num_qubits_per_state)]
        self.nq += self.num_qubits_per_state
        if self.prepare_input_states:
            self.input_1 = [cirq.LineQubit(i) for i in range(self.nq, self.nq +self.num_qubits_per_state)]
            self.nq += self.num_qubits_per_state
            self.input_2 = [cirq.LineQubit(i) for i in range(self.nq, self.nq +self.num_qubits_per_state)]
            self.nq += self.num_qubits_per_state
        self.other_state_qubits = [cirq.LineQubit(i) for i in range(self.nq,self.nq + 1 +self.num_qubits_per_state)]
        self.nq += 1 + self.num_qubits_per_state
        self.circuit = cirq.Circuit()
        
        
    def dist_circuit(self, input_1_norm=1, input_2_norm=1,input_1=None,input_2=None,
                     input_1_transforms=None,input_2_transforms=None,input_1_circuit=None,input_2_circuit=None):
        self.input_1_norm = input_1_norm
        self.input_2_norm = input_2_norm
        self.input_1_circuit = input_1_circuit
        self.input_2_circuit = input_2_circuit
        if input_1 is not None:
            self.input_1 = input_1
        if input_2 is not None:
            self.input_2 = input_2
        if input_1_transforms is not None:
            self.input_1_circuit = []
            for op in input_1_transforms:
                self.circuit.append(op.on_each(self.input_1))
                self.input_1_circuit.append(op.on_each(self.input_1))
        if input_2_transforms is not None:
            self.input_2_circuit = []
            for op in input_2_transforms:
                self.circuit.append(op.on_each(self.input_2))
                self.input_2_circuit.append(op.on_each(self.input_2))
        self.input_1_uncompute = cirq.inverse(self.input_1_circuit)
        self.input_2_uncompute = cirq.inverse(self.input_2_circuit)
        
        # Create the required state 1
        self.circuit.append(cirq.H(self.control_qubit))
        for i in range(len(self.input_2)):
            self.circuit.append(cirq.CSWAP(self.control_qubit,self.state_store_qubits[i],self.input_2[i]))
        self.circuit.append(cirq.X(self.control_qubit))
        for i in range(len(self.input_1)):
            self.circuit.append(cirq.CSWAP(self.control_qubit,self.state_store_qubits[i],self.input_1[i]))
        for c in self.input_2_uncompute:
            self.circuit.append(c[0].controlled_by(self.control_qubit))
        self.circuit.append(cirq.X(self.control_qubit))
        for c in self.input_1_uncompute:
            self.circuit.append(c[0].controlled_by(self.control_qubit))
            
        # Prepare the other state qubit
        self.Z = self.input_1_norm**2 + self.input_2_norm**2
        print(self.Z)
        theta = 2*math.acos(self.input_1_norm/np.sqrt(self.Z))
        self.circuit.append(cirq.ry(theta).on(self.other_state_qubits[0]))
        self.circuit.append(cirq.Z(self.other_state_qubits[0]))
        self.st = SwapTest(prepare_input_states=False,input_state_dim=4,nq=self.nq,measure=False)
        print(self.other_state_qubits)
        self.state = [self.control_qubit] +self.state_store_qubits
        self.st.build_circuit(input_1=self.state,input_2=self.other_state_qubits)
        self.circuit += self.st.circuit
        self.circuit.append(cirq.measure(self.st.ancilla_qubit, key='k'))
        print(self.circuit)
        
        
    def compute_distance(self):
        sim = cirq.Simulator()
        results = sim.run(self.circuit,repetitions=self.copies).histogram(key='k')
        results = dict(results)
        print(results)
        results = dict(results)
        prob_0 = results[0]/self.copies
        print(prob_0)
        Euclidean_distance = 4*self.Z*max((prob_0 - 0.5),0)
        print("Euclidean distance",Euclidean_distance)


