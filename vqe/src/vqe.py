import numpy as np
import scipy

import cirq

from openfermion import expectation
from openfermion.linalg import get_sparse_operator
from src.utils_vqe import fix_nelec_in_state_vector

class VqeHardwareEfficient(object):
    def __init__(self, n_qubits, n_layers, n_electrons=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qubits = cirq.LineQubit.range(n_qubits)
        self.num_params = n_qubits * (n_layers + 1)
        self.final_state_vector_best = None
        self.best_vqe_params = None
        self.best_vqe_energy = None
        self.n_electrons = n_electrons

    def layers(self, params):
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        hw_eff_cirq = cirq.Circuit()

        for q in range(n_qubits):
            hw_eff_cirq.append(cirq.Ry(rads=params[q, 0])(self.qubits[q]))
            hw_eff_cirq.append(cirq.Rz(rads=params[q, 0])(self.qubits[q]))

        for i in range(n_layers):
            hw_eff_cirq.append([cirq.CNOT(self.qubits[q], self.qubits[q + 1]) for q in range(n_qubits-1)])

            for q in range(n_qubits):
                hw_eff_cirq.append(cirq.Ry(rads=params[q, i+1])(self.qubits[q]))
                hw_eff_cirq.append(cirq.Rz(rads=params[q, i+1])(self.qubits[q]))

        return hw_eff_cirq

    def run(self, hamiltonian, init_vals, options):
        def func(x):
            params = x.reshape((self.n_qubits, self.n_layers + 1))
            simulator = cirq.Simulator()
            result = simulator.simulate(program=self.layers(params),
                                        qubit_order=self.qubits
                                        )
            real_wave_function = (result.final_state_vector + result.final_state_vector.conj()) / 2
            norm_real_wf = np.dot(real_wave_function.conj(), real_wave_function)**(0.5)
            real_wave_function /= norm_real_wf
            return np.real(expectation(get_sparse_operator(hamiltonian),
                           real_wave_function)
                           )

        def callbackF(x):
            np.set_printoptions(precision=4,
                                suppress=True,
                                linewidth=10000,
                                sign=" ")
            print(np.array(x), f"{func(x):+.5f}")

        if 'callback' in options:
            callback = [None, callbackF][options['callback']]
            options.pop('callback')
        else:
            callback = None

        res = scipy.optimize.minimize(func,
                                      init_vals,
                                      tol=1e-7,
                                      method='COBYLA',
                                      options=options,
                                      callback=callback
                                      )

        self.best_vqe_params = res['x']
        self.best_vqe_energy = res['fun']

        params = res['x'].reshape((self.n_qubits, self.n_layers + 1))
        simulator = cirq.Simulator()
        result = simulator.simulate(program=self.layers(params),
                                    qubit_order=self.qubits
                                    )
        real_wave_function = (result.final_state_vector + result.final_state_vector.conj()) / 2
        norm_real_wf = np.dot(real_wave_function.conj(), real_wave_function) ** (0.5)
        real_wave_function /= norm_real_wf
        if self.n_electrons:
            print("# Projecting VQE wave function on the correct n_electron subspace:", self.n_electrons)
            self.final_state_vector_best = fix_nelec_in_state_vector(real_wave_function, self.n_electrons)
            self.best_vqe_energy = np.real(expectation(get_sparse_operator(hamiltonian),
                                self.final_state_vector_best))

        else:
            self.final_state_vector_best = real_wave_function
        return res

