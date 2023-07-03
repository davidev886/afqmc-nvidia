import numpy as np
import scipy
import cirq
from openfermion import expectation
from openfermion.linalg import get_sparse_operator
from src.qnp_gates import QNP_PX, QNP_OR
from src.utils_vqe import number_operator, spin_z_operator


class VqeQNP(object):
    """
    Implements the quantum-number-preserving ansatz from
    Anselmetti et al. (2021) <https://doi.org/10.1088/1367-2630/ac2cb3>`
    """

    def __init__(self, n_qubits, n_layers, hamiltonian, init_mo_occ=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.number_of_Q_blocks = n_qubits // 2 - 1
        self.num_parameters = 2 * self.number_of_Q_blocks * n_layers
        self.init_mo_occ = init_mo_occ
        self.final_state_vector_best = None
        self.best_vqe_params = None
        self.best_vqe_energy = None
        self.hamiltonian = get_sparse_operator(hamiltonian)
        self.qubits = cirq.LineQubit.range(n_qubits)
        print("# VQE num_parameters", self.num_parameters)
        self.initial_x_gates_pos = self.prepare_initial_circuit()

    def prepare_initial_circuit(self):
        """
        Creates a list with the position of the X gates that should be applied to the initial |00...0>
        state to set the number of electrons and the spin correctly
        """
        x_gates_pos_list = []
        if self.init_mo_occ is not None:
            for idx_occ, occ in enumerate(self.init_mo_occ):
                if int(occ) == 2:
                    x_gates_pos_list.extend([2 * idx_occ, 2 * idx_occ + 1])
                elif int(occ) == 1:
                    x_gates_pos_list.append(2 * idx_occ)

        return x_gates_pos_list

    def ansatz(self, params_ansatz):
        """
            params: list/np.array
            [theta_0, ..., theta_{M-1}, phi_0, ..., phi_{M-1}]
            where M is the total number of blocks = layer * (n_qubits/2 - 1)
        """

        n_layers = self.n_layers
        number_of_blocks = self.number_of_Q_blocks

        thetas = params_ansatz[:number_of_blocks * n_layers]
        phis = params_ansatz[number_of_blocks * n_layers:]
        qnp_cirq_ansatz = cirq.Circuit()

        for init_gate_position in self.initial_x_gates_pos:
            qnp_cirq_ansatz.append(cirq.X(self.qubits[init_gate_position]))

        for idx_layer in range(n_layers):
            # print("\nlayer:", idx_layer)
            for idx_block in range(0, number_of_blocks, 2):
                qubit_list = [self.qubits[2 * idx_block + j] for j in range(4)]
                # print(idx_block, "theta", idx_layer * number_of_blocks + idx_block, "qubits", qubit_list)
                qnp_cirq_ansatz.append(QNP_PX(theta=thetas[idx_layer * number_of_blocks + idx_block]).on(*qubit_list))
                qnp_cirq_ansatz.append(QNP_OR(theta=phis[idx_layer * number_of_blocks + idx_block]).on(*qubit_list))

            for idx_block in range(1, number_of_blocks, 2):
                qubit_list = [self.qubits[2 * idx_block + j] for j in range(4)]
                # print(idx_block, "theta", idx_layer * number_of_blocks + idx_block, "qubits", qubit_list)
                qnp_cirq_ansatz.append(QNP_PX(theta=thetas[idx_layer * number_of_blocks + idx_block]).on(*qubit_list))
                qnp_cirq_ansatz.append(QNP_OR(theta=phis[idx_layer * number_of_blocks + idx_block]).on(*qubit_list))

        return qnp_cirq_ansatz

    def func(self, x):
        """
            Computes the expectation value of the Hamiltonian

            x is a numpy array with self.num_parameters elements
            and contains the variational parameters in this order
            x = [theta_0, ..., theta_{M-1}, phi_0, ..., phi_{M-1}]
            where M  = layer * (n_qubits/2 - 1) is the total number of blocks
        """
        sim = cirq.Simulator()
        result = sim.simulate(program=self.ansatz(x))
        state_vector = result.final_state_vector
        return np.real(expectation(self.hamiltonian,
                                   state_vector)
                       )

    def callback_f(self, x):
        """A callback function for the vqe optimization steps
        """
        np.set_printoptions(precision=6,
                            suppress=True,
                            linewidth=10000,
                            sign=" ")

        sim = cirq.Simulator()
        result = sim.simulate(program=self.ansatz(x))
        state_vector = result.final_state_vector

        n_electrons = expectation(get_sparse_operator(number_operator(self.n_qubits)), state_vector)
        spin_z = expectation(get_sparse_operator(spin_z_operator(self.n_qubits)), state_vector)

        print(np.array(x), f"{self.func(x):+.5f} {n_electrons.real:+.5f} {spin_z.real:+.5f}")

        energy_iter = self.func(x)
        return energy_iter

    def check_initial_circuit(self):
        """
         Compute the energy, the number of electrons and the spin of the initial state
         (when the VQE variational parameters are set to 0.0)
        """
        qnp_cirq_ansatz = cirq.Circuit()

        for init_gate_position in self.initial_x_gates_pos:
            qnp_cirq_ansatz.append(cirq.X(self.qubits[init_gate_position]))

        sim = cirq.Simulator()
        result = sim.simulate(program=qnp_cirq_ansatz, qubit_order=self.qubits)
        state_vector = result.final_state_vector

        energy = np.real(expectation(self.hamiltonian, state_vector))
        n_electrons = np.real(expectation(get_sparse_operator(number_operator(self.n_qubits)), state_vector))
        spin_z = np.real(expectation(get_sparse_operator(spin_z_operator(self.n_qubits)), state_vector))

        return energy, n_electrons, spin_z

    def run(self, init_vals, options):
        """
            Runs the VQE optimization
        """

        energy, n_electrons, spin_z = self.check_initial_circuit()
        print("# Energy of the initial VQE state with params=0.0:", energy)
        print("# Number of electrons of the initial VQE state with params=0.0:", n_electrons)
        print("# Spin of the initial VQE state with params=0.0:", spin_z)

        if 'callback' in options:

            callback = [None, self.callback_f][options['callback']]

            options.pop('callback')
            if callback:
                energy_iter = self.func(init_vals)
                string_params = "params"
                print(f"#\t\t {string_params:30s} energy\t  n_electron\t spin")
                print(energy_iter)
        else:
            callback = None

        res = scipy.optimize.minimize(self.func,
                                      init_vals,
                                      tol=1e-8,
                                      method='COBYLA',
                                      options=options,
                                      callback=callback,
                                      )

        self.best_vqe_params = res['x']
        self.best_vqe_energy = res['fun']

        print("Best VQE parameters", self.best_vqe_params)
        print("Best VQE energy", self.best_vqe_energy)

        # Compute the state vector corresponding to the best energy
        sim = cirq.Simulator()
        result = sim.simulate(program=self.ansatz(res['x']))

        self.final_state_vector_best = result.final_state_vector

        return res
