import cirq
import math


class QNP_OR(cirq.Gate):
    def __init__(self, theta):
        """
        Implementation of the spatial orbital rotation gate.
        See Fig. 2 of Anselmetti et al New J. Phys. 23 (2021) 113010
        """
        super(QNP_OR, self)
        self.theta = theta

    def _num_qubits_(self):
        return 4

    def _decompose_(self, qubits):
        a, b, c, d = qubits
        yield cirq.givens(angle_rads=-self.theta/2).on(a, c)
        yield cirq.givens(angle_rads=-self.theta/2).on(b, d)

    def _circuit_diagram_info_(self, _):
        return [f"QNP_OR({self.theta:.2f})"] * self.num_qubits()


class QNP_PX(cirq.Gate):
    def __init__(self, theta):
        """
        Implementation of the pair exchange gate with CNOT and Controlled Givens rotations
        See Fig. 2 of Anselmetti et al New J. Phys. 23 (2021) 113010
        """
        super(QNP_PX, self)
        self.theta = theta

    def _num_qubits_(self):
        return 4

    def _decompose_(self, qubits):
        a, b, c, d = qubits
        yield cirq.X(b)
        yield cirq.CNOT(b, a)
        yield cirq.X(b)
        yield cirq.X(c)
        yield cirq.CNOT(c, d)
        yield cirq.X(c)
        yield cirq.givens(angle_rads=-self.theta/2).on(b, c).controlled_by(a, d)
        yield cirq.X(b)
        yield cirq.CNOT(b, a)
        yield cirq.X(b)
        yield cirq.X(c)
        yield cirq.CNOT(c, d)
        yield cirq.X(c)

    def _circuit_diagram_info_(self, _):
        return [f"QNP_PX({self.theta:.2f})"] * self.num_qubits()


class QNP_PX_decomposed(cirq.Gate):
    def __init__(self, theta):
        """
        Implementation of the pair exchange gate decomposed in terms of standard gates and controlled Y rotations.
        See Appendix E1 of Anselmetti et al New J. Phys. 23 (2021) 113010
        """
        super(QNP_PX_decomposed, self)
        self.theta = theta

    def _num_qubits_(self):
        return 4

    def _decompose_(self, qubits):
        a, b, c, d = qubits
        yield cirq.CNOT(b, a)
        yield cirq.CNOT(d, c)
        yield cirq.X(a)
        yield cirq.CNOT(d, b)
        yield cirq.ControlledGate(cirq.Ry(rads=self.theta/4)).on(a, d)
        yield cirq.CNOT(a, c)
        yield cirq.ControlledGate(cirq.Ry(rads=self.theta/4)).on(c, d)
        yield cirq.CNOT(a, c)
        yield cirq.ControlledGate(cirq.Ry(rads=-self.theta/4)).on(c, d)
        yield cirq.CZ(b, d)
        yield cirq.ControlledGate(cirq.Ry(rads=-self.theta/4)).on(a, d)
        yield cirq.CNOT(a, c)
        yield cirq.Rz(rads=math.pi/2).on(b)
        yield cirq.ControlledGate(cirq.Ry(rads=-self.theta/4)).on(c, d)
        yield cirq.CNOT(a, c)
        yield cirq.X(a)
        yield cirq.ControlledGate(cirq.Ry(rads=self.theta/4)).on(c, d)
        yield cirq.CNOT(d, b)
        yield cirq.Rz(rads=-math.pi/2).on(b)
        yield cirq.S(d)
        yield cirq.CNOT(d, c)
        yield cirq.CNOT(b, a)

    def _circuit_diagram_info_(self, _):
        return [f"QNP_PX({self.theta:.2f})"] * self.num_qubits()


class QNP_OR_decomposed(cirq.Gate):
    def __init__(self, theta):
        """
        Implementation of the spatial orbital rotation gate decomposed in terms of standard gates.
        See Appendix E2 of Anselmetti et al New J. Phys. 23 (2021) 113010
        """
        super(QNP_OR_decomposed, self)
        self.theta = theta

    def _num_qubits_(self):
        return 4

    def _decompose_(self, qubits):
        a, b, c, d = qubits
        yield cirq.H(a)
        yield cirq.H(b)
        yield cirq.CNOT(a, c)
        yield cirq.Ry(rads=self.theta/2).on(a)
        yield cirq.CNOT(b, d)
        yield cirq.Ry(rads=self.theta/2).on(b)
        yield cirq.Ry(rads=self.theta/2).on(c)
        yield cirq.Ry(rads=self.theta/2).on(d)
        yield cirq.CNOT(a, c)
        yield cirq.H(a)
        yield cirq.CNOT(b, d)
        yield cirq.H(b)

    def _circuit_diagram_info_(self, _):
        return [f"QNP_OR({self.theta:.2f})"] * self.num_qubits()


class OFSWAP_gate(cirq.Gate):

    def __init__(self):
        super(OFSWAP_gate, self)

    def _num_qubits_(self):
        return 4

    def _decompose_(self, qubits):
        a, b, c, d = qubits

        yield cirq.SWAP(a, b)
        yield cirq.CZ(a, b)
        yield cirq.SWAP(c, d)
        yield cirq.CZ(c, d)

    def _circuit_diagram_info_(self, _):
        return ["OFSWAP"] * self.num_qubits()


def test_gates():
    """
    Just a check that the gates work properly
    """
    import numpy as np

    theta_angle = 0.123
    qnppx = QNP_PX(theta_angle)
    qnpor = QNP_OR(theta_angle)
    fswap = OFSWAP_gate()

    qreg = cirq.LineQubit.range(4)
    q1, q2, q3, q4 = qreg[0], qreg[1], qreg[2], qreg[3]

    circ_local = cirq.Circuit(
        qnppx(q1, q2, q3, q4), qnpor(q1, q2, q3, q4), fswap.on(q1, q2, q3, q4)
    )

    simulator = cirq.Simulator()
    result = simulator.simulate(program=circ_local)

    state_vector = result.final_state_vector
    print()
    print("Check if the state vector is real:", np.allclose(state_vector, state_vector.conj()))
    print("Populated kets in state vector:")
    for dec_ket in np.where(np.abs(state_vector) > 1e-4)[0]:
        print(f"{state_vector[dec_ket].real:+.4f}", f"|{np.binary_repr(int(dec_ket), 4)}>")


if __name__ == "__main__":
    test_gates()
