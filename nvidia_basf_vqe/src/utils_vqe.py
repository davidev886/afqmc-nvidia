import numpy as np
from pyscf import ao2mo
from functools import reduce
from openfermion import FermionOperator


def get_active_space(n_electrons, n_active_orbitals=None, n_active_electrons=None):
    r"""
    Computes the orbitals that should be considered doubly occupied and
    the orbital that should be considered active.
    Args:
        n_electrons (int): number of total electrons
        n_active_orbitals (int):  number of spatial orbitals desired in the active space.
        n_active_electrons (int): number of electrons desired in the active space.

    Return:
        occupied_indices(list): A list of spatial orbital indices indicating
            which orbitals should be considered doubly occupied.
            active_indices(list): A list of spatial orbital indices indicating
            which orbitals should be considered active.
   """

    if n_active_electrons in (0, None):
        n_core_orbitals = 0
        occupied_indices = None
    else:
        n_core_orbitals = (n_electrons - n_active_electrons) // 2
        occupied_indices = list(range(n_core_orbitals))

    if n_active_orbitals in (0, None):
        active_indices = None
    else:
        active_indices = list(
            range(n_core_orbitals, n_core_orbitals + n_active_orbitals))
    return occupied_indices, active_indices


def compute_integrals(pyscf_molecule, pyscf_scf):
    r"""
    Compute the 1-electron and 2-electron integrals.

    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.

    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    # Get one electrons integrals.
    n_orbitals = pyscf_scf.mo_coeff.shape[1]
    one_electron_compressed = reduce(np.dot, (pyscf_scf.mo_coeff.T,
                                              pyscf_scf.get_hcore(),
                                              pyscf_scf.mo_coeff))

    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_compressed = ao2mo.kernel(pyscf_molecule,
                                           pyscf_scf.mo_coeff)

    two_electron_integrals = ao2mo.restore(
        1,  # no permutation symmetry
        two_electron_compressed, n_orbitals)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = np.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')

    # Return.
    return one_electron_integrals, two_electron_integrals


def fermionic_zero_op():
    r"""
    Returns the fermionic zero operator
    """
    return FermionOperator()


def fermionic_id():
    r"""
    Returns the fermionic operator identity
    """
    return FermionOperator('')


def creat(j):
    r"""
    Returns a creation operator for the j-th fermionic mode
    """
    return FermionOperator(((j, 1),), 1.)


def destr(j):
    r"""
    Returns a annihilation operator for the j-th fermionic mode
    """
    return FermionOperator(((j, 0),), 1.)


def number_operator(n_spin_orbitals):
    r"""
    Defines the total number operator:

    \hat{N} = \sum_j^{n_so} c^dag_j c_j
    """
    num_ops = []
    for j in range(n_spin_orbitals):
        num_ops.append(creat(j) * destr(j))

    tot_num_operator = sum(num_ops, fermionic_zero_op())
    return tot_num_operator


def spin_z_operator(n_spin_orbitals):
    r"""
    Defines the spin_z operator for the convention (alpha beta alpha beta)

    \hat{N} = \hat{n_alpha} - \hat{n_beta} = \sum_j^{n_so} (-1)**j c^dag_j c_j
    """
    spin_z_ops = []
    for j in range(n_spin_orbitals):
        spin_z_ops.append((-1) ** j * creat(j) * destr(j))

    spinz_operator = sum(spin_z_ops, fermionic_zero_op())
    return spinz_operator


def fix_nelec_in_state_vector(final_state_vector, nelec):
    """
    Projects the wave function final_state_vector in the subspace with the fix number of electrons given by nelec
    :param final_state_vector Cirq object representing the state vector from a VQE simulation
    :param nelec (tuple) with n_alpha, n_beta number of electrons
    return: state vector (correctly normalized) with fixed number of electrons
    """
    n_alpha, n_beta = nelec
    n_qubits = int(np.log2(len(final_state_vector)))
    projected_vector = np.array(final_state_vector)
    for decimal_ket, coeff in enumerate(final_state_vector):
        string_ket = bin(decimal_ket)[2:].zfill(n_qubits)
        string_alpha = string_ket[::2]  # alpha orbitals occupy the even positions
        string_beta = string_ket[1::2]  # beta orbitals occupy the odd positions
        alpha_occ = [pos for pos, char in enumerate(string_alpha) if char == '1']
        beta_occ = [pos for pos, char in enumerate(string_beta) if char == '1']
        if (len(alpha_occ) != n_alpha) or (len(beta_occ) != n_beta):
            projected_vector[decimal_ket] = 0.0

    normalization = np.sqrt(np.dot(projected_vector.conj(), projected_vector))
    return projected_vector / normalization
