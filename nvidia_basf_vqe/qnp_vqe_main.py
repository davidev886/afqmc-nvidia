import numpy as np
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

from src.vqe_ansatz import VqeQNP

if __name__ == "__main__":
    run_fci = False  # very costly if the molecule is big
    n_layers = 2
    basis = 'sto-3g'
    np.random.seed(12)
    spin = 0
    multiplicity = spin + 1
    charge = 0

    geometry = [('Li', (0.0000000, 0.0000000, 0.000000)),
                ('H', (0.0000000, 0.0000000, 1.600000))]

    molecule = MolecularData(geometry, basis, multiplicity, charge)

    molecule = run_pyscf(molecule, run_scf=True, run_fci=run_fci)
    mf = molecule._pyscf_data['scf']
    mol = molecule._pyscf_data['mol']
    noccas, noccbs = mol.nelec
    mol.unit = 'AU'

    print(f"# SCF energy: {molecule.hf_energy}")
    if run_fci:
        print(f"# FCI energy: {molecule.fci_energy}")
        fci_energy = molecule.fci_energy
    n_qubits = molecule.n_qubits
    n_electron = molecule.n_electrons
    fermionic_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
    jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    print(f"# Starting VQE with {n_qubits} qubits and {n_layers} layers")

    vqe = VqeQNP(n_qubits=n_qubits,
                 n_layers=n_layers,
                 hamiltonian=jw_hamiltonian,
                 init_mo_occ=mf.mo_occ)

    vqe.run(init_vals=np.random.rand(vqe.num_parameters),
            options={'maxiter': 100, 'callback': True}
            )

    print(f"# VQE energy: {vqe.best_vqe_energy}")
    print(f"# VQE params: {vqe.best_vqe_params}")
