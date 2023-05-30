import numpy as np
from ipie.utils.io import write_json_input_file
from pyscf.scf.chkfile import dump_scf

from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

from src.utils_vqe import write_trial_ipie, write_hamiltonian_ipie
from src.vqe import VqeHardwareEfficient


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=10000)
    np.random.seed(12)
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    distance = 1.23
    geometry = [('H', (0, 0, 0)),
                ('H', (0, 0, distance)),
                ('H', (distance, 0, 0)),
                ('H', (distance, 0, distance))]

    basis = 'sto-3g'
    spin = 0

    molecule = MolecularData(geometry, basis, multiplicity, charge)

    molecule = run_pyscf(molecule, run_scf=True, run_fci=False)
    mf = molecule._pyscf_data['scf']
    mol = molecule._pyscf_data['mol']
    write_hamiltonian_ipie(mf, file_name="hamiltonian.h5")
    noccas, noccbs = mol.nelec
    # dump_scf(mol, "new.chk", mf.e_tot, mf.mo_energy, mf.mo_coeff, mf.mo_occ)

    print(f"SCF energy: {molecule.hf_energy}")

    print("starting VQE")
    n_qubit = molecule.n_qubits
    n_electron = molecule.n_electrons
    fermionic_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
    jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    vqe = VqeHardwareEfficient(n_qubits=n_qubit, n_layers=1)
    vqe.run(jw_hamiltonian,
            init_vals=np.random.rand(vqe.num_params),
            options={'maxiter': 200, 'callback': True}
            )
    print(f"VQE energy: {vqe.best_vqe_energy}")
    write_trial_ipie(vqe.final_state_vector_best, mol.nelec, file_name="wavefunction.h5")

    write_json_input_file(input_filename="ipie_input.json",
                          hamil_filename="hamiltonian.h5",
                          wfn_filename="wavefunction.h5",
                          nelec=mol.nelec)

    # run ipie with
    #       ipie ipie_input.json