# AFQMC

* In the folder `qmcpack_input` the script [script_pyscf_to_qmcpack.sh](qmcpack_input%2Fscript_pyscf_to_qmcpack.sh)
contains the basic steps necessary to generate AFQMC input from a pyscf scf calculation
and to run qmcpack.


* In the folder `vqe` the script [generate_input_ipie.py](vqe%2Fgenerate_input_ipie.py)
generates a vqe trial wavefunction, the Hamiltonian and basic input file for running [ipie](https://github.com/linusjoonho/ipie).


* In the folder `nvidia_basf_vqe` the script [qnp_vqe_main.py](nvidia_basf_vqe%2Fqnp_vqe_main.py) implements a vqe computation of a LiH molecule.
 
