#!/bin/bash

#
#
# based on
# https://github.com/QMCPACK/qmcpack/tree/develop/examples/afqmc/01-neon_atom
# https://github.com/QMCPACK/qmcpack/tree/develop/examples/afqmc/07-diamond_2x2x2_supercell
#
#

# threshold cholesky
threshold=1e-5

# creates the pyscf mol object and stores Hamiltonian information for afqmc
python -u  pyin.py

# generate the input files for QMCPACK
python afqmctools/bin/pyscf_to_afqmc.py \
        -i 'uhf_111_dzvp.chk' \
        -o afqmc_${threshold}.h5 \
        -t ${threshold} \
        -v \
        -a \
        -q afqmc_${threshold}.xml

# run qmcpack
qmcpack_complex afqmc_1e-5.xml
