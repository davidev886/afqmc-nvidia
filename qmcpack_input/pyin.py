#!/usr/bin/env python

from pyscf.pbc import gto, scf
from pathlib import Path
from functools import reduce
import sys
import h5py


# Building the cell
cell = gto.Cell()
cell.atom='''
Ni    0.000000000         0.000000000         0.000000000
Ni    1.468240743         2.207373377         1.252030022
O    -0.000000226         0.881833166         1.878045033
O     2.936481713         3.532913588         0.626015011
'''
cell.basis = 'gth-dzvp-molopt-sr'
cell.pseudo = 'gth-pbe'
cell.a = '''
2.9364821911, 0.0000000000, 0.0000000000
1.4682404910, 4.8584537991, 0.0000000000
-1.4682411958, -0.4437070453, 2.5040600446'''
cell.unit = 'A'
cell.precision = 1e-8
cell.verbose = 4
cell.build()
cell.spin = 0


# UHF
chkptfile="./uhf_111_dzvp.chk" # absolute path
mf = cell.UHF().density_fit()
dm = mf.from_chk(chkptfile)
mf.chkfile = 'uhf_111_dzvp_extended.chk'
mf.kernel(dm)

print("-----------------------")
print(mf.spin_square())
print("-----------------------")
mf.analyze()


nk = [1, 1, 1]
kpts = cell.make_kpts(nk)

from afqmctools.utils.linalg import get_ortho_ao
hcore = mf.get_hcore()
fock = (hcore + mf.get_veff())
X, nmo_per_kpt = get_ortho_ao(cell, kpts, 1e-14)
with h5py.File(mf.chkfile, 'r+') as fh5:
  fh5['scf/hcore'] = hcore
  fh5['scf/fock'] = fock
  fh5['scf/orthoAORot'] = X
  fh5['scf/nmo_per_kpt'] = nmo_per_kpt
