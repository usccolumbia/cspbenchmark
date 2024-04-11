import numpy as np
from ase import Atoms

from delta import delta as delta_cy
from delta_py import delta as delta_py

import timeit

cov_dist = 1
deltaFunc_cy = delta_cy(cov_dist=cov_dist)
deltaFunc_py = delta_py(cov_dist=cov_dist)

Natoms = 4
dim = 3
L = 2
d = 1
pbc = [0,0,0]
atomtypes = ['H', 'H', 'H', 'H']
cell = [L,L,d]

def new_structure():
    x = np.array([0,0,0,
                          0,1,0,
                          1,0,0,
                          1,1,0]) + 0.1*np.random.rand(Natoms * dim)
    positions = x.reshape((-1,dim))
    a = Atoms(atomtypes, positions=positions, cell=cell, pbc=pbc)
    return a

a = new_structure()

def func_cy():
    return deltaFunc_cy.energy(a)

def func_py():
    return deltaFunc_py.energy(a)

py = timeit.timeit('func_py()', setup="from __main__ import func_py", number=10000)
cy = timeit.timeit('func_cy()', setup="from __main__ import func_cy", number=10000)

print(py/10000)
print(cy/10000)

print('E_py=', deltaFunc_py.energy(a))
print('E_cy=', deltaFunc_cy.energy(a))

print('F_py=', deltaFunc_py.forces(a))
print('F_cy=', deltaFunc_cy.forces(a))
