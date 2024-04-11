import numpy as np
import matplotlib.pyplot as plt

from angular_fingerprintFeature import Angular_Fingerprint
from featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint as Angular_Fingerprint_cy
from custom_calculators import doubleLJ_calculator

from ase import Atoms
from ase.visualize import view
from ase.io import read, write

from time import time

dim = 3

L = 1
d = 1
pbc = [0,0,0]

"""
N = 2
x = np.array([0.7*L, 0.9*L, d/2,
              0.7*L, 0.5*L, d/2])
positions = x.reshape((-1, dim))
atomtypes = ['He', 'He']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)

N = 3
x = np.array([1.5*L, 0.2*L, d/2,
              0.5*L, 0.9*L, d/2,
              -0.5*L, 0.5*L, d/2,])
positions = x.reshape((-1,dim))
atomtypes = ['He', 'H', 'H']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)

x = np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 1.5, 1, 0])
positions = x.reshape((-1,dim))
a = Atoms('H4',
          positions=positions,
          cell=[4,2,1],
          pbc=[0, 0, 0])

N = 4
x = np.array([0.2*L, 0.7*L, d/2,
              0.3*L, 0.2*L, d/2,
              0.7*L, 0.9*L, d/2,
              0.7*L, 0.5*L, d/2])
positions = x.reshape((-1,dim))
atomtypes = ['H', 'H', 'He', 'He']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)

N = 5
x = np.array([0.2*L, 0.7*L, d/2,
              0.3*L, 0.2*L, d/2,
              0.7*L, 0.9*L, d/2,
              0.7*L, 0.5*L, d/2,
              0.9*L, 0.1*L, d/2])
positions = x.reshape((-1,dim))
atomtypes = ['H', 'He', 'O', 'H', 'H']
a = Atoms(atomtypes,
          positions=positions,
          cell=[L,L,d],
          pbc=pbc)
"""

atoms = read('graphene_all2.traj', index=':')
a = atoms[100]
atomtypes = a.get_atomic_numbers()
N = len(a.get_atomic_numbers())
x = a.get_positions().reshape(-1)




"""
calc = doubleLJ_calculator()
a.set_calculator(calc)
print('1:', a.get_potential_energy())
print(a.get_positions())
print('2:', a.get_potential_energy())
print(a.get_positions())
#a.set_positions(a.get_scaled_positions())
a.wrap()
print('3:', a.get_potential_energy())
print(a.get_positions())
"""
#view(a)

from delta import delta as delta_cy
from delta_py import delta as delta_py
from ase.data import covalent_radii
from ase.ga.utilities import closest_distances_generator

dcy = delta_cy(atoms=a)
dpy = delta_py(atoms=a)

print(dcy.energy(a))
print(dpy.energy(a))

print(dcy.forces(a))
print(dpy.forces(a))

"""
print('pbc check:')
print('before wrap')
print(dpy.energy(a))
a.wrap()
print('after wrap')
print(dpy.energy(a))
print(dcy.energy(a))


print('pbc check:')
print('before wrap')
print(dpy.forces(a))
a.wrap()
print('after wrap')
print(dpy.forces(a))
print(dcy.forces(a))
"""



