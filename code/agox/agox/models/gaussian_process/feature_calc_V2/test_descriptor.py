import numpy as np

from time import time

from ase.io import read

from prior_old import repulsive_prior as repulsive_prior_old
from prior import repulsive_prior

a = read('/home/mkb/DFTB/TiO_2layer/ref/Ti13O26_GM_done.traj', index='0')

Nrep = 2

prior_old = repulsive_prior_old()

E_old = prior_old.energy(a)
F_old = prior_old.forces(a)

t0=time()
for i in range(Nrep):
    prior_old.energy(a)
dt_old = (time()-t0)/Nrep

prior = repulsive_prior()

E = prior.energy(a)
F = prior.forces(a)

t0=time()
for i in range(Nrep):
    prior.energy(a)
dt = (time()-t0)/Nrep

print('dF =\n', F_old-F)

print('dE =', E_old-E)
print('E_old =', E_old)
print('E =', E)

print(f'runtime: (Nrep={Nrep})')
print(dt_old)
print(dt)
