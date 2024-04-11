import pytest 
from agox.models.descriptors import SpectralGraphDescriptor
from ase import Atoms
import numpy as np
from ase.data import covalent_radii

descriptor = SpectralGraphDescriptor()

d = covalent_radii[1] * 2

positions = np.array([
    [0, 0, 0],
    [d, 0, 0],
    [0, d, 0],
    [d, d, 0]
])

atoms = Atoms('H4', positions)

D = descriptor.get_distances(atoms)

# All atoms have two perfect neighbours and one at d * sqrt(2)
assert (np.sort(D, axis=1) == np.array([0, 1, 1, np.sqrt(2)])).all()

A_compare = np.array([
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1]
])

A = descriptor.get_adjacency_matrix(atoms)

assert (A == A_compare).all()

L = descriptor.get_laplacian_matrix(atoms)

L_compare = np.eye(4) * 3 - A_compare

assert (L == L_compare).all()

feature = descriptor.get_global_features(atoms)[0]

assert np.allclose(feature, np.array([-1., 1., 1., 3.]))







