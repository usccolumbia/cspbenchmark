import numpy as np
from scipy.spatial.distance import euclidean
from itertools import product
from ase.ga.utilities import closest_distances_generator

class delta():

    def __init__(self, atoms, rcut=5, ratio_of_covalent_radii=0.7):
        self.rcut = rcut

        self.pbc = atoms.get_pbc()
        self.cell = atoms.get_cell()

        self.cell_displacements = self.__get_neighbour_cells_displacement(self.pbc, self.cell)
        self.Ncells = len(self.cell_displacements)

        num = atoms.get_atomic_numbers()
        self.blmin = closest_distances_generator(num, ratio_of_covalent_radii=ratio_of_covalent_radii)

    def energy(self, a):
        num = a.get_atomic_numbers()
        x = a.get_positions()
        E = 0
        for i, xi in enumerate(x):
            for cell_index in range(self.Ncells):
                displacement = self.cell_displacements[cell_index]
                for j, xj in enumerate(x):
                    key = (num[i], num[j])
                    rmin = self.blmin[key]
                    radd = 1 - rmin
                    r = euclidean(xi, xj + displacement)
                    if r > 1e-5 and r < self.rcut:
                        r_scaled = r + radd
                        E += 1/r_scaled**12
        # Devide by two decause every pair is counted twice
        return E/2

    def forces(self, a):
        num = a.get_atomic_numbers()
        x = a.get_positions()
        Natoms, dim = x.shape
        dE = np.zeros((Natoms, dim))
        for i, xi in enumerate(x):
            for cell_index in range(self.Ncells):
                displacement = self.cell_displacements[cell_index]
                for j, xj in enumerate(x):
                    key = (num[i], num[j])
                    rmin = self.blmin[key]
                    radd = 1 - rmin
                    xj_pbc = xj+displacement
                    r = euclidean(xi,xj_pbc)
                    if r > 1e-5 and r < self.rcut:
                        r_scaled = r + radd
                        rijVec = xi-xj_pbc

                        dE[i] += 12*rijVec*(-1 / (r_scaled**13*r))
                        dE[j] += -12*rijVec*(-1 / (r_scaled**13*r))

        # Devide by two decause every pair is counted twice
        return - dE.reshape(-1) / 2

    def __get_neighbour_cells_displacement(self, pbc, cell):
        # Calculate neighbour cells
        cell_vec_norms = np.linalg.norm(cell, axis=0)
        neighbours = []
        for i in range(3):
            if pbc[i]:
                ncellmax = int(np.ceil(abs(self.rcut/cell_vec_norms[i]))) + 1  # +1 because atoms can be outside unitcell.
                neighbours.append(range(-ncellmax,ncellmax+1))
            else:
                neighbours.append([0])

        neighbourcells_disp = []
        for x,y,z in product(*neighbours):
            xyz = (x,y,z)
            displacement = np.dot(xyz, cell)
            neighbourcells_disp.append(list(displacement))

        return neighbourcells_disp
