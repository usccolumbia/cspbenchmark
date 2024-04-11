import numpy as np
from itertools import product
from ase.ga.utilities import closest_distances_generator
cimport numpy as np

from libc.math cimport *

from cymem.cymem cimport Pool
cimport cython

# Custom functions
ctypedef struct Point:
    double coord[3]

cdef Point subtract(Point p1, Point p2):
    cdef Point p
    p.coord[0] = p1.coord[0] - p2.coord[0]
    p.coord[1] = p1.coord[1] - p2.coord[1]
    p.coord[2] = p1.coord[2] - p2.coord[2]
    return p

cdef Point add(Point p1, Point p2):
    cdef Point p
    p.coord[0] = p1.coord[0] + p2.coord[0]
    p.coord[1] = p1.coord[1] + p2.coord[1]
    p.coord[2] = p1.coord[2] + p2.coord[2]
    return p

cdef double norm(Point p):
    return sqrt(p.coord[0]*p.coord[0] + p.coord[1]*p.coord[1] + p.coord[2]*p.coord[2])

cdef double euclidean(Point p1, Point p2):
    return norm(subtract(p1,p2))

def convert_atom_types(num):
    cdef int Natoms = len(num)
    cdef list atomic_types = sorted(list(set(num)))
    cdef int Ntypes = len(atomic_types)
    cdef list num_converted = [0]*Natoms
    cdef int i, j
    for i in range(Natoms):
        for j in range(Ntypes):
            if num[i] == atomic_types[j]:
                num_converted[i] = j
    return num_converted

class delta():

    def __init__(self, atoms, rcut=5, ratio_of_covalent_radii=0.7):
        self.rcut = rcut

        self.pbc = atoms.get_pbc()
        self.cell = atoms.get_cell()

        self.cell_displacements = self.__get_neighbour_cells_displacement(self.pbc, self.cell)
        self.Ncells = len(self.cell_displacements)

        num = atoms.get_atomic_numbers()
        atomic_types = sorted(list(set(num)))
        self.Ntypes = len(atomic_types)

        blmin_dict = closest_distances_generator(num, ratio_of_covalent_radii=ratio_of_covalent_radii)

        self.blmin = -np.ones((self.Ntypes, self.Ntypes))
        for i1, type1 in enumerate(atomic_types):
            for i2, type2 in enumerate(atomic_types):
                self.blmin[i1,i2] = blmin_dict[(type1, type2)]
        self.blmin = self.blmin.reshape(-1).tolist()

    def energy(self, a):

        # Memory allocation pool
        cdef Pool mem
        mem = Pool()

        cdef int Natoms = len(a)
        cdef double rcut = self.rcut

        cdef list x_np = a.get_positions().tolist()
        cdef Point *x
        x = <Point*>mem.alloc(Natoms, sizeof(Point))
        cdef int m
        for m in range(Natoms):
            x[m].coord[0] = x_np[m][0]
            x[m].coord[1] = x_np[m][1]
            x[m].coord[2] = x_np[m][2]

        # Get neighbourcells and convert to Point-struct
        cdef int Ncells = self.Ncells
        cdef list cell_displacements_old = self.cell_displacements
        cdef Point *cell_displacements
        cell_displacements = <Point*>mem.alloc(Ncells, sizeof(Point))
        for m in range(Ncells):
            cell_displacements[m].coord[0] = cell_displacements_old[m][0]
            cell_displacements[m].coord[1] = cell_displacements_old[m][1]
            cell_displacements[m].coord[2] = cell_displacements_old[m][2]

        # Convert 2body bondtype list into c-array
        cdef int Ntypes = self.Ntypes
        cdef list blmin_old = self.blmin
        cdef double *blmin
        blmin = <double*>mem.alloc(Ntypes*Ntypes, sizeof(double))
        for m in range(Ntypes*Ntypes):
            blmin[m] = blmin_old[m]

        # Get converted atom Ntypes
        cdef list num_converted_old = convert_atom_types(a.get_atomic_numbers())
        cdef int *num_converted
        num_converted = <int*>mem.alloc(Natoms, sizeof(int))
        for m in range(Natoms):
            num_converted[m] = num_converted_old[m]

        cdef double E=0, rmin, radd
        cdef int i, j
        for i in range(Natoms):
            xi = x[i]
            for cell_index in range(Ncells):
                displacement = cell_displacements[cell_index]
                for j in range(Natoms):
                    rmin = blmin[num_converted[i]*Ntypes + num_converted[j]]
                    radd = 1 - rmin
                    xj = add(x[j], displacement)
                    r = euclidean(xi, xj)
                    if r > 1e-5 and r < rcut:
                        r_scaled = r + radd
                        E += 1/pow(r_scaled,12)
        return E/2

    def forces(self, a):
        # Memory allocation pool
        cdef Pool mem
        mem = Pool()

        cdef int Natoms = len(a)
        cdef int dim = 3
        cdef double rcut = self.rcut

        cdef list x_np = a.get_positions().tolist()
        cdef Point *x
        x = <Point*>mem.alloc(Natoms, sizeof(Point))
        cdef int m
        for m in range(Natoms):
            x[m].coord[0] = x_np[m][0]
            x[m].coord[1] = x_np[m][1]
            x[m].coord[2] = x_np[m][2]

        # Get neighbourcells and convert to Point-struct
        cdef int Ncells = self.Ncells
        cdef list cell_displacements_old = self.cell_displacements
        cdef Point *cell_displacements
        cell_displacements = <Point*>mem.alloc(Ncells, sizeof(Point))
        for m in range(Ncells):
            cell_displacements[m].coord[0] = cell_displacements_old[m][0]
            cell_displacements[m].coord[1] = cell_displacements_old[m][1]
            cell_displacements[m].coord[2] = cell_displacements_old[m][2]

        # Convert 2body bondtype list into c-array
        cdef int Ntypes = self.Ntypes
        cdef list blmin_old = self.blmin
        cdef double *blmin
        blmin = <double*>mem.alloc(Ntypes*Ntypes, sizeof(double))
        for m in range(Ntypes*Ntypes):
            blmin[m] = blmin_old[m]

        # Get converted atom Ntypes
        cdef list num_converted_old = convert_atom_types(a.get_atomic_numbers())
        cdef int *num_converted
        num_converted = <int*>mem.alloc(Natoms, sizeof(int))
        for m in range(Natoms):
            num_converted[m] = num_converted_old[m]

        #cdef double rmin = 0.7 * self.cov_dist
        #cdef double radd = 1 - rmin

        # Initialize Force object
        cdef double *dE
        dE = <double*>mem.alloc(Natoms * dim, sizeof(double))

        cdef int i, j, k
        for i in range(Natoms):
            xi = x[i]
            for cell_index in range(Ncells):
                displacement = cell_displacements[cell_index]
                for j in range(Natoms):
                    rmin = blmin[num_converted[i]*Ntypes + num_converted[j]]
                    radd = 1 - rmin
                    xj = add(x[j], displacement)
                    r = euclidean(xi,xj)
                    if r > 1e-5 and r < rcut:
                        r_scaled = r + radd
                        rijVec = subtract(xi,xj)

                        for k in range(dim):
                            dE[dim*i + k] += -12*rijVec.coord[k] / (pow(r_scaled,13)*r)
                            dE[dim*j + k] += 12*rijVec.coord[k] / (pow(r_scaled,13)*r)

        dE_np = np.zeros((Natoms, dim))
        for i in range(Natoms):
            for k in range(dim):
                dE_np[i,k] = dE[dim*i + k]

        return - dE_np.reshape(-1)/2

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
