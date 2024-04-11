import os
import sys
from math import erf
from itertools import product
from scipy.spatial.distance import cdist

import time

import numpy as np
cimport numpy as np

from libc.math cimport *

from cymem.cymem cimport Pool
cimport cython

try:
    cwd = sys.argv[1]
except:
    cwd = os.getcwd()

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

cdef double dot(Point v1, Point v2):
    return v1.coord[0]*v2.coord[0] + v1.coord[1]*v2.coord[1] + v1.coord[2]*v2.coord[2]

cdef double get_angle(Point v1, Point v2):
    """
    Returns angle with convention [0,pi]
    """
    norm1 = norm(v1)
    norm2 = norm(v2)
    arg = dot(v1,v2)/(norm1*norm2)
    # This is added to correct for numerical errors
    if arg < -1:
        arg = -1.
    elif arg > 1:
        arg = 1.
    return acos(arg)

@cython.cdivision(True)
cdef double f_cutoff(double r, double gamma, double Rcut):
    """
    Polinomial cutoff function in the, with the steepness determined by "gamma"
    gamma = 2 resembels the cosine cutoff function.
    For large gamma, the function goes towards a step function at Rc.
    """
    if not gamma == 0:
        return 1 + gamma*pow(r/Rcut, gamma+1) - (gamma+1)*pow(r/Rcut, gamma)
    else:
        return 1

@cython.cdivision(True)
cdef double f_cutoff_grad(double r, double gamma, double Rcut):
    if not gamma == 0:
        return gamma*(gamma+1)/Rcut * (pow(r/Rcut, gamma) - pow(r/Rcut, gamma-1))
    else:
        return 0

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

class Angular_Fingerprint(object):
    """ comparator for construction of angular fingerprints
    """
    def __init__(self, atoms, Rc1=4.0, Rc2=4.0, binwidth1=0.1, Nbins2=30, sigma1=0.2, sigma2=0.10, nsigma=4, eta=1, gamma=3, use_angular=True):
        """ Set a common cut of radius
        """

        self.Rc1 = Rc1
        self.Rc2 = Rc2
        self.binwidth1 = binwidth1
        self.Nbins2 = Nbins2
        self.binwidth2 = np.pi / Nbins2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.nsigma = nsigma
        self.eta = eta
        self.gamma = gamma
        self.use_angular = use_angular


        self.volume = atoms.get_volume()
        self.pbc = atoms.get_pbc()
        self.Natoms = len(atoms)
        self.dim = 3

        # parameters for the binning:
        self.m1 = self.nsigma*self.sigma1/self.binwidth1  # number of neighbour bins included.
        self.smearing_norm1 = erf(1/np.sqrt(2) * self.m1 * self.binwidth1/self.sigma1)  # Integral of the included part of the gauss
        self.Nbins1 = int(np.ceil(self.Rc1/self.binwidth1))

        self.m2 = self.nsigma*self.sigma2/self.binwidth2  # number of neighbour bins included.
        self.smearing_norm2 = erf(1/np.sqrt(2) * self.m2 * self.binwidth2/self.sigma2)  # Integral of the included part of the gauss
        self.binwidth2 = np.pi/Nbins2

        # Prepare stuff to handle multiple species
        self.num = atoms.get_atomic_numbers()
        self.atomic_types = sorted(list(set(self.num)))
        self.atomic_count = {type:list(self.num).count(type) for type in self.atomic_types}
        self.Ntypes = len(self.atomic_types)

        type_converter = {}
        for i, type in enumerate(self.atomic_types):
            type_converter[type] = i

        # Bondtypes - 2body
        self.Nbondtypes_2body = 0
        self.bondtypes_2body = -np.ones((self.Ntypes, self.Ntypes)).astype(int)
        count = 0
        for tt1 in self.atomic_types:
            for tt2 in self.atomic_types:
                type1, type2 = tuple(sorted([tt1, tt2]))
                t1 = type_converter[type1]
                t2 = type_converter[type2]
                if type1 < type2:
                    if self.bondtypes_2body[t1,t2] == -1:
                        self.bondtypes_2body[t1,t2] = count
                        self.Nbondtypes_2body += 1
                        count += 1
                elif type1 == type2 and (self.atomic_count[type1] > 1 or sum(self.pbc) > 0):
                    if self.bondtypes_2body[t1,t2] == -1:
                        self.bondtypes_2body[t1,t2] = count
                        self.Nbondtypes_2body += 1
                        count += 1
        self.bondtypes_2body = self.bondtypes_2body.reshape(-1).tolist()


        # Bondtypes - 3body
        self.bondtypes_3body = -np.ones((self.Ntypes, self.Ntypes, self.Ntypes)).astype(int)
        bondtypes_3body_keys = []
        for type1 in self.atomic_types:
            for type2 in self.atomic_types:
                if type1 == type2 and not (self.atomic_count[type1] > 1 or sum(self.pbc) > 0):
                    continue
                for type3 in self.atomic_types:
                    if type1 == type3 and not (self.atomic_count[type1] > 1 or sum(self.pbc) > 0):
                        continue
                    key = tuple([type1] + sorted([type2, type3]))
                    if type2 < type3:
                        bondtypes_3body_keys.append(key)
                    elif type2 == type3:
                        if type2 == type1 and (self.atomic_count[type1] > 2 or sum(self.pbc) > 0):
                            bondtypes_3body_keys.append(key)
                        elif type2 != type1 and (self.atomic_count[type2] > 1 or sum(self.pbc) > 0):
                            bondtypes_3body_keys.append(key)
        self.Nbondtypes_3body = len(bondtypes_3body_keys)
        for i, key in enumerate(bondtypes_3body_keys):
            k1, k2, k3 = key
            self.bondtypes_3body[type_converter[k1],type_converter[k2],type_converter[k3]] = i
        self.bondtypes_3body = self.bondtypes_3body.reshape(-1).tolist()

        self.Nelements_2body = self.Nbondtypes_2body * self.Nbins1
        self.Nelements_3body = self.Nbondtypes_3body * self.Nbins2

        if use_angular:
            self.Nelements = self.Nelements_2body + self.Nelements_3body
        else:
            self.Nelements = self.Nelements_2body

        # Get relevant neighbour unit-cells
        self.pbc = atoms.get_pbc()
        self.cell = atoms.get_cell()
        self.neighbourcells_disp = self.__get_neighbour_cells_displacement(self.pbc, self.cell)

    def get_feature(self, atoms):
        """
        """
        cdef double Rc1 = self.Rc1
        cdef double Rc2 = self.Rc2
        cdef double binwidth1 = self.binwidth1
        cdef double binwidth2 = self.binwidth2
        cdef int Nbins1 = self.Nbins1
        cdef int Nbins2 = self.Nbins2
        cdef double sigma1 = self.sigma1
        cdef double sigma2 = self.sigma2
        cdef int nsigma = self.nsigma

        cdef double eta = self.eta
        cdef double gamma = self.gamma
        cdef use_angular = self.use_angular

        cdef double volume = self.volume
        cdef int dim = self.dim

        cdef double m1 = self.m1
        cdef double m2 = self.m2
        cdef double smearing_norm1 = self.smearing_norm1
        cdef double smearing_norm2 = self.smearing_norm2

        cdef int Nelements_2body = self.Nelements_2body
        cdef int Nelements_3body = self.Nelements_3body
        cdef int Nelements = self.Nelements

        # Memory allocation pool
        cdef Pool mem
        mem = Pool()

        cell = atoms.get_cell()
        cdef int Natoms = self.Natoms

        # Get positions and convert to Point-struct
        cdef list pos_np = atoms.get_positions().tolist()
        cdef Point *pos
        pos = <Point*>mem.alloc(Natoms, sizeof(Point))
        cdef int m
        for m in range(Natoms):
            pos[m].coord[0] = pos_np[m][0]
            pos[m].coord[1] = pos_np[m][1]
            pos[m].coord[2] = pos_np[m][2]

        # Get neighbourcells and convert to Point-struct
        cdef int Ncells = len(self.neighbourcells_disp)
        cdef list cell_displacements_old = self.neighbourcells_disp
        cdef Point *cell_displacements
        cell_displacements = <Point*>mem.alloc(Ncells, sizeof(Point))
        for m in range(Ncells):
            cell_displacements[m].coord[0] = cell_displacements_old[m][0]
            cell_displacements[m].coord[1] = cell_displacements_old[m][1]
            cell_displacements[m].coord[2] = cell_displacements_old[m][2]

        # Convert 2body bondtype list into c-array
        cdef int Ntypes = self.Ntypes
        cdef int Nbondtypes_2body = self.Nbondtypes_2body
        cdef list bondtypes_2body_old = self.bondtypes_2body
        cdef int *bondtypes_2body
        bondtypes_2body = <int*>mem.alloc(Ntypes*Ntypes, sizeof(int))
        for m in range(Ntypes*Ntypes):
            bondtypes_2body[m] = bondtypes_2body_old[m]

        # Get converted atom Ntypes
        cdef list num_converted_old = convert_atom_types(atoms.get_atomic_numbers())
        cdef int *num_converted
        num_converted = <int*>mem.alloc(Natoms, sizeof(int))
        for m in range(Natoms):
            num_converted[m] = num_converted_old[m]

        # RADIAL FEATURE

        # Initialize radial feature
        cdef double *feature1
        feature1 = <double*>mem.alloc(Nelements_2body, sizeof(double))

        cdef int num_pairs, center_bin, minbin_lim, maxbin_lim, newbin, bondtype_index, type1, type2
        cdef double Rij, normalization, binpos, c, erfarg_low, erfarg_up, value
        cdef int i, j, n
        for i in range(Natoms):
            for cell_index in range(Ncells):
                displacement = cell_displacements[cell_index]
                for j in range(Natoms):
                    Rij = euclidean(pos[i], add(pos[j], displacement))

                    # Stop if distance too long or atoms are the same one.
                    if Rij > Rc1+nsigma*sigma1 or Rij < 1e-6:
                        continue

                    # determine bondtype
                    if num_converted[i] <= num_converted[j]:
                        type1 = num_converted[i]
                        type2 = num_converted[j]
                    else:
                        type1 = num_converted[j]
                        type2 = num_converted[i]
                    bondtype_index = Nbins1*bondtypes_2body[Ntypes*type1+type2]

                    # Calculate normalization
                    num_pairs = Natoms*Natoms
                    normalization = 1./smearing_norm1
                    normalization /= 4*M_PI*Rij*Rij * binwidth1 * num_pairs/volume

                    # Identify what bin 'Rij' belongs to + it's position in this bin
                    center_bin = <int> floor(Rij/binwidth1)
                    binpos = Rij/binwidth1 - center_bin

                    # Lower and upper range of bins affected by the current atomic distance deltaR.
                    minbin_lim = <int> -ceil(m1 - binpos)
                    maxbin_lim = <int> ceil(m1 - (1-binpos))

                    for n in range(minbin_lim, maxbin_lim + 1):
                        newbin = center_bin + n
                        if newbin < 0 or newbin >= Nbins1:
                            continue

                        # Calculate gauss contribution to current bin
                        c = 1./sqrt(2)*binwidth1/sigma1
                        erfarg_low = max(-m1, n-binpos)
                        erfarg_up = min(m1, n+(1-binpos))
                        value = 0.5*erf(c*erfarg_up)-0.5*erf(c*erfarg_low)

                        # Apply normalization
                        value *= normalization

                        feature1[bondtype_index + newbin] += value
                        #feature1[newbin] += value

        # Convert radial feature to numpy array
        feature1_np = np.zeros(Nelements_2body)
        for m in range(Nelements_2body):
            feature1_np[m] = feature1[m]

        # Return feature if only radial part is desired
        if not use_angular:
            return feature1_np

        # ANGULAR FEATURE

        # Convert 3body bondtype list into c-array
        cdef int Nbondtypes_3body = self.Nbondtypes_3body
        cdef list bondtypes_3body_old = self.bondtypes_3body
        cdef int *bondtypes_3body
        bondtypes_3body = <int*>mem.alloc(Ntypes*Ntypes*Ntypes, sizeof(int))
        for m in range(Ntypes*Ntypes*Ntypes):
            bondtypes_3body[m] = bondtypes_3body_old[m]

        # Initialize angular feature
        cdef double *feature2
        feature2 = <double*>mem.alloc(Nelements_3body, sizeof(double))

        cdef Point RijVec, RikVec
        cdef double angle
        cdef int k, cond_ij, cond_ik, type3
        for i in range(Natoms):
            pos_i = pos[i]
            for cell_index1 in range(Ncells):
                displacement1 = cell_displacements[cell_index1]
                for j in range(Natoms):
                    pos_j = add(pos[j], displacement1)
                    Rij = euclidean(pos[i], pos_j)
                    if Rij > Rc2 or Rij < 1e-6:
                        continue
                    for cell_index2 in range(cell_index1, Ncells):
                        displacement2 = cell_displacements[cell_index2]
                        if cell_index1 == cell_index2:
                            k_start = j+1
                        else:
                            k_start = 0
                        for k in range(k_start, Natoms):
                            pos_k = add(pos[k], displacement2)
                            Rik = euclidean(pos_i, pos_k)
                            if Rik > Rc2 or Rik < 1e-6:
                                continue

                            # determine bondtype
                            type1 = num_converted[i]
                            if num_converted[j] <= num_converted[k]:
                                type2 = num_converted[j]
                                type3 = num_converted[k]
                            else:
                                type2 = num_converted[k]
                                type3 = num_converted[j]
                            bondtype_index = Nbins2*bondtypes_3body[Ntypes*Ntypes*type1 + Ntypes*type2 + type3]

                            # Calculate angle
                            RijVec = subtract(pos_j,pos_i)
                            RikVec = subtract(pos_k, pos_i)
                            angle = get_angle(RijVec, RikVec)

                            # Calculate normalization
                            num_pairs = Natoms*Natoms*Natoms
                            normalization = 1./smearing_norm2
                            normalization /= num_pairs/volume

                            # Identify what bin 'Rij' belongs to + it's position in this bin
                            center_bin = <int> floor(angle/binwidth1)
                            binpos = angle/binwidth2 - center_bin

                            # Lower and upper range of bins affected by the current atomic distance deltaR.
                            minbin_lim = <int> -ceil(m2 - binpos)
                            maxbin_lim = <int> ceil(m2 - (1-binpos))

                            for n in range(minbin_lim, maxbin_lim + 1):
                                newbin = center_bin + n

                                # Wrap current bin into correct bin-range
                                if newbin < 0:
                                    newbin = abs(newbin)
                                if newbin > Nbins2-1:
                                    newbin = 2*Nbins2 - newbin - 1

                                # Calculate gauss contribution to current bin
                                c = 1./sqrt(2)*binwidth2/sigma2
                                erfarg_low = max(-m2, n-binpos)
                                erfarg_up = min(m2, n+(1-binpos))
                                value = 0.5*erf(c*erfarg_up)-0.5*erf(c*erfarg_low)
                                value *= f_cutoff(Rij, gamma, Rc2) * f_cutoff(Rik, gamma, Rc2)
                                # Apply normalization
                                value *= normalization

                                feature2[bondtype_index + newbin] += value
                                #feature2[newbin] += value

        # Convert angular feature to numpy array
        feature2_np = np.zeros(Nelements_3body)
        for m in range(Nelements_3body):
            feature2_np[m] = eta * feature2[m]

        feature_np = np.zeros(Nelements)
        feature_np[:Nelements_2body] = feature1_np
        feature_np[Nelements_2body:] = feature2_np
        return feature_np

    def get_featureMat(self, atoms_list):
        featureMat = np.array([self.get_feature(atoms) for atoms in atoms_list])
        featureMat = np.array(featureMat)
        return featureMat

    def get_featureGradient(self, atoms):

        cdef double Rc1 = self.Rc1
        cdef double Rc2 = self.Rc2
        cdef double binwidth1 = self.binwidth1
        cdef double binwidth2 = self.binwidth2
        cdef int Nbins1 = self.Nbins1
        cdef int Nbins2 = self.Nbins2
        cdef double sigma1 = self.sigma1
        cdef double sigma2 = self.sigma2
        cdef int nsigma = self.nsigma

        cdef double eta = self.eta
        cdef double gamma = self.gamma
        cdef use_angular = self.use_angular

        cdef double volume = self.volume
        cdef int dim = self.dim

        cdef double m1 = self.m1
        cdef double m2 = self.m2
        cdef double smearing_norm1 = self.smearing_norm1
        cdef double smearing_norm2 = self.smearing_norm2

        cdef int Nelements_2body = self.Nelements_2body
        cdef int Nelements_3body = self.Nelements_3body
        cdef int Nelements = self.Nelements

        # Memory allocation pool
        cdef Pool mem
        mem = Pool()

        cell = atoms.get_cell()
        cdef int Natoms = self.Natoms

        # Get positions and convert to Point-struct
        cdef list pos_np = atoms.get_positions().tolist()
        cdef Point *pos
        pos = <Point*>mem.alloc(Natoms, sizeof(Point))
        cdef int m
        for m in range(Natoms):
            pos[m].coord[0] = pos_np[m][0]
            pos[m].coord[1] = pos_np[m][1]
            pos[m].coord[2] = pos_np[m][2]

        # Get neighbourcells and convert to Point-struct
        cdef int Ncells = len(self.neighbourcells_disp)
        cdef list cell_displacements_old = self.neighbourcells_disp
        cdef Point *cell_displacements
        cell_displacements = <Point*>mem.alloc(Ncells, sizeof(Point))
        for m in range(Ncells):
            cell_displacements[m].coord[0] = cell_displacements_old[m][0]
            cell_displacements[m].coord[1] = cell_displacements_old[m][1]
            cell_displacements[m].coord[2] = cell_displacements_old[m][2]

        # Convert bondtype list into c-array
        cdef int Ntypes = self.Ntypes
        cdef int Nbondtypes_2body = self.Nbondtypes_2body
        cdef list bondtypes_2body_old = self.bondtypes_2body
        cdef int *bondtypes_2body
        bondtypes_2body = <int*>mem.alloc(Ntypes*Ntypes, sizeof(int))
        for m in range(Ntypes*Ntypes):
            bondtypes_2body[m] = bondtypes_2body_old[m]

        # Get converted atom Ntypes
        cdef list num_converted_old = convert_atom_types(atoms.get_atomic_numbers())
        cdef int *num_converted
        num_converted = <int*>mem.alloc(Natoms, sizeof(int))
        for m in range(Natoms):
            num_converted[m] = num_converted_old[m]

        # RADIAL FEATURE GRADIENT

        # Initialize radial feature-gradient
        cdef double *feature_grad1
        feature_grad1 = <double*>mem.alloc(Nelements_2body * Natoms * dim, sizeof(double))

        cdef Point RijVec
        cdef int num_pairs, center_bin, minbin_lim, maxbin_lim, newbin, bondtype_index, type1, type2
        cdef double Rij, normalization, binpos, c, arg_low, arg_up, value1, value2, value
        cdef int i, j, n
        for i in range(Natoms):
            pos_i = pos[i]
            for cell_index in range(Ncells):
                displacement = cell_displacements[cell_index]
                for j in range(Natoms):
                    pos_j = add(pos[j], displacement)
                    Rij = euclidean(pos_i, pos_j)

                    # Stop if distance too long or atoms are the same one.
                    if Rij > Rc1+nsigma*sigma1 or Rij < 1e-6:
                        continue
                    RijVec = subtract(pos_j,pos_i)

                    # determine bondtype
                    if num_converted[i] <= num_converted[j]:
                        type1 = num_converted[i]
                        type2 = num_converted[j]
                    else:
                        type1 = num_converted[j]
                        type2 = num_converted[i]
                    bondtype_index = Nbins1*bondtypes_2body[Ntypes*type1+type2]

                    # Calculate normalization
                    num_pairs = Natoms*Natoms
                    normalization = 1./smearing_norm1
                    normalization /= 4*M_PI*Rij*Rij * binwidth1 * num_pairs/volume

                    # Identify what bin 'Rij' belongs to + it's position in this bin
                    center_bin = <int> floor(Rij/binwidth1)
                    binpos = Rij/binwidth1 - center_bin

                    # Lower and upper range of bins affected by the current atomic distance deltaR.
                    minbin_lim = <int> -ceil(m1 - binpos)
                    maxbin_lim = <int> ceil(m1 - (1-binpos))

                    for n in range(minbin_lim, maxbin_lim + 1):
                        newbin = center_bin + n
                        if newbin < 0 or newbin >= Nbins1:
                            continue

                        # Calculate gauss contribution to current bin
                        c = 1./sqrt(2)*binwidth1/sigma1
                        arg_low = max(-m1, n-binpos)
                        arg_up = min(m1, n+(1-binpos))
                        value1 = -1./Rij * (erf(c*arg_up) - erf(c*arg_low))
                        value2 = -1./(sigma1*sqrt(2*M_PI)) * (exp(-pow(c*arg_up,2)) - exp(-pow(c*arg_low,2)))
                        value = value1 + value2

                        # Apply normalization
                        value *= normalization

                        # Add to the the gradient matrix
                        for m in range(3):
                            feature_grad1[(bondtype_index + newbin) * Natoms*dim + dim*i+m] += -value/Rij * RijVec.coord[m]
                            feature_grad1[(bondtype_index + newbin) * Natoms*dim + dim*j+m] += value/Rij * RijVec.coord[m]


        # Convert radial feature to numpy array
        cdef int grad_index
        feature_grad1_np = np.zeros((Natoms*dim, Nelements_2body))
        for m in range(Nelements_2body):
            for grad_index in range(Natoms*dim):
                feature_grad1_np[grad_index][m] = feature_grad1[m * Natoms*dim + grad_index]

        # Return feature if only radial part is desired
        if not use_angular:
            return feature_grad1_np

        # ANGULAR FEATURE-GRADIENT

        # Convert 3body bondtype list into c-array
        cdef int Nbondtypes_3body = self.Nbondtypes_3body
        cdef list bondtypes_3body_old = self.bondtypes_3body
        cdef int *bondtypes_3body
        bondtypes_3body = <int*>mem.alloc(Ntypes*Ntypes*Ntypes, sizeof(int))
        for m in range(Ntypes*Ntypes*Ntypes):
            bondtypes_3body[m] = bondtypes_3body_old[m]

        # Initialize angular feature-gradient
        cdef double *feature_grad2
        feature_grad2 = <double*>mem.alloc(Nelements_3body * Natoms * dim, sizeof(double))

        cdef Point RikVec, angle_grad_i, angle_grad_j, angle_grad_k
        cdef double angle, cos_angle, a
        cdef int k, cond_ij, cond_ik, bin_index, type3
        for i in range(Natoms):
            pos_i = pos[i]
            for cell_index1 in range(Ncells):
                displacement1 = cell_displacements[cell_index1]
                for j in range(Natoms):
                    pos_j = add(pos[j], displacement1)
                    Rij = euclidean(pos[i], pos_j)
                    if Rij > Rc2 or Rij < 1e-6:
                        continue
                    for cell_index2 in range(cell_index1, Ncells):
                        displacement2 = cell_displacements[cell_index2]
                        if cell_index1 == cell_index2:
                            k_start = j+1
                        else:
                            k_start = 0
                        for k in range(k_start, Natoms):
                            pos_k = add(pos[k], displacement2)
                            Rik = euclidean(pos_i, pos_k)
                            if Rik > Rc2 or Rik < 1e-6:
                                continue

                            # determine bondtype
                            type1 = num_converted[i]
                            if num_converted[j] <= num_converted[k]:
                                type2 = num_converted[j]
                                type3 = num_converted[k]
                            else:
                                type2 = num_converted[k]
                                type3 = num_converted[j]
                            bondtype_index = Nbins2*bondtypes_3body[Ntypes*Ntypes*type1 + Ntypes*type2 + type3]

                            # Calculate angle
                            RijVec = subtract(pos_j,pos_i)
                            RikVec = subtract(pos_k, pos_i)


                            angle = get_angle(RijVec, RikVec)
                            cos_angle = cos(angle)

                            for m in range(3):
                                if not (angle == 0 or angle == M_PI):
                                    a = -1/sqrt(1 - cos_angle*cos_angle)
                                    angle_grad_j.coord[m] = a * (RikVec.coord[m]/(Rij*Rik) - cos_angle*RijVec.coord[m]/(Rij*Rij))
                                    angle_grad_k.coord[m] = a * (RijVec.coord[m]/(Rij*Rik) - cos_angle*RikVec.coord[m]/(Rik*Rik))
                                    angle_grad_i.coord[m] = -(angle_grad_j.coord[m] + angle_grad_k.coord[m])
                                else:
                                    angle_grad_j.coord[m] = 0
                                    angle_grad_k.coord[m] = 0
                                    angle_grad_i.coord[m] = 0

                            fc_ij = f_cutoff(Rij, gamma, Rc2)
                            fc_ik = f_cutoff(Rik, gamma, Rc2)
                            fc_grad_ij = f_cutoff_grad(Rij, gamma, Rc2)
                            fc_grad_ik = f_cutoff_grad(Rik, gamma, Rc2)

                            # Calculate normalization
                            num_pairs = Natoms*Natoms*Natoms
                            normalization = 1./smearing_norm2
                            normalization /= num_pairs/volume

                            # Identify what bin 'Rij' belongs to + it's position in this bin
                            center_bin = <int> floor(angle/binwidth1)
                            binpos = angle/binwidth2 - center_bin

                            # Lower and upper range of bins affected by the current atomic distance deltaR.
                            minbin_lim = <int> -ceil(m2 - binpos)
                            maxbin_lim = <int> ceil(m2 - (1-binpos))

                            for n in range(minbin_lim, maxbin_lim + 1):
                                newbin = center_bin + n

                                # Wrap current bin into correct bin-range
                                if newbin < 0:
                                    newbin = abs(newbin)
                                if newbin > Nbins2-1:
                                    newbin = 2*Nbins2 - newbin - 1

                                # Calculate gauss contribution to current bin
                                c = 1./sqrt(2)*binwidth2/sigma2
                                arg_low = max(-m2, n-binpos)
                                arg_up = min(m2, n+(1-binpos))
                                value1 = 0.5*erf(c*arg_up)-0.5*erf(c*arg_low)
                                value2 = -1./(sigma2*sqrt(2*M_PI)) * (exp(-pow(c*arg_up, 2)) - exp(-pow(c*arg_low, 2)))

                                # Apply normalization
                                value1 *= normalization
                                value2 *= normalization

                                bin_index = (bondtype_index + newbin) * Natoms*dim
                                for m in range(3):
                                    feature_grad2[bin_index + dim*i+m] += -value1 * fc_ik*fc_grad_ij * RijVec.coord[m]/Rij
                                    feature_grad2[bin_index + dim*j+m] += value1 * fc_ik*fc_grad_ij * RijVec.coord[m]/Rij

                                    feature_grad2[bin_index + dim*i+m] += -value1 * fc_ij*fc_grad_ik * RikVec.coord[m]/Rik
                                    feature_grad2[bin_index + dim*k+m] += value1 * fc_ij*fc_grad_ik * RikVec.coord[m]/Rik

                                    feature_grad2[bin_index + dim*i+m] += value2 * fc_ij * fc_ik * angle_grad_i.coord[m]
                                    feature_grad2[bin_index + dim*j+m] += value2 * fc_ij * fc_ik * angle_grad_j.coord[m]
                                    feature_grad2[bin_index + dim*k+m] += value2 * fc_ij * fc_ik * angle_grad_k.coord[m]


        feature_grad2_np = np.zeros((Natoms*dim, Nelements_3body))
        for m in range(Nelements_3body):
            for grad_index in range(Natoms*dim):
                feature_grad2_np[grad_index][m] = eta * feature_grad2[m * Natoms*dim + grad_index]

        feature_grad_np = np.zeros((Natoms*dim, Nelements))
        feature_grad_np[:, :Nelements_2body] = feature_grad1_np
        feature_grad_np[:, Nelements_2body:] = feature_grad2_np
        return feature_grad_np

    def get_all_featureGradients(self, atoms_list):
        feature_grads = np.array([self.get_featureGradient(atoms) for atoms in atoms_list])
        feature_grads = np.array(feature_grads)
        return feature_grads

    def __get_neighbour_cells_displacement(self, pbc, cell):
        # Calculate neighbour cells
        Rc_max = max(self.Rc1+self.sigma1*self.nsigma, self.Rc2)  # relevant cutoff
        cell_vec_norms = np.linalg.norm(cell, axis=0)
        neighbours = []
        for i in range(3):
            if pbc[i]:
                ncellmax = int(np.ceil(abs(Rc_max/cell_vec_norms[i]))) + 1  # +1 because atoms can be outside unitcell.
                neighbours.append(range(-ncellmax,ncellmax+1))
            else:
                neighbours.append([0])

        neighbourcells_disp = []
        for x,y,z in product(*neighbours):
            xyz = (x,y,z)
            displacement = np.dot(xyz, cell)
            neighbourcells_disp.append(list(displacement))

        return neighbourcells_disp
