import os
import sys
import numpy as np
from math import erf
from itertools import product
from scipy.spatial.distance import cdist

try:
    cwd = sys.argv[1]
except:
    cwd = os.getcwd()

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

        self.pbc = atoms.get_pbc()
        self.cell = atoms.get_cell()
        self.n_atoms = len(atoms)
        self.num = atoms.get_atomic_numbers()
        self.atomic_types = sorted(list(set(self.num)))
        self.atomic_count = {type:list(self.num).count(type) for type in self.atomic_types}
        self.volume = atoms.get_volume()
        self.dim = 3

        # parameters for the binning:
        self.m1 = self.nsigma*self.sigma1/self.binwidth1  # number of neighbour bins included.
        self.smearing_norm1 = erf(1/np.sqrt(2) * self.m1 * self.binwidth1/self.sigma1)  # Integral of the included part of the gauss
        self.Nbins1 = int(np.ceil(self.Rc1/self.binwidth1))

        self.m2 = self.nsigma*self.sigma2/self.binwidth2  # number of neighbour bins included.
        self.smearing_norm2 = erf(1/np.sqrt(2) * self.m2 * self.binwidth2/self.sigma2)  # Integral of the included part of the gauss
        self.binwidth2 = np.pi/Nbins2

        Nelements_2body = self.Nbins1
        Nelements_3body = self.Nbins2

        if use_angular:
            self.Nelements = Nelements_2body + Nelements_3body
        else:
            self.Nelements = Nelements_2body

    def get_feature(self, atoms):
        """
        """

        cell = self.cell
        n_atoms = self.n_atoms
        pos = atoms.get_positions().tolist()
        num = atoms.get_atomic_numbers()
        atomic_count = self.atomic_count

        
        return pos

    def __angle(self, vec1, vec2):
        """
        Returns angle with convention [0,pi]
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        arg = np.dot(vec1,vec2)/(norm1*norm2)
        # This is added to correct for numerical errors
        if arg < -1:
            arg = -1.
        elif arg > 1:
            arg = 1.
        return np.arccos(arg), arg

    def __f_cutoff(self, r, gamma, Rc):
        """
        Polinomial cutoff function in the, with the steepness determined by "gamma"
        gamma = 2 resembels the cosine cutoff function.
        For large gamma, the function goes towards a step function at Rc.
        """
        if not gamma == 0:
            return 1 + gamma*(r/Rc)**(gamma+1) - (gamma+1)*(r/Rc)**gamma
        else:
            return 1

    def __f_cutoff_grad(self, r, gamma, Rc):
        if not gamma == 0:
            return gamma*(gamma+1)/Rc * ((r/Rc)**gamma - (r/Rc)**(gamma-1))
        else:
            return 0

    def angle2_grad(self, RijVec, RikVec):
        Rij = np.linalg.norm(RijVec)
        Rik = np.linalg.norm(RikVec)

        a = RijVec/Rij - RikVec/Rik
        b = RijVec/Rij + RikVec/Rik
        A = np.linalg.norm(a)
        B = np.linalg.norm(b)
        D = A/B

        RijMat = np.dot(RijVec[:,np.newaxis], RijVec[:,np.newaxis].T)
        RikMat = np.dot(RikVec[:,np.newaxis], RikVec[:,np.newaxis].T)

        a_grad_j = -1/Rij**3 * RijMat + 1/Rij * np.identity(3)
        b_grad_j = a_grad_j

        a_sum_j = np.sum(a*a_grad_j, axis=1)
        b_sum_j = np.sum(b*b_grad_j, axis=1)

        grad_j = 2/(1+D**2) * (1/(A*B) * a_sum_j - A/(B**3) * b_sum_j)



        a_grad_k = 1/Rik**3 * RikMat - 1/Rik * np.identity(3)
        b_grad_k = -a_grad_k

        a_sum_k = np.sum(a*a_grad_k, axis=1)
        b_sum_k = np.sum(b*b_grad_k, axis=1)

        grad_k = 2/(1+D**2) * (1/(A*B) * a_sum_k - A/(B**3) * b_sum_k)


        a_grad_i = -(a_grad_j + a_grad_k)
        b_grad_i = -(b_grad_j + b_grad_k)

        a_sum_i = np.sum(a*a_grad_i, axis=1)
        b_sum_i = np.sum(b*b_grad_i, axis=1)

        grad_i = 2/(1+D**2) * (1/(A*B) * a_sum_i - A/(B**3) * b_sum_i)

        return grad_i, grad_j, grad_k
