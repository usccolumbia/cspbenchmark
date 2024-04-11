from agox.models.descriptors.ABC_descriptor import DescriptorBaseClass
import numpy as np
from scipy.spatial.distance import cdist
from ase.neighborlist import neighbor_list
import warnings


class BehlerParrinello(DescriptorBaseClass):
    DescriptorType = 'local'
    ParameterNames = ['eta', 'xi', 'rs', 'lambda', 'eta_ang', 'rcut']
    ImplementedProperties = ['specific_atoms', 'dimension', 'jacobian']
    DefaultParameters = {'eta':np.array([]), 'xi':np.array([]), 'rs':np.array([]),
                         'lambda':np.array([]), 'eta_ang':np.array([]), 'rcut':np.array([])}
    ID = 'BehlerParanello'

    def __init__(self, eta, rs, rc=5, xi=[], lamb=[], eta_ang=[]):
        """
        Class for calculation of Behler Paranello features.

        Inputs:
           -- eta: Exponent of radial functions
           -- rs: Offset of Gaussians. 
           -- rc: Cutoff radius
           -- xi: Polynomial exponents of angular functions
           -- lamb (da): Signs for angular functions
           -- eta_ang: Exponential exponents of angular functions.
        """        
        # Dimensions:
        if isinstance(eta, list):
            self.num_radial = len(eta)
        elif isinstance(rs, list):
            self.num_radial = len(rs)
        else:
            self.num_radial = 1
            
        self.num_angular = len(xi)

        self.dimension = self.num_radial+self.num_angular

        # Unpack and assign from dictionary:
        self.eta = np.array(eta)
        self.rs = np.array(rs)
        self.xi = np.array(xi)
        self.lamb = np.array(lamb)
        self.eta_ang = np.array(eta_ang)
        self.rc = rc
        
    def get_feature(self, atoms):
        """
        Calculates the Behler Paranello feature vectors for all atoms in a given structure.
        """
        # Matrix to save output:
        num_atoms = len(atoms)
        F = np.zeros((num_atoms, self.dimension))
        
        # Calculate distances:
        I, J, dists, D = neighbor_list('ijdD', atoms, self.rc)
        dists = dists[:, np.newaxis]
        
        # Radial functions given by:
        rc2 = self.rc**2
        for i, j, d in zip(I, J, dists):
            F[i, 0:self.num_radial] += np.exp(-self.eta*(d-self.rs)**2)*self.cutoff(d)
        
        for i in range(num_atoms):
            neigh_mask = I==i
            for Rij, rij, j in zip(D[neigh_mask], dists[neigh_mask].ravel(), J[neigh_mask]):
                for Rik, rik, k in zip(D[neigh_mask], dists[neigh_mask].ravel(), J[neigh_mask]):
                    #print(i, j, k)
                    #if j != k and k != i: Equal conditions?
                    if j < k:
                        Rjk = Rik-Rij
                        rjk = np.sqrt(Rjk@Rjk)
                        theta = (Rij@Rik)/(np.sqrt(Rij@Rij)*np.sqrt(Rik@Rik))
                        F[i, self.num_radial::] += ((1-self.lamb*theta)**self.xi
                        *np.exp(-self.eta_ang*(rij**2+rik**2+rjk**2)/rc2)
                        *self.cutoff(rij)*self.cutoff(rik)*self.cutoff(rjk))
        F[:, self.num_radial::] *= 2**(1-self.xi)

        return F

    def calculate_specifics(self, atoms, index):
        """
        Arguments:
        -- Atoms: Atoms object
        -- Index: List/np.array of indicies of the atoms to calculate features for. 
        """

        if type(index) == int or type(index)==np.int64:
            index = [index]
        
        F = np.zeros((len(index), self.dimension), dtype=float)
        
        num_atoms = len(atoms)
        for ii, idx in enumerate(index):
            allowed_atoms = [i for i in range(num_atoms) if i != idx]
            
            d = cdist(atoms.positions, atoms.positions[idx].reshape(1, -1))
            F[ii, 0:self.num_radial] = np.sum((np.exp(-self.eta*(d[allowed_atoms]-self.rs)**2
                                        /self.rc**2)*self.cutoff(d[allowed_atoms])), axis=0)

            for j in range(num_atoms):
                if j == idx:
                    continue
                Rij = atoms.positions[j, :]-atoms.positions[idx, :]
                rij = d[j]
                Fij = self.cutoff(rij)
                for k in range(j+1, num_atoms):
                    if k == idx:
                        continue
                    Rjk = atoms.positions[k, :]-atoms.positions[j, :]
                    Rik = atoms.positions[k, :]-atoms.positions[idx, :]
                    theta = (Rij@Rik)/(np.sqrt(Rij@Rij)*np.sqrt(Rik@Rik))
                    rjk = np.sqrt(Rjk@Rjk)
                    rik = d[k]
                    Fik = self.cutoff(rik); Fjk = self.cutoff(rjk)
                    F[ii, self.num_radial::] += ((1-self.lamb*theta)**self.xi
                    *np.exp(-self.eta_ang*(rij**2+rik**2+rjk**2)/self.rc**2)
                                                  *Fij*Fjk*Fik)

            F[:, self.num_radial::] *= 2**(1-self.xi)   
        return F

    def calculate_jacobian(self, atoms):
        """
        Calculate the Jacobian of the feature vector at the current positions
        """

        N = len(atoms)
        
        # Arrays for saving:
        J = np.zeros((self.dimension*N, 3*N))
        
        # Calculate distances:
        rij = cdist(atoms.positions, atoms.positions)
        X = atoms.positions

        for i in range(N):
            for j in range(N):
                if rij[i, j] < self.rc:
                    for x in range(3):
                        if i != j:
                            J[i*self.dimension:(i+1)*self.dimension, j*3+x] = (
                            np.exp(-self.eta*(rij[i, j]-self.rs)**2/self.rc**2)
                            * (-2*self.eta*(rij[i, j]-self.rs)/self.rc**2 * self.cutoff(rij[i, j])
                            + self.cutoff_derivative(rij[i, j])) * (X[i, x] - X[j, x])/rij[i, j])

                            J[i*self.dimension:(i+1)*self.dimension, i*3+x] = (
                            J[i*self.dimension:(i+1)*self.dimension, j*3+x])
            
        return J                                        
                            
    def cutoff(self, r):
        return (r<=self.rc)*0.5*(1+np.cos(np.pi*r/self.rc))

    def cutoff_derivative(self, r):
        return -0.5*np.pi/self.rc * np.sin(np.pi*r/self.rc)

class BehlerParanello(BehlerParrinello):
    def __init__(self, *args, **kwargs):
        warnings.warn('Use BehlerParrinello instead - this will be deleted')
        super().__init__(*args, **kwargs)
