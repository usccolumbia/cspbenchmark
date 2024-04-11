import numpy as np
from copy import copy
from agox.models.descriptors import DescriptorBaseClass

class SOAP(DescriptorBaseClass):

    feature_types = ['local', 'global', 'local_gradient']

    name = 'SOAP'

    def __init__(self, species, r_cut=4, nmax=3, lmax=2, sigma=1.0,
                 weight=True, periodic=True, dtype='float64', normalize=False, crossover=True, **kwargs):
        super().__init__(**kwargs)

        from dscribe.descriptors import SOAP as dscribeSOAP
        self.normalize = normalize
        self.feature_types = self.feature_types.copy()

        if periodic:
            self.feature_types.remove('local_gradient')
        
        if weight is True:
            weighting = {'function':'poly', 'r0':r_cut, 'm':2, 'c':1}
        elif weight is None:
            weighting = None
        elif weight is False:
            weighting = None
        else:
            weighting = weight
            

        self.soap = dscribeSOAP(
            species=species,
            periodic=periodic,
            rcut=r_cut,
            nmax=nmax,
            lmax=lmax,
            sigma=sigma,
            weighting=weighting,
            dtype=dtype,
            crossover=crossover,
            sparse=False)

        self.lenght = self.soap.get_number_of_features()
        print('SOAP length:', self.lenght)

    def create_local_features(self, atoms):
        """Returns soap descriptor for "atoms".
        Dimension of output is [n_centers, n_features]
        """
        return self.soap.create(atoms)
            
    def create_local_feature_gradient(self, atoms):
        """Returns derivative of soap descriptor for "atoms" with
        respect to atomic coordinates.
        Dimension of output is [n_centers, 3*n_atoms, n_features]
        """
        f_deriv = self.soap.derivatives(atoms, return_descriptor=False)
        n_centers, n_atoms, n_dim, n_features = f_deriv.shape
        return f_deriv.reshape(n_centers, n_dim*n_atoms, n_features)
    
    def create_global_features(self, atoms):
        features = self.soap.create(atoms)
        return np.sum(features, axis=0)
