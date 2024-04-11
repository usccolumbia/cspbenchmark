import numpy as np
from copy import copy

from agox.models.descriptors.ABC_descriptor import DescriptorBaseClass
from dscribe.descriptors import SOAP as dscribeSOAP



class SOAP(DescriptorBaseClass):

    def __init__(self, species, r_cut=4, nmax=3, lmax=2, sigma=1.0,
                 weight=True, periodic=True, dtype='float64', normalize=False, crossover=True):
        self.normalize = normalize
        self.periodic = periodic
        
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
            average='inner',
            sparse=False)

        self.lenght = self.soap.get_number_of_features()
        print('SOAP lenght:', self.lenght)

    def get_feature(self, atoms):
        """Returns soap descriptor for "atoms".
        Dimension of output is [n_centers, n_features]
        """
        return self.soap.create(atoms)
            

    def get_featureMat(self, atoms_list):
        return self.soap.create(atoms_list)
    

    def get_featureGradient(self, atoms):
        if self.periodic:
            d = 0.0001
            num_gradient = np.zeros((3*len(atoms), self.lenght))
            for n in range(len(atoms)):
                for i in range(3):
                    atoms.positions[n,i] += d
                    f = self.get_feature(atoms)
                    atoms.positions[n,i] -= 2*d
                    b = self.get_feature(atoms)
                    atoms.positions[n,i] += d

                    num_gradient[(n*3)+i] = (f-b)/(2*d)

            return num_gradient
        else:
            return self.soap.derivatives(atoms, return_descriptor=False).reshape(-1,self.lenght)



if __name__ == '__main__':
    from ase.io import read
    
    def dimensions():
        data = read('/home/roenne/papers/local-model/GOFEE/pyridine/md/10best.traj', index=':')
        soap = SOAP(['H','C','N'], periodic=False)
        print(soap.get_feature(data[0]).shape)
        print(soap.get_featureMat(data).shape)
        # num_f = soap.get_featureGradient(data[0]).shape
        # ana_f = soap.test(data[0]).shape
        # print(np.allclose(num_f, ana_f))


    dimensions()
