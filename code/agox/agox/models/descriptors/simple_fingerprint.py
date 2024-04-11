from agox.models.descriptors import DescriptorBaseClass
import numpy as np

def eval_gauss(r, x, width):
    return np.exp(-0.5*((x-r)/width)**2)

class SimpleFingerprint(DescriptorBaseClass):

    feature_types = ['local', 'global']
    name = 'SimpleFingerprint'

    def __init__(self, species, Nbins=30, width=0.2, r_cut=3, separate_center_species=True):
        self.species = species
        self.Nbins = Nbins
        self.width = width
        self.r_cut = r_cut
        self.separate_center_species = separate_center_species

        self.species_dict = {species[i]: i for i in range(len(species))}
        self.Nspecies = len(self.species)
        self.r_bins = np.linspace(0,self.r_cut, Nbins)

    def create_local_features(self, atoms):
        Natoms = len(atoms)
        symbols = atoms.get_chemical_symbols()

        distances = atoms.get_all_distances(mic=True)
        
        if self.separate_center_species:
            descriptor = np.zeros((Natoms, self.Nspecies, self.Nspecies, self.Nbins))
        else:
            descriptor = np.zeros((Natoms, self.Nspecies, self.Nbins))
        for i, (distances_i, sym_i) in enumerate(zip(distances, symbols)):
            for j, (r_ij, sym_j) in enumerate(zip(distances_i, symbols)):
                if i == j:
                    continue
                if r_ij > self.r_cut + 4*self.width:
                    continue 
                idx_sym_i = self.species_dict[sym_i]
                idx_sym_j = self.species_dict[sym_j]
                if self.separate_center_species:
                    descriptor[i, idx_sym_i, idx_sym_j] += 0.5*eval_gauss(r_ij, self.r_bins, self.width) / r_ij**2
                else:
                    descriptor[i, idx_sym_j] += 0.5*eval_gauss(r_ij, self.r_bins, self.width) / r_ij**2
        return descriptor.reshape(Natoms, -1)

    def create_global_features(self, atoms):
        return self.create_local_features(atoms).sum(axis=0)
