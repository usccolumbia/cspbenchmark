from agox.generators.ABC_generator import GeneratorBaseClass
from ase.data import covalent_radii
from ase.geometry import get_distances
from ase.io import write
import numpy as np


class ReplaceGenerator(GeneratorBaseClass):
    name = 'ReplaceGenerator'    
    def __init__(self, acquisitor=None, n_replace=5, amplitude=3, **kwargs):
        super().__init__(**kwargs)
        self.n_replace = n_replace
        self.amplitude = amplitude

    def get_candidates(self, sampler, environment):

        candidate = sampler.get_random_member()
        if candidate is None:
            return [None]
        
        indices_to_move = self.get_indices_to_move(candidate)
        
        for index in list(indices_to_move):
            for _ in range(100):
                radius = self.get_radius(candidate)
                displacement = self.get_displacement_vector(radius)
                new_position = self.get_new_position_center(candidate, index, indices_to_move) + displacement

                # Check confinement limits:
                if not self.check_confinement(new_position).all():
                    continue

                if self.check_new_position(candidate, new_position, candidate[index].number, skipped_indices=indices_to_move):
                    candidate[index].position = new_position
                    indices_to_move.remove(index)
                    break


        candidate = self.convert_to_candidate_object(candidate, candidate.get_template())
        candidate.add_meta_information('description', self.name)
        return [candidate]

    def get_indices_to_move(self, candidate):
        number_of_atoms = len(candidate)
        template = candidate.get_template()
        number_of_template_atoms = len(template)
        number_of_non_template_atoms = number_of_atoms - number_of_template_atoms

        probability = self.n_replace/number_of_non_template_atoms
        randoms = np.random.rand(number_of_non_template_atoms)
        indices_to_move = np.arange(number_of_template_atoms,number_of_atoms)[randoms < probability]
        indices_to_move = np.random.permutation(indices_to_move)
        if len(indices_to_move) == 0:
            indices_to_move = [np.random.randint(number_of_template_atoms,number_of_atoms)]
        return list(indices_to_move)


    def get_new_position_center(self, candidate, index, indices_to_move):
        return candidate.positions[index]

    def get_radius(self, candidate):
        return self.amplitude * np.random.rand()**(1/self.dimensionality)