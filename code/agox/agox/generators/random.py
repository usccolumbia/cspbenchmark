from agox.generators.ABC_generator import GeneratorBaseClass
from ase.data import covalent_radii
from ase import Atom
import numpy as np

from scipy.spatial.distance import cdist

class RandomGenerator(GeneratorBaseClass):

    name = 'RandomGenerator'

    def __init__(self, may_nucleate_at_several_places=False, **kwargs):
        super().__init__(**kwargs)
        self.may_nucleate_at_several_places = may_nucleate_at_several_places        

    def get_candidates(self, sampler, environment):
        template = environment.get_template()
        numbers_list = environment.get_numbers()
        len_of_template = len(template)

        candidate = template.copy()

        while len(numbers_list) > 0:
            np.random.shuffle(numbers_list)
            atomic_number = numbers_list[0]
            numbers_list = numbers_list[1:]

            placing_first_atom = (len(candidate) == len_of_template)

            for _ in range(100):
                if placing_first_atom or self.may_nucleate_at_several_places: # If the template is completely empty. 
                    suggested_position = self.get_box_vector()
                else: # Pick only among atoms placed by the generator. 
                    placed_atom = candidate[np.random.randint(len_of_template,len(candidate))]
                    suggested_position = placed_atom.position.copy()
                    # Get a vector at an appropriate radius from the picked atom. 
                    vec = self.get_sphere_vector(atomic_number, placed_atom.number) 
                    suggested_position += vec

                if not self.check_confinement(suggested_position).all():
                    build_succesful = False
                    continue
                
                # Check that suggested_position is not too close/far to/from other atoms
                if self.check_new_position(candidate, suggested_position, atomic_number) or len(candidate) == 0:
                    build_succesful = True
                    candidate.extend(Atom(atomic_number, suggested_position))
                    break
                else:
                    build_succesful = False

            if not build_succesful:
                self.writer('Start generator failing at producing valid structure')
                return [None]
        
        candidate = self.convert_to_candidate_object(candidate, template)
        candidate.add_meta_information('description', self.name)

        return [candidate]