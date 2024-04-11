from agox.generators.ABC_generator import GeneratorBaseClass
from ase.data import covalent_radii
from ase import Atom
import numpy as np

class SteepestDescentGenerator(GeneratorBaseClass):

    name = 'SteepestDescentGenerator'

    def __init__(self, use_xy_only=False, immunize=False, **kwargs):
        super().__init__(**kwargs)
        self.use_xy_only = use_xy_only
        self.immunize = immunize

    def get_candidates(self, sampler, environment):

        # Changed in-place because sampler returns a COPY!
        candidate = sampler.get_random_member_with_calculator()

        # If the Sampler is not able to give a structure with forces
        if candidate is None or candidate.calc is None or 'forces' not in candidate.calc.results:
            return [None]

        template = candidate.get_template()
        len_of_template = len(template)

        delta = np.random.uniform(0, 0.1)

        self.writer(self.name,': got a candidate with a force. will take step:',delta)

        candidate.positions[len_of_template:] += delta * candidate.get_forces()[len_of_template:]
        
        candidate = self.convert_to_candidate_object(candidate, template)

        if self.immunize:
            candidate.set_postprocess_immunity(True)
        candidate.add_meta_information('description', self.name)

        return [candidate]

