from agox.generators.ABC_generator import GeneratorBaseClass
from ase.data import covalent_radii
import numpy as np

class RattleGenerator(GeneratorBaseClass):

    name = 'RattleGenerator'

    def __init__(self, acquisitor=None, n_rattle=3, rattle_amplitude=3, **kwargs):
        super().__init__(**kwargs)
        self.acquisitor = acquisitor
        self.n_rattle = n_rattle
        self.rattle_amplitude = rattle_amplitude
        
    def get_candidates(self, sampler, environment):
        candidate = sampler.get_random_member()

        # Happens if no candidates are part of the sample yet. 
        if candidate is None:
            return [None]
    
        template = candidate.get_template()
        indices_to_rattle = self.get_indices_to_rattle(candidate)

        for i in indices_to_rattle:
            for _ in range(100): # For atom i attempt to rattle up to 100 times.
                radius = self.rattle_amplitude * np.random.rand()**(1/self.dimensionality)
                displacement = self.get_displacement_vector(radius)
                suggested_position = candidate.positions[i] + displacement

                # Check confinement limits:
                if not self.check_confinement(suggested_position).all():
                    continue

                # Check that suggested_position is not too close/far to/from other atoms
                # Skips the atom it self. 
                if self.check_new_position(candidate, suggested_position, candidate[i].number, skipped_indices=[i]):
                    candidate[i].position = suggested_position 
                    break
        
        candidate = self.convert_to_candidate_object(candidate, template)
        candidate.add_meta_information('description', self.name)
        return [candidate]

    def get_indices_to_rattle(self, candidate):
        # Establish indices_to_rattle as the indices of the atoms that should be rattled
        template = candidate.get_template()
        n_template = len(template)
        n_total = len(candidate)
        n_non_template = n_total - n_template
        probability = self.n_rattle/n_non_template
        indices_to_rattle = np.arange(n_template,n_total)[np.random.rand(n_non_template)
                                         < probability]
        indices_to_rattle = np.random.permutation(indices_to_rattle)
        if len(indices_to_rattle) == 0:
            indices_to_rattle = [np.random.randint(n_template,n_total)]
        # now indices_to_rattle is ready to use
        return indices_to_rattle
