from agox.samplers.ABC_sampler import SamplerBaseClass
import numpy as np
from agox.writer import agox_writer
from agox.observer import Observer

class MetropolisSampler(SamplerBaseClass):

    name = 'MetropolisSampler'

    def __init__(self, temperature=1, gets={'get_key':'evaluated_candidates'},
        **kwargs):
        """
        Temperature: The selection temperature in eV. 
        """
        super().__init__(gets=gets, **kwargs)

        self.temperature = temperature

        self.reset_observer() # We remove observers added by the base-class. 
        self.add_observer_method(self.setup_sampler, gets=self.gets[0], sets=self.sets[0], order=self.order[0], handler_identifier='AGOX')

    def get_random_member(self):
        if len(self.sample) > 0:
            return self.sample[0].copy()
        else:
            return None

    @agox_writer
    @Observer.observer_method
    def setup_sampler(self, state):
        if self.do_check():            
            evaled_candidates = state.get_from_cache(self, self.get_key)
            evaled_candidates = list(filter(None, evaled_candidates))
            if len(evaled_candidates) > 0:
                best_candidate = evaled_candidates[np.argmin([atoms.get_potential_energy() for atoms in evaled_candidates])]
                self.setup(best_candidate)

    def setup(self, potential_step):
        if isinstance(potential_step, list):
            potential_step = potential_step[-1]

        # If the current sample is not empty
        if len(self.sample) > 0:
            chosen_candidate = self.sample[0]
        else: # If it is.
            chosen_candidate = None

        # If the there is no new candidate to consider we return. 
        if potential_step is None:
            return

        # If the current sample was empty we just take the new possiblility.
        if chosen_candidate is None:
            chosen_candidate = potential_step
            if chosen_candidate is not None:
                chosen_candidate.add_meta_information('accepted', True)
        else: # Otherwise we look at the probabilities.
            Eold = chosen_candidate.get_potential_energy()
            Enew = potential_step.get_potential_energy()
            if Enew < Eold: # Always accept if the step is lower in energy. 
                chosen_candidate = potential_step
                accepted = True
            else: # Calculate the probability and roll to see if we accept. 
                P = np.exp(-(Enew - Eold)/self.temperature)
                r = np.random.rand()
                if r < P:
                    accepted = True
                    chosen_candidate = potential_step
                else:
                    accepted = False
            # Update the acceptance.
            potential_step.add_meta_information('accepted', accepted)

        # Add to the sample.
        self.sample = [chosen_candidate]
        
        self.writer(f'Length of sample: {len(self.sample)}')