from agox.generators.ABC_generator import GeneratorBaseClass
from agox.observer import Observer
import numpy as np

class ReuseGenerator(GeneratorBaseClass):

    name = 'ReuseGenerator'

    def __init__(self, order=8, gets={'candidates_key':'prioritized_candidates', 'evaluated_key':'evaluated_candidates'}, **kwargs):
        """
        Acts as an observer to reuse previously generated candidates. 
        Make sure the order is set correctly (such as just after or just before the database).
        """
        super().__init__(order=order, gets=gets, **kwargs)
        self.prioritized_candidates = []
        self.evaluated_candidates = []

        self.add_observer_method(self.get_candidates_from_cache, sets={}, gets=self.gets[0], order=self.order[0])

    def get_candidates(self, sample, environment):
        candidate = self.get_candidate_for_reuse()
        self.writer('ReuseCandidate', candidate)

        if candidate is None:
            return [None]

        description = self.name + candidate.get_meta_information('description')
        candidate.add_meta_information('description', description)

        return [candidate]

    def get_candidates_from_cache(self):        
        """
        Access the shared cache at the end of an iteration in order to get the most recently generated & prioritized 
        candidates and the most recently evaluated candidates. 
        """
        self.prioritized_candidates = self.get_from_cache(self.candidates_key)
        self.evaluated_candidates = self.get_from_cache(self.evaluated_key)

    def get_candidate_for_reuse(self):
        """
        Get the best candidate that was not selected for evaluation in the previous iteration. 

        This is not strictly neccesary if using an evaluator that pops candiadtes, but cannot guarantee that 
        and this not (too) slow anyway (currently). [Shouldnt scale poorly though].
        """
        if len(self.prioritized_candidates) == 0:
            return None
        for i, candidate in enumerate(self.prioritized_candidates):
            not_evaluated = True
            for eval_candidate in self.evaluated_candidates:
                if (candidate.positions == eval_candidate.positions).all():
                    self.prioritized_candidates.pop(i) # Remove the candidate.
                    not_evaluated = False
            if not_evaluated:
                return candidate
        return None