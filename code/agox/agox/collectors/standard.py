from agox.collectors.ABC_collector import CollectorBaseClass
from timeit import default_timer as dt

class StandardCollector(CollectorBaseClass):

    name = 'StandardCollector'

    def __init__(self, num_candidates, **kwargs):
        super().__init__(**kwargs)
        """
        The StandardCollector allows for either constant or iteration dependent 
        control of the number of candidates produced by each given generator. 

        Parameters
        ----------
        num_candidates : list or dict
            If a list it have the same length as the generators list containing 
            integers dictating how many candidates are produced by each 
            generator in the same order as generators are given. E.g. 
                num_candidates = [10, 5, 5]
            If a dict it must have integer keys that and 0 must be a key. The 
            values of the dict are lists like a list is passed, the keys dictate 
            at which episode the corresponding list is used, e.g
                num_candidates = {0:[10, 5, 5], 10:[10, 10, 10]}
        """
        self.num_candidates = num_candidates

    def get_number_of_candidates(self):
        if type(self.num_candidates) == list:
            return self.num_candidates
        elif type(self.num_candidates) == dict:
            return self.get_number_of_candidates_for_iteration()
            
    def get_number_of_candidates_for_iteration(self):
        # self.num_candidates must have this form: {0: [], 500: []}
        keys = list(self.num_candidates.keys())
        keys.sort()
        iteration = self.get_iteration_counter()
        if iteration is None:
            iteration = 0

        num_candidates = self.num_candidates[0] # yes, it must contain 0
        # now step through the sorted list (as long as the iteration is past the key) and extract the num_candidates
        # the last one extracted will be the most recent num_candidates enforced and should apply to this iteration
        for k in keys:
            if iteration < k:
                break
            num_candidates = self.num_candidates[k]
        return num_candidates

    def make_candidates(self):

        all_candidates = []
        for generator, num_candidates in zip(self.generators, self.get_number_of_candidates()):
            for sample in range(num_candidates):
                candidates = generator(sampler=self.sampler, environment=self.environment)
                    
                for candidate in candidates:
                    all_candidates.append(candidate)
        
        # The generator may have returned None if it was unable to build a candidate.
        all_candidates = list(filter(None, all_candidates))
                
        return all_candidates
            

