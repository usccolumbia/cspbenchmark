from agox.collectors.ABC_collector import CollectorBaseClass
import ray
from timeit import default_timer as dt
import numpy as np
from agox.utils.ray_utils import RayBaseClass, ray_kwarg_keys

@ray.remote
def make_candidate_par(generators, sampler, environment, index, seed):
    """

    Remote function to make a candidate given a list of generators

    Parameters
    ----------
    generators : list
        List of generators. 
    sampler : AOGX Sampler
        Sampler to give the generator.
    environment : AGOX Environment
        Environemnt to give the generator.
    index : int
        Index that decides which generator to use. 
    seed : int
        Integer to use to seed Numpy for this generation event. 
    Returns
    -------
    list
        List of generated candidates.
    """
    np.random.seed(seed)
    return generators[index](sampler, environment)

class ParallelCollector(CollectorBaseClass, RayBaseClass):

    name = 'ParallelCollector'

    def __init__(self,  num_candidates, update_generators=False, **kwargs):
        """
        Works exactly like the StandardCollector, with the exception that 
        the 'update_generators' setting has to be set to True if generators 
        ever update their parameters.         

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
        update_generators : bool, optional
            If True generators are are 'put' each episode, otherwise they wont 
            be updated and will run with the parameters set at the start of the 
            run.
        """
        ray_kwargs = {key:kwargs.pop(key, None) for key in ray_kwarg_keys}
        CollectorBaseClass.__init__(self, **kwargs)
        RayBaseClass.__init__(self, **ray_kwargs)
        self.num_candidates = num_candidates
        self.update_generators = update_generators
        self.ray_startup()
        self.environment_re=ray.put(self.environment)
        self.gen_ref = None

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
        generator_indices = []
        tot_num = 0
        #a quick loop to establich the table of wanted generators to pass them all
        for i, num_candidates in enumerate(self.get_number_of_candidates()):
             generator_indices.extend([i]*num_candidates)
             tot_num += num_candidates

        if self.gen_ref == None or self.update_generators:
            self.gen_ref = ray.put(self.generators)
        seeds = np.random.randint(low=0, high=10e6, size=tot_num)
        sampler_ref = ray.put(self.sampler)
        candidates = [make_candidate_par.remote(self.gen_ref, sampler_ref, 
            self.environment_re, generator_indices[idx], seeds[idx]) for idx in range(tot_num)]
        candidates = ray.get(candidates)

        # Flatten the list  get rid of None and copy to avoid immutable arrays all in one
        for candidate_list in candidates:
            for candidate in candidate_list:
                if candidate is not None:
                    all_candidates.append(candidate.copy())

        return all_candidates
