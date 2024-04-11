import ray
import numpy as np
from agox.collectors.ABC_collector import CollectorBaseClass
from agox.utils.ray_utils import RayPoolUser


def remote_generate(generator, sampler, environment):
    return generator(sampler, environment)

class ParallelCollector(CollectorBaseClass, RayPoolUser):

    name = 'PoolParallelCollector'

    def __init__(self, num_candidates=None, **kwargs):        
        # Get the kwargs that are relevant for RayPoolUser
        pool_user_kwargs = {key:kwargs.pop(key, None) for key in RayPoolUser.kwargs if kwargs.get(key, None) is not None}
        RayPoolUser.__init__(self, **pool_user_kwargs)
        # Give the rest to the CollectorBaseClass.
        CollectorBaseClass.__init__(self, **kwargs)        
        self.num_candidates = num_candidates

        self.generator_keys = []
        for generator in self.generators:
            key = self.pool_add_module(generator)
            self.generator_keys.append(key)
        
    def make_candidates(self):
        # We need to build the args and kwargs sent to the actors.
        number_of_candidates = self.get_number_of_candidates()
        
        # This specifies which module each actor of the pool will use. 
        modules = []
        for generator_key, number in zip(self.generator_keys, number_of_candidates):
            modules += [[generator_key]] * number

        sampler_id = ray.put(self.sampler)
        environment_id = ray.put(self.environment)
        # The args and kwargs passed to the function - in this case the remote_generate 
        # function defined above. 
        args = [[sampler_id, environment_id]] * np.sum(number_of_candidates)
        #kwargs = [{}] * np.sum(number_of_candidates)
        kwargs = [{} for _ in range(np.sum(number_of_candidates))]

        # Generate in parallel using the pool.
        candidates = self.pool_map(remote_generate, modules, args, kwargs)

        # Flatten the output which is a list of lists. 
        flat_candidates = []
        for cand_list in candidates:
            for cand in cand_list:
                if cand is not None:
                    flat_candidates.append(cand.copy())
        return flat_candidates

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


