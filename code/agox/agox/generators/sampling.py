from agox.generators.ABC_generator import GeneratorBaseClass

class SamplingGenerator(GeneratorBaseClass):

    name = 'SamplingGenerator'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_candidates(self, sampler, environment):

        if len(sampler) == 0:
            return [None]
        
        candidates = sampler.get_all_members()

        return candidates

