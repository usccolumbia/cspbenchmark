import numpy as np
import pytest 
from agox.environments import Environment
from agox.samplers import MetropolisSampler
from ase import Atoms
from agox.generators import RandomGenerator, RattleGenerator, CenterOfGeometryGenerator, ReplaceGenerator, ReuseGenerator, SamplingGenerator, PermutationGenerator
from agox.candidates import CandidateBaseClass

from agox.test.test_utils import environment_and_dataset

@pytest.mark.parametrize('generator_class', [RandomGenerator, RattleGenerator, ReplaceGenerator, CenterOfGeometryGenerator, SamplingGenerator])
class TestGenerator:

    def assertions(self, candidates, environment, sampler):
        for candidate in candidates:
            assert issubclass(candidate.__class__, CandidateBaseClass)
            assert len(candidate) == len(environment.get_all_numbers())
            assert (candidate.cell == environment.get_template().get_cell()).all()

    def setup_generator(self, generator_class, environment, **kwargs):
        return generator_class(**environment.get_confinement(), **kwargs)

    def setup_sampler(self, dataset):
        sampler = MetropolisSampler()
        sampler.sample = [dataset[0]]
        return sampler

    def test_generators(self, generator_class, environment_and_dataset):
        environment, dataset = environment_and_dataset
        
        generator = self.setup_generator(generator_class, environment)
        sampler = self.setup_sampler(dataset)
        candidates = [None]
        for i in range(1):
            candidates = generator(sampler, environment)
            if not candidates[0] == None:
                break
        self.assertions(candidates, environment, sampler)