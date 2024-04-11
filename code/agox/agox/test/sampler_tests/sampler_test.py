import pytest 
from agox.samplers import KMeansSampler, MetropolisSampler, GeneticSampler, DistanceComparator, SpectralGraphSampler, KernelSimSampler
from agox.test.test_utils import environment_and_dataset
from agox.models.descriptors.fingerprint import Fingerprint
from agox.models import ModelGPR

from agox.candidates.ABC_candidate import CandidateBaseClass
from agox.test.test_utils import TemporaryFolder

def base_setup(environment, dataset):

    def args_func(dataset):
        return []
    return {'args_func':args_func}

def kmeans_setup(environment, dataset):
    model = ModelGPR.default(environment, database=None)
    model.pretrain(dataset[0:10])
    descriptor = model.get_descriptor()
    def args_func(dataset):
        return []
    return {'model_calculator':model, 'descriptor':descriptor, 'args_func':args_func}

def genetic_setup(environment, dataset):
    descriptor = Fingerprint(dataset[0])
    comparator = DistanceComparator(descriptor, 0.05)
    def args_func(dataset):
        return [dataset]
    return {'comparator':comparator, 'args_func':args_func}

def spectral_graph_setup(environment, dataset):
    model = ModelGPR.default(environment, database=None)
    model.pretrain(dataset[0:10])
    def args_func(dataset):
        return []
    return {'model_calculator':model, 'args_func':args_func}

@pytest.mark.parametrize('sampler_class, setup_kwargs, setup_func', [
    [KMeansSampler, {}, kmeans_setup],
    [MetropolisSampler, {}, base_setup],
    [GeneticSampler, {}, genetic_setup],
    [SpectralGraphSampler, {}, spectral_graph_setup],
    [KernelSimSampler, {}, kmeans_setup]
])
def test_sampler(sampler_class, setup_kwargs, setup_func, environment_and_dataset, tmp_path):

    with TemporaryFolder(tmp_path):

        environment, dataset = environment_and_dataset

        additional_kwargs = setup_func(environment, dataset)
        args_func = additional_kwargs.pop('args_func')
        sampler = sampler_class(**setup_kwargs, **additional_kwargs)
        sampler.iteration_counter = 0

        candidate = sampler.get_random_member() # Empty sampler should return None.
        assert candidate is None

        sampler.setup(dataset, *args_func(dataset))
        member = sampler.get_random_member()
        all_members = sampler.get_all_members()
        member_calc = sampler.get_random_member_with_calculator()

        assert issubclass(member.__class__, CandidateBaseClass)
        assert isinstance(all_members, list)
        assert member_calc.get_potential_energy()
        assert len(sampler) != 0
        assert len(sampler) == len(all_members)



    


