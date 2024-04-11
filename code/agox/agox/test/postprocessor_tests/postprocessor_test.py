import pytest
from agox.test.test_utils import environment_and_dataset
from agox.postprocessors import CenteringPostProcess, WrapperPostprocess, RelaxPostprocess, ParallelRelaxPostprocess, ParallelRemoteRelaxPostprocess
from agox.models import ModelGPR
import numpy as np

def base_setup(environment, dataset):
    return {}

def make_model(environment, dataset):
    model = ModelGPR.default(environment, database=None)
    energies = np.array([atoms.get_potential_energy() for atoms in dataset])
    model.train_model(dataset, energies)
    return {'model':model}

@pytest.mark.parametrize('postprocess_class, setup_kwargs, setup_func', [
    (CenteringPostProcess, {}, base_setup),
    (WrapperPostprocess, {}, base_setup),
    (RelaxPostprocess, {'optimizer_run_kwargs':{'steps':10}}, make_model),
    (ParallelRelaxPostprocess, {'optimizer_run_kwargs':{'steps':10}, 'cpu_count':4}, make_model),
    (ParallelRemoteRelaxPostprocess, {'optimizer_run_kwargs':{'steps':10}, 'cpu_count':4}, make_model)

])
def test_postprocess(postprocess_class, setup_kwargs, setup_func, environment_and_dataset):
    environment, dataset = environment_and_dataset
    postprocessor = postprocess_class(**setup_kwargs, **setup_func(environment, dataset))

    try:
        postprocessor.postprocess(dataset[0])
    except NotImplementedError:
        postprocessor.process_list(dataset[0:8])





