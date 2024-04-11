import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator

from agox.models import ModelGPR
from agox.postprocessors import (ParallelRelaxPostprocess,
                                 ParallelRemoteRelaxPostprocess,
                                 RelaxPostprocess)
from agox.test.test_utils import environment_and_dataset


def make_model(environment, dataset):
    model = ModelGPR.default(environment, database=None)
    energies = np.array([atoms.get_potential_energy() for atoms in dataset])
    model.train_model(dataset, energies)
    return model


@pytest.mark.parametrize('postprocess_class', [RelaxPostprocess, ParallelRelaxPostprocess, ParallelRemoteRelaxPostprocess])
def test_postprocess_results(postprocess_class, environment_and_dataset):
    environment, dataset = environment_and_dataset
    postprocessor = postprocess_class(model=make_model(environment, dataset),
                                      optimizer_run_kwargs={'steps': 1})

    try:
        candidates = [postprocessor.postprocess(dataset[0])]
    except NotImplementedError:
        candidates = postprocessor.process_list(dataset[0:8])

    assert len(candidates) > 0

    for candidate in candidates:
        assert isinstance(candidate.calc, SinglePointCalculator)
        assert candidate.get_potential_energy() != 0
