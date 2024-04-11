import pytest
from ase.io import read, write
from agox.environments import Environment
from agox.candidates import StandardCandidate
from ase import Atoms
import numpy as np

from agox.test.test_utils import test_data_dicts

@pytest.fixture(params=test_data_dicts)
def environment_and_dataset(request):
    atoms = read(request.param['path'])
    cell = atoms.get_cell()
    corner = np.array([0, 0, 0])
    remove = request.param['remove']
    numbers = atoms.get_atomic_numbers()[len(atoms)-remove:]

    template = read(request.param['path'])
    del template[len(template)-remove:len(template)]
    environment = Environment(template=template, numbers=numbers, confinement_cell=cell, 
            confinement_corner=corner)

    data = read(request.param['path'], ':')
    candidates = [StandardCandidate.from_atoms(template, a) for a in data]

    return environment, candidates

def test_environment(environment_and_dataset):
    environment, dataset = environment_and_dataset

    template = environment.get_template()
    assert template is not None
    environment.set_template(template)

    numbers = environment.get_numbers()
    assert isinstance(numbers, np.ndarray)
    environment.set_numbers(numbers)

    missing = environment.get_missing_types()
    assert isinstance(missing, np.ndarray)





