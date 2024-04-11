import os
import numpy as np
from importlib_resources import files
from agox.candidates import StandardCandidate
from ase.io import read, write
import pickle
import pytest
import glob

test_folder_path = os.path.join(files('agox'), 'test/')

test_data_dicts = [
    {'path':'datasets/AgO-dataset.traj', 'remove':6, 'name':'AgO'}, 
    {'path':'datasets/B12-dataset.traj', 'remove':12, 'name':'B12'},        
    {'path':'datasets/C30-dataset.traj', 'remove':30, 'name':'C30'},
    ]

for dictionary in test_data_dicts:
    dictionary['path'] = os.path.join(test_folder_path, dictionary['path'])

class TemporaryFolder:

    def __init__(self, path):
        d = path / ""
        if not os.path.exists(d):
            d.mkdir()
        self.start_dir = os.getcwd()
        self.path = d

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, *args):
        os.chdir(self.start_dir)

def compare_candidates(atoms_1, atoms_2, tolerance):

    if atoms_1 is None and atoms_2 is None:
        return True

    np.testing.assert_allclose(atoms_1.positions, atoms_2.positions, **tolerance)
    np.testing.assert_allclose(atoms_1.cell, atoms_2.cell, **tolerance)
    np.testing.assert_allclose(atoms_1.numbers, atoms_2.numbers, **tolerance)
    return True

def get_test_environment(path, remove):
    from agox.environments import Environment

    atoms = read(path, index=0)
    cell = atoms.get_cell()
    corner = np.array([0, 0, 0])
    numbers = atoms.get_atomic_numbers()[len(atoms)-remove:]

    template = atoms.copy()
    del template[len(template)-remove:len(template)]
    environment = Environment(template=template, numbers=numbers, confinement_cell=cell, 
            confinement_corner=corner)

    return environment

def get_test_data(path, environment):
    template = environment.get_template()
    atoms_data = read(path, index=':')
    candidate_data = [StandardCandidate.from_atoms(template, atoms) for atoms in atoms_data]
    return candidate_data

def get_test_sampler(data):
    from agox.samplers import SamplerBaseClass

    class DummySampler(SamplerBaseClass):
        
        name = 'DummySampler'

        def __init__(self, data):
            self.sample = data

        def setup(self, *args, **kwargs):
            return None

    return DummySampler(data)

def get_name(module_name, subfolder, dataset_name, parameter_index):
    folder  = 'expected_outputs/'
    name = f'{test_folder_path}{subfolder}{folder}{module_name}_data{dataset_name}_parameter{parameter_index}.pckl'
    return name

def save_expected_data(name, data):
    check_file_is_deleted(name)
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def load_expected_data(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def label_dict_list(list_of_dicts):
    for index, dictionary in enumerate(list_of_dicts):
        list_of_dicts[index] = (dictionary, index)
    return list_of_dicts

@pytest.fixture(params=test_data_dicts)
def environment_and_dataset(request):
    from agox.environments import Environment
    from agox.candidates import StandardCandidate
    
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

def check_folder_is_empty(path):

    if not os.path.exists(path):
        os.mkdir(path)

    files = glob.glob(str(path) + '/*')
    assert len(files) == 0, 'Expected path is not empty, delete the files if you are sure you want to remake them!'

def check_file_is_deleted(filepath):
    assert not os.path.exists(filepath), 'File already exists - manually delete it if youre sure you want to recreate it.'
