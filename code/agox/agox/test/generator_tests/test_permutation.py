import pytest
import numpy as np
from agox.generators import PermutationGenerator
from agox.test.generator_tests.generator_utils import generator_testing
from agox.test.test_utils import test_data_dicts

seed = 1
generator_args = []
generator_base_kwargs = {'c1':0.75, 'c2':1.25}
generator_class = PermutationGenerator

list_of_other_kwargs = [
    {}, # Tests that defaults havent changed. 
    {'max_number_of_swaps':1, 'rattle_strength':0.},
    {'max_number_of_swaps':3, 'rattle_strength':0.5},
    ]

# Because the PermutationGenerator only works for multicompoennt systems we filter abit:
from ase.io import read, write
multicomp_test_data_dicts = []
for test_data_dict in test_data_dicts:

    atoms = read(test_data_dict['path'])
    numbers = atoms.get_atomic_numbers()[len(atoms)-test_data_dict['remove']:]

    if len(np.unique(numbers)) >= 2:
        multicomp_test_data_dicts.append(test_data_dict)

for index, dictionary in enumerate(list_of_other_kwargs):
    list_of_other_kwargs[index] = (dictionary, index)

@pytest.fixture(params=list_of_other_kwargs)
def other_kwargs(request):
    return request.param

@pytest.mark.parametrize('test_data_dict', multicomp_test_data_dicts)
def test_generator(test_data_dict, other_kwargs, cmd_options):
    parameter_index = other_kwargs[1]
    other_kwargs = other_kwargs[0]

    create_mode = cmd_options['create_mode']
    test_mode = not create_mode
    tolerance = cmd_options['tolerance']

    generator_testing(generator_class, test_data_dict, generator_args, 
        generator_base_kwargs, other_kwargs, parameter_index, tolerance, 
        seed=seed, test_mode=test_mode)
