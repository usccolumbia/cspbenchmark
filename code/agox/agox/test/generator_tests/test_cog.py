import pytest
import numpy as np
from agox.generators import CenterOfGeometryGenerator
from agox.test.generator_tests.generator_utils import generator_testing
from agox.test.test_utils import test_data_dicts

seed = 1
generator_args = []
generator_base_kwargs = {'c1':0.75, 'c2':1.25}
generator_class = CenterOfGeometryGenerator

list_of_other_kwargs = [
    {},
    {'selection_percentages':{'low':0.25, 'high':0.5}, 'extra_radius_amplitude':1, 'extra_radius_params':{'low':-0.5, 'high':3}},
    {'selection_percentages':{'low':0.25, 'high':0.75}, 'extra_radius_amplitude':2, 'extra_radius_params':{'low':-0.4, 'high':5}}
    ]

for index, dictionary in enumerate(list_of_other_kwargs):
    list_of_other_kwargs[index] = (dictionary, index)

@pytest.fixture(params=list_of_other_kwargs)
def other_kwargs(request):
    return request.param

@pytest.mark.parametrize('test_data_dict', test_data_dicts)
def test_generator(test_data_dict, other_kwargs, cmd_options):
    parameter_index = other_kwargs[1]
    other_kwargs = other_kwargs[0]

    create_mode = cmd_options['create_mode']
    test_mode = not create_mode
    tolerance = cmd_options['tolerance']

    generator_testing(generator_class, test_data_dict, generator_args, 
        generator_base_kwargs, other_kwargs, parameter_index, tolerance, 
        seed=seed, test_mode=test_mode)                    






