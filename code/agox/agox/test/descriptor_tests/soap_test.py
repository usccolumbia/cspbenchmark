import pytest
import numpy as np
from agox.models.descriptors import SOAP
from agox.test.test_utils import test_data_dicts, get_test_environment, get_test_data, save_expected_data, load_expected_data, test_folder_path

@pytest.mark.parametrize('test_data_dict', test_data_dicts)
def test_descriptor(test_data_dict, cmd_options):

    create_mode = cmd_options['create_mode']
    tolerance = cmd_options['tolerance']

    path = test_data_dict['path']
    remove = test_data_dict['remove']
    name = test_data_dict['name']
    environment = get_test_environment(path, remove)
    data = get_test_data(path, environment)

    species = set(data[0].get_chemical_symbols())
    descriptor = SOAP(species=species)

    F = np.array(descriptor.get_local_features(data))

    name = f'{test_folder_path}/descriptor_tests/expected_outputs/{descriptor.name}_{name}.pckl'
    
    if create_mode:
        save_expected_data(name, F)
    else:
        F_exp = load_expected_data(name)
        np.testing.assert_allclose(F, F_exp, **tolerance)




