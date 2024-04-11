import pytest
import numpy as np
from agox.models.local_GPR.LSGPR_CUR import LSGPRModelCUR
from agox.models.descriptors.soap import SOAP

from agox.test.model_tests.model_utils import model_tester
from agox.test.test_utils import get_test_data, get_test_environment, load_expected_data, save_expected_data, label_dict_list
from agox.test.test_utils import test_data_dicts, get_name, check_file_is_deleted
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Weird kernel specification is because of accurcay in Sklearn that does such a conversion. 
kernel = C(1.)*RBF(length_scale=np.exp(np.log(20.)))
model_class = LSGPRModelCUR
model_maker = LSGPRModelCUR
model_base_args = []
model_base_kwargs = {}

model_update_kwargs = [
    {'kernel':kernel, 'database':None},
]

model_update_kwargs = label_dict_list(model_update_kwargs)

@pytest.fixture(params=model_update_kwargs)
def update_kwargs(request):
    return request.param

@pytest.mark.parametrize('test_data_dict', test_data_dicts)
def test_model(test_data_dict, update_kwargs, cmd_options):
    create_mode = cmd_options['create_mode']
    test_mode = not create_mode
    tolerance = cmd_options['tolerance']

    update_kwargs, parameter_index  = update_kwargs

    path = test_data_dict['path']
    remove = test_data_dict['remove']
    dataset_name = test_data_dict['name']
    parameter_index = 0

    # Gather environment and data:
    environment = get_test_environment(path, remove)
    data = get_test_data(path, environment)

    # Update the kwargs for the model. 
    descriptor = SOAP(environment.get_all_species(), periodic=environment.get_template().pbc.any())
    update_kwargs['descriptor'] = descriptor
    # Slightly complicated way of building input args & kwargs:
    if 'environment' in update_kwargs.keys():
        update_kwargs['environment'] = environment

    model_kwargs = model_base_kwargs.copy()
    model_kwargs.update(update_kwargs)

    # Where to load/save data. 
    subfolder = 'model_tests/'
    module_name = model_class.name
    name = get_name(module_name, subfolder, dataset_name, parameter_index)
    if test_mode:
        expected_data = load_expected_data(name) 
    else:
        expected_data = None

    output = model_tester(model_maker, model_base_args, model_kwargs, data, 
        test_mode=test_mode, expected_energies=expected_data, tolerance=tolerance)

    if not test_mode:
        check_file_is_deleted(name)
        save_expected_data(name, output)










