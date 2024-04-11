import numpy as np
from agox.test.test_utils import get_test_environment, get_test_data
from agox.test.test_utils import test_data_dicts, get_name

def model_tester(model_maker, model_args, model_kwargs, data, test_mode=True, expected_energies=None, 
    tolerance=None):
    seed = 42
    np.random.seed(seed)

    # Data split:
    training_data = data[0:len(data)//2]
    training_energies = np.array([atoms.get_potential_energy() for atoms in training_data])
    test_data = data[len(data)//2:]

    # Make model instance:
    model = model_maker(*model_args, **model_kwargs)

    # Train the model:
    model.train_model(training_data, energies=training_energies)

    # Test on the remaining data:
    E = np.zeros(len(test_data))
    for i, test_atoms in enumerate(test_data):
        test_atoms.calc = model
        E[i] = test_atoms.get_potential_energy()

    if test_mode:
        np.testing.assert_allclose(E, expected_energies, **tolerance)

        # # Model parameters:
        parameters = model.get_model_parameters()
        recreated_model = model_maker(*model_args, **model_kwargs)

        recreated_model.set_model_parameters(parameters)
        recreated_energies = np.array([recreated_model.predict_energy(atoms) for atoms in test_data])
        np.testing.assert_allclose(E, recreated_energies)

    return E






