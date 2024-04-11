import numpy as np
from pathlib import Path
from agox.test.test_utils import compare_candidates
from agox.test.test_utils import TemporaryFolder, test_folder_path, check_folder_is_empty
from importlib import import_module
from agox.databases import Database

def agox_test_run(mode, tmp_path, cmd_options):
    expected_path = Path(f'{test_folder_path}/run_tests/expected_outputs/{mode}_test/')

    create_mode = cmd_options['create_mode']
    test_mode = not create_mode
    tolerance = cmd_options['tolerance']

    if create_mode:
        tmp_path = expected_path
        check_folder_is_empty(tmp_path)

    with TemporaryFolder(tmp_path):
        
        # This loads the database file from the script file. 
        # This means that the documentation can link to this run-file.
        #from agox.test.run_tests.script_rss import database
        database = import_module(f'agox.test.run_tests.script_{mode}').database

        if test_mode: 
            expected_database = Database(f'{expected_path}/db0.db')
            compare_runs(database, expected_database, tolerance)

def compare_runs(database, expected_database, tolerance):
    test_candidates = database.get_all_candidates()
    test_energies = database.get_all_energies()
    test_forces = np.array([atoms.get_forces(apply_constraint=False) for atoms in database.get_all_candidates()])

    # Saved database:
    expected_database.restore_to_memory()
    expected_candidates = expected_database.get_all_candidates()
    expected_energies = expected_database.get_all_energies()
    expected_forces = np.array([atoms.get_forces(apply_constraint=False) for atoms in expected_database.get_all_candidates()])

    for candidate, expected_candidate in zip(test_candidates, expected_candidates):
        assert compare_candidates(candidate, expected_candidate, tolerance), 'Candidates dont match.'
    assert len(expected_candidates) == len(test_candidates), 'Different numbers of candidates.'

    np.testing.assert_allclose(expected_energies, test_energies, **tolerance)
    np.testing.assert_allclose(expected_forces, test_forces, **tolerance)
