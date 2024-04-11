import pytest
import numpy as np
import os
from agox.databases import Database
from agox.test.test_utils import environment_and_dataset

def test_new_database(tmp_path, environment_and_dataset):

    # Make and move to temp dir:
    start_dir = os.getcwd()
    d = tmp_path / 'database_test'
    os.mkdir(d)
    os.chdir(d)

    database_created = Database('db1.db')
    environment, dataset = environment_and_dataset

    meta_data_names = ['str', 'int', 'float', '1darr', '2darr']
    meta_data_examples = ['str', 0, 0.1, np.array([0, 1, 2]), np.random.rand(3, 3)]

    for candidate in dataset:
        for meta_data_name, meta_data in zip(meta_data_names, meta_data_examples):
            candidate.add_meta_information(meta_data_name, meta_data)
        database_created.store_candidate(candidate)

    assert len(database_created) == len(dataset)

    for db_cand, data_cand in zip(database_created.get_all_candidates(), dataset):
        assert (db_cand.positions == data_cand.positions).all()
        assert (db_cand.cell == data_cand.cell).all()
        assert (db_cand.pbc == data_cand.pbc).all()
        for meta_data_name in meta_data_names:
            assert db_cand.get_meta_information(meta_data_name) is not None
            assert data_cand.get_meta_information(meta_data_name) is not None
            assert np.array(db_cand.get_meta_information(meta_data_name) == data_cand.get_meta_information(meta_data_name)).all()

    database_created_energies = np.array([atoms.get_potential_energy() for atoms in database_created.get_all_candidates()])
    dataset_energies = np.array([atoms.get_potential_energy() for atoms in dataset])
    assert (database_created_energies == dataset_energies).all()

    database_created_forces = np.array([atoms.get_forces() for atoms in database_created.get_all_candidates()])
    dataset_forces = np.array([atoms.get_forces() for atoms in dataset])
    assert (database_created_forces == dataset_forces).all()
    database_created.write(already_stored=True, force_write=True)

    # Load the database from disk:
    database_loaded = Database('db1.db')
    database_loaded.restore_to_memory()

    assert len(database_loaded) == len(dataset)

    for db_cand, data_cand in zip(database_loaded.get_all_candidates(), dataset):
        assert (db_cand.positions == data_cand.positions).all()
        assert (db_cand.cell == data_cand.cell).all()
        assert (db_cand.pbc == data_cand.pbc).all()
        for meta_data_name in meta_data_names:
            assert db_cand.get_meta_information(meta_data_name) is not None
            assert data_cand.get_meta_information(meta_data_name) is not None
            assert np.array(db_cand.get_meta_information(meta_data_name) == data_cand.get_meta_information(meta_data_name)).all()

    database_loaded_energies = np.array([atoms.get_potential_energy() for atoms in database_loaded.get_all_candidates()])
    dataset_energies = np.array([atoms.get_potential_energy() for atoms in dataset])
    assert (database_loaded_energies == dataset_energies).all()

    database_loaded_forces = np.array([atoms.get_forces() for atoms in database_loaded.get_all_candidates()])
    dataset_forces = np.array([atoms.get_forces() for atoms in dataset])

    assert (database_loaded_forces == dataset_forces).all()
    database_loaded.write(already_stored=True, force_write=True)

    # Stuff about restoring parts of the database:        
    energies_directly_restored = database_loaded.get_all_energies()
    assert (energies_directly_restored == dataset_energies).all()

    atoms_direct = database_loaded.get_structure(10)
    assert (atoms_direct.positions == dataset[9].positions).all() # Index offset

    atoms_best_direct = database_loaded.get_best_structure()
    E = atoms_best_direct.get_potential_energy()
    assert E == np.min(dataset_energies)

    ebest = database_loaded.get_best_energy()
    assert E == ebest

    latest = database_loaded.get_most_recent_candidate()
    assert (latest.positions == dataset[-1].positions).all()

    # Move back to start dir as other tests fail otherwise.
    os.chdir(start_dir)



