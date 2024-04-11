import numpy as np
import os
from agox.databases import Database
from argparse import ArgumentParser
from ase.io import read, write

def convert_database_to_traj(database_path):
    """
    Convert database to list of ASE Atoms objects. 

    Currently assumes that the database can be read using the standard database 
    class (agox/modules/databases/database.py).

    Parameters
    -----------
    database_path: str
        Path to database on disk. 

    Returns
    --------
    list
        List of ASE Atoms objects. 
    """

    if os.path.exists(database_path):
        database = Database(database_path, initialize=False)
    else:
        print(f'Trajectory not found: {database_path}')
        return []

    trajectory = database.restore_to_trajectory()

    return trajectory

def find_file_types(paths):
    """
    Find file extension of given paths. 

    Parameters
    -----------
    paths: list of str
        Paths to check
    
    Returns 
    --------
    list: 
        List of strs indicating of the file types (extensions) of the given paths. 
    """
    file_types = []
    for path in paths:
        file_types.append(path.split('.')[-1])
    return file_types

def convert():
    """
    Command line function to convert database files to trajectories. 

    Does not take any Python inputs but has the following comamnd line arguments: 

    files: (positional): Paths of database files to convert e.g. given as db1.db db2.db 
    name: (-n, --name): Name of the trajectory file to save including .traj extension. 
    """
    parser = ArgumentParser()
    parser.add_argument('files', type=str, nargs='+')
    parser.add_argument('-n', '--name', type=str, default=None)
    args = parser.parse_args()

    # Determine the file extension & ensure all paths have the same extension. 
    file_types = find_file_types(args.files)
    assert np.array([file_type == file_types[0] for file_type in file_types]).all()
    file_type = file_types[0]

    # If database files - convert to trajectory. 
    trajectory = []
    if file_type == 'db':
        for path in args.files:
            print(f'Converting: {path}')
            trajectory += convert_database_to_traj(path)

        # Naming: 
        name = args.name
        if name is None:
            name = 'converted_db.traj'
        
        # Save: 
        print(f'Saving trajectory file {name}')
        write(name, trajectory)

        
if __name__ == '__main__':
    convert()



