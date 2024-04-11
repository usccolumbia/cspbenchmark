import os
from ase.io import read

path = os.path.dirname(os.path.abspath(__file__))

Ag5O3 = read(os.path.join(path, 'Ag5O3-dataset.traj'), ':')

datasets = {
    'Ag5O3': Ag5O3,
}
