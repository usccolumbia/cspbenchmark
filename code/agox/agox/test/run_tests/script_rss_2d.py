import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
from agox import AGOX
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RandomGenerator
from ase import Atoms
from argparse import ArgumentParser

# Manually set seed and database-index
seed = 42
database_index = 0

# Using argparse if e.g. using array-jobs on Slurm to do several independent searches. 
# parser = ArgumentParser()
# parser.add_argument('-i', '--run_idx', type=int, default=0)
# args = parser.parse_args()

# seed = args.run_idx
# database_index = args.run_idx

##############################################################################
# Calculator
##############################################################################

from ase.calculators.emt import EMT

calc = EMT()

##############################################################################    
# System & general settings:
##############################################################################
    
template = Atoms('', cell=np.eye(3)*12)
confinement_cell = np.eye(3) * 6
confinement_cell[2, 2] = 0 # Zero height of confinement cell for the third dimension. 
confinement_corner = np.array([3, 3, 6])
environment = Environment(template=template, symbols='Au4Ni4', 
    confinement_cell=confinement_cell, confinement_corner=confinement_corner)

# Database
db_path = 'db{}.db'.format(database_index) # From input argument!
database = Database(filename=db_path, order=3, write_frequency=1)

# We add an additional constraint from ASE to keep atoms confined to 2D during 
# relaxations. 
# The box constraint will keep them in the specified cell in the other dimensions.
from ase.constraints import FixedPlane
fixed_plane = [FixedPlane(i, [0, 0, 1]) for i in environment.get_missing_indices()]
constraints = environment.get_constraints() + fixed_plane

##############################################################################
# Search Settings:
##############################################################################
    
random_generator = RandomGenerator(**environment.get_confinement(), 
    environment=environment, order=1)

# Wont relax fully with steps:5 - more realistic setting would be 100+.
evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
    optimizer_run_kwargs={'fmax':0.05, 'steps':5}, store_trajectory=False,
    order=2, constraints=constraints)

##############################################################################
# Let get the show running! 
##############################################################################

agox = AGOX(random_generator, database, evaluator, seed=seed)

agox.run(N_iterations=10)

# Confirm that positions of all atoms placed by the search are in a plane. 
# Can leave out of a run. 
assert (np.array([c.get_positions() for c in database.get_all_candidates()])[:, environment.get_missing_indices(), 2] == confinement_corner[2]).all(), 'Not all positions are 2D'