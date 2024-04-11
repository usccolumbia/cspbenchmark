import matplotlib
matplotlib.use('Agg')

import numpy as np
from agox import AGOX
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RattleGenerator
from agox.samplers import MetropolisSampler
from ase import Atoms

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
confinement_corner = np.array([3, 3, 3])
environment = Environment(template=template, symbols='Au8Ni8', 
    confinement_cell=confinement_cell, confinement_corner=confinement_corner)

# Database
db_path = 'db{}.db'.format(database_index) # From input argument!
database = Database(filename=db_path, order=4, write_frequency=1)

##############################################################################
# Search Settings:
##############################################################################

sampler = MetropolisSampler(temperature=0.25, order=3)
    
rattle_generator = RattleGenerator(**environment.get_confinement(), 
    environment=environment, sampler=sampler, order=1)

# Wont relax fully with steps:5 - more realistic setting would be 100+.
evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
    store_trajectory=False, optimizer_run_kwargs={'fmax':0.05, 'steps':5}, 
    order=2, constraints=environment.get_constraints())

##############################################################################
# Let get the show running! 
##############################################################################
    
agox = AGOX(rattle_generator, database, sampler, evaluator, seed=seed)

agox.run(N_iterations=10)
