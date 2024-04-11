import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
from agox import AGOX
from agox.databases import ConcurrentDatabase
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RattleGenerator
from agox.samplers import ParallelTemperingSampler
from ase import Atoms

from argparse import ArgumentParser

# Manually set seed and run_idx
seed = 42
run_idx = 0

# Using argparse if e.g. using array-jobs on Slurm to do several independent searches. 
# parser = ArgumentParser()
# parser.add_argument('-i', '--run_idx', type=int, default=0)
# args = parser.parse_args()

# seed = args.run_idx
# run_idx = args.run_idx

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
environment = Environment(template=template, symbols='Au12Ni2', 
    confinement_cell=confinement_cell, confinement_corner=confinement_corner)

# Database
total_workers = 1 # This should be higher for a real run. 
sync_frequency = 10
database_index = (run_idx-1) // total_workers + 1
worker_index = (run_idx-1) % total_workers
db_path = 'db{}.db'.format(database_index)
database = ConcurrentDatabase(filename=db_path, store_meta_information=True, 
    write_frequency=1, worker_number=worker_index, total_workers=total_workers, 
    sync_frequency=sync_frequency, order=4, sleep_timing=0.1)

# ##############################################################################
# # Search Settings:
# ##############################################################################

temperatures = [0.1*1.5**power for power in range(total_workers)]
sampler = ParallelTemperingSampler(temperatures=temperatures, order=3, 
    database=database, swap_order=5)
    
rattle_generator = RattleGenerator(**environment.get_confinement(), 
    environment=environment, sampler=sampler, order=1)

# Very few steps used here - Set to higher number (>100) for a real run. 
evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
    store_trajectory=False, optimizer_run_kwargs={'fmax':0.05, 'steps':5}, 
    order=2, constraints=environment.get_constraints())

##############################################################################
# Let get the show running! 
##############################################################################
    
agox = AGOX(rattle_generator, database, evaluator, sampler, seed=seed)

agox.run(N_iterations=20)