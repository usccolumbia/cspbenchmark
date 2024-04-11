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

##############################################################################
# Calculator
##############################################################################

from ase.calculators.emt import EMT

# calc = EMT()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  ### for disabling tensorflow warnings

from m3gnet.models import M3GNet
calc = M3GNet.load()

##############################################################################
# Parser:
##############################################################################

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--iter', help = "Number of iterations", type = int, default = 5)
parser.add_argument('--comp', help = "Composition of the material", type = str, default = 'SrTiO3')
parser.add_argument('--seed', help = "Seed", type = int, default = 42)

args = parser.parse_args()
num_iters = args.iter
composition = args.comp
seed = args.seed

##############################################################################    
# System & general settings:
##############################################################################
    
template = Atoms('', cell=np.eye(3)*12)
confinement_cell = np.eye(3) * 6
confinement_corner = np.array([3, 3, 3])
environment = Environment(template=template, symbols=composition, 
    confinement_cell=confinement_cell, confinement_corner=confinement_corner)

# Database
total_workers = 1
sync_frequency = 10
worker_index = 0

db_folder = 'results/' + composition + '/parallel_tempering/'
if not os.path.exists(db_folder):
    os.makedirs(db_folder)
db_path = db_folder + 'results0.db'

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
    store_trajectory=False, optimizer_run_kwargs={'fmax':0.05, 'steps':0}, 
    order=2, constraints=environment.get_constraints())

##############################################################################
# Let get the show running! 
##############################################################################
    
agox = AGOX(rattle_generator, database, evaluator, sampler, seed=seed)

agox.run(N_iterations=num_iters)
