import matplotlib
matplotlib.use('Agg')

import numpy as np
from agox import AGOX
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RandomGenerator, RattleGenerator
from agox.samplers import KMeansSampler
from agox.collectors import ParallelCollector
from agox.models import ModelGPR
from agox.acquisitors import LowerConfidenceBoundAcquisitor
from agox.postprocessors import ParallelRelaxPostprocess
from ase import Atoms

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
parser.add_argument('--prsteps', help = "ParallelRelaxPostprocess steps", type = int, default = 5000)

args = parser.parse_args()
num_iters = args.iter
composition = args.comp
prsteps = args.prsteps
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
db_folder = 'results/' + composition + '/GOFEE/'
if not os.path.exists(db_folder):
    os.makedirs(db_folder)
db_path = db_folder + 'results0.db'

database = Database(filename=db_path, order=6, write_frequency=1)

##############################################################################
# Search Settings:
##############################################################################

model = ModelGPR.default(environment, database)

sample_size = 10
sampler = KMeansSampler(feature_calculator=model.get_feature_calculator(), 
    database=database, sample_size=sample_size, order=1)

rattle_generator = RattleGenerator(**environment.get_confinement())
random_generator = RandomGenerator(**environment.get_confinement())

# Dict specificies how many candidates are created with and the dict-keys are iterations. 
generators = [random_generator, rattle_generator]
num_candidates = {0:[10, 0], 5:[3, 7]}

acquisitor = LowerConfidenceBoundAcquisitor(model_calculator=model, 
    kappa=2, order=4)

# CPU-count is set here for Ray - leave it out to use as many cores as are available. 
collector = ParallelCollector(generators=generators, sampler=sampler,
    environment=environment, num_candidates=num_candidates, order=2, 
    cpu_count=5)
    
# Number of steps is very low - should be set higher for a real search!
relaxer = ParallelRelaxPostprocess(model=acquisitor.get_acquisition_calculator(), 
    constraints=environment.get_constraints(), order=3, start_relax=8, 
    optimizer_run_kwargs={'steps':prsteps})

evaluator = LocalOptimizationEvaluator(calc, 
    gets={'get_key':'prioritized_candidates'}, 
    optimizer_kwargs={'logfile':None}, store_trajectory=True,
    optimizer_run_kwargs={'fmax':0.05, 'steps':0}, order=5)

##############################################################################
# Let get the show running! 
##############################################################################
    
agox = AGOX(collector, acquisitor, relaxer, database, evaluator, seed = seed)

agox.run(N_iterations = num_iters)

