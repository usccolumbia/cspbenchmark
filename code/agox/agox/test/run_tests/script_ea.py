import matplotlib
matplotlib.use('Agg')

import numpy as np
from agox import AGOX
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RattleGenerator, RandomGenerator
from agox.samplers import GeneticSampler, DistanceComparator
from agox.collectors import StandardCollector
from agox.models.descriptors.simple_fingerprint import SimpleFingerprint
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
confinement_corner = np.array([3, 3, 3])
environment = Environment(template=template, symbols='Au8Ni8', 
    confinement_cell=confinement_cell, confinement_corner=confinement_corner)

# Database
db_path = 'db{}.db'.format(database_index) # From input argument!
database = Database(filename=db_path, order=3, write_frequency=1)

##############################################################################
# Search Settings:
##############################################################################

population_size = 10
descriptor = SimpleFingerprint(species=['Au', 'Ni'])
comparator = DistanceComparator(descriptor, threshold=0.5)
sampler = GeneticSampler(population_size=population_size, comparator=comparator, 
    order=4, database=database)
    
rattle_generator = RattleGenerator(**environment.get_confinement())
random_generator = RandomGenerator(**environment.get_confinement())
generators = [random_generator, rattle_generator]
num_candidates = {
    0:[population_size, 0], 
    5:[2, population_size-2], 
    10:[0, population_size]}

collector = StandardCollector(generators=generators, sampler=sampler, 
    environment=environment, num_candidates=num_candidates, order=1)

# Very few steps used here - Set to higher number (>100) for a real run. 
# The number fo evaluate is equal to to population size - which is also 
# the total number of structures generated pr. iteration. 
evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
    optimizer_run_kwargs={'fmax':0.05, 'steps':5}, store_trajectory=False,
    order=2, constraints=environment.get_constraints(), 
    number_to_evaluate=population_size)

##############################################################################
# Let get the show running! 
##############################################################################
    
agox = AGOX(collector, database, evaluator, seed=seed)

agox.run(N_iterations=5)