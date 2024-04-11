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
parser.add_argument('--pop', help = "Population size", type = int, default = 50)

args = parser.parse_args()
num_iters = args.iter
composition = args.comp
seed = args.seed
population_size = args.pop

##############################################################################    
# System & general settings:
##############################################################################
    
template = Atoms('', cell=np.eye(3)*12)
confinement_cell = np.eye(3) * 6
confinement_corner = np.array([3, 3, 3])
environment = Environment(template=template, symbols=composition, 
    confinement_cell=confinement_cell, confinement_corner=confinement_corner)

# Database
db_folder = 'results/' + composition + '/evolutionary_algorithm/'
if not os.path.exists(db_folder):
    os.makedirs(db_folder)
db_path = db_folder + 'results0.db'

database = Database(filename=db_path, order=3, write_frequency=1)

##############################################################################
# Search Settings:
##############################################################################

from pymatgen.core.composition import Composition
_formula = Composition(composition)
_species = []
for elem in _formula.elements:
    _species.append(str(elem))
# print(_species)

descriptor = SimpleFingerprint(species=_species)
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
# The number to evaluate is equal to to population size - which is also 
# the total number of structures generated pr. iteration. 
evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
    optimizer_run_kwargs={'fmax':0.05, 'steps':0}, store_trajectory=False,
    order=2, constraints=environment.get_constraints(), 
    number_to_evaluate=population_size)

##############################################################################
# Let get the show running! 
##############################################################################
    
agox = AGOX(collector, database, evaluator, seed=seed)

agox.run(N_iterations = num_iters)
