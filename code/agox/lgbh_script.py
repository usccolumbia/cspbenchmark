import matplotlib
matplotlib.use('Agg')

import numpy as np
from agox import AGOX
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RattleGenerator
from agox.samplers import MetropolisSampler
from agox.postprocessors import RelaxPostprocess
from agox.models.local_GPR.LSGPR_CUR import LSGPRModelCUR
from agox.models.descriptors.soap import SOAP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from ase import Atoms
from ase.optimize import BFGS

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
seed = args.seed
prsteps = args.prsteps

##############################################################################    
# System & general settings:
##############################################################################
    
template = Atoms('', cell=np.eye(3)*12)
confinement_cell = np.eye(3) * 6
confinement_corner = np.array([3, 3, 3])
environment = Environment(template=template, symbols=composition, 
    confinement_cell=confinement_cell, confinement_corner=confinement_corner)

# Database
db_folder = 'results/' + composition + '/local_gpr_basin_hopin/'
if not os.path.exists(db_folder):
    os.makedirs(db_folder)
db_path = db_folder + 'results0.db'

database = Database(filename=db_path, order=4, write_frequency=1)

##############################################################################
# Search Settings:
##############################################################################

kernel = C(1)*RBF(length_scale=20)
descriptor = SOAP(environment.get_all_species(), r_cut=5., nmax=3, lmax=2, sigma=1, 
    weight=True, periodic=False)
model = LSGPRModelCUR(database=database, kernel=kernel,  descriptor = descriptor, 
    noise=0.01, prior=None, verbose=True, iteration_start_training=0, use_prior_in_training=True)

# Number of steps is very low - should be set higher for a real search!
relaxer = RelaxPostprocess(model, optimizer=BFGS, order=2, 
    optimizer_run_kwargs={'steps':prsteps, 'fmax':0.1})

sampler = MetropolisSampler(temperature=0.25, order=4)
    
rattle_generator = RattleGenerator(**environment.get_confinement(), 
    environment=environment, sampler=sampler, order=1)

# With the model pre-relax we dont want to take many steps in the real potential!
# As we are training the model all the data is saved with the store_trajectory argument. 
evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
    store_trajectory=True, optimizer_run_kwargs={'fmax':0.05, 'steps':0}, 
    order=3, constraints=environment.get_constraints())

##############################################################################
# Let get the show running! 
##############################################################################
    
agox = AGOX(rattle_generator, database, sampler, evaluator, relaxer, seed=seed)

agox.run(N_iterations=num_iters)
