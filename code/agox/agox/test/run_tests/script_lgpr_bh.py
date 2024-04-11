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

kernel = C(1)*RBF(length_scale=20)
descriptor = SOAP(environment.get_all_species(), r_cut=5., nmax=3, lmax=2, sigma=1, 
    weight=True, periodic=False)
model = LSGPRModelCUR(database=database, kernel=kernel,  descriptor = descriptor, 
    noise=0.01, prior=None, verbose=True, iteration_start_training=0, use_prior_in_training=True)

# Number of steps is very low - should be set higher for a real search!
relaxer = RelaxPostprocess(model, optimizer=BFGS, order=2, 
    optimizer_run_kwargs={'steps':5, 'fmax':0.1})

sampler = MetropolisSampler(temperature=0.25, order=4)
    
rattle_generator = RattleGenerator(**environment.get_confinement(), 
    environment=environment, sampler=sampler, order=1)

# With the model pre-relax we dont want to take many steps in the real potential!
# As we are training the model all the data is saved with the store_trajectory argument. 
evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
    store_trajectory=True, optimizer_run_kwargs={'fmax':0.05, 'steps':3}, 
    order=3, constraints=environment.get_constraints())

##############################################################################
# Let get the show running! 
##############################################################################
    
agox = AGOX(rattle_generator, database, sampler, evaluator, relaxer, seed=seed)

agox.run(N_iterations=10)
