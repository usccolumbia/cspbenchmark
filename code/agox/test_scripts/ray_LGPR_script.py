import matplotlib
matplotlib.use('Agg')

import numpy as np
from agox.main import AGOX
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RandomGenerator, RattleGenerator
from agox.samplers import KMeansSampler
from agox.collectors import ParallelCollector
from agox.acquisitors import LowerConfidenceBoundAcquisitor
from agox.postprocessors import ParallelRelaxPostprocess

from agox.models import ModelGPR

from agox.models.descriptors.soap import SOAP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from agox.models.local_GPR.LSGPR_CUR import LSGPRModelCUR
from agox.models.priors.repulsive import Repulsive

from ase import Atoms

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', '--run_idx', type=int, default=0)
args = parser.parse_args()

##############################################################################
# Calculator
##############################################################################

from ase.calculators.emt import EMT

calc = EMT()

##############################################################################    
# System & general settings:
##############################################################################
    
template = Atoms('', cell=np.eye(3)*12)
confinement_cell = np.eye(3) * 8
confinement_corner = np.array([3, 3, 3])
environment = Environment(template=template, symbols='Au8Ni8', 
    confinement_cell=confinement_cell, confinement_corner=confinement_corner)

# Database
db_path = 'db{}.db'.format(args.run_idx) # From input argument!
database = Database(filename=db_path, order=6)

##############################################################################
# Search Settings:
##############################################################################

gmodel = ModelGPR.default(environment, None)

descriptor = SOAP(environment.get_all_species(), r_cut=4., nmax=3, lmax=2, sigma=1., weight=True, periodic=True)
kernel = C(1)*RBF(length_scale=20)
model = LSGPRModelCUR(database=database, kernel=kernel, descriptor=descriptor, noise=0.01, prior=Repulsive(),
                      iteration_start_training=-1, m_points=2000, verbose=True)

sample_size = 10
sampler = KMeansSampler(feature_calculator=gmodel.get_feature_calculator(), 
                        database=database, sample_size=sample_size, order=1)

rattle_generator = RattleGenerator(**environment.get_confinement())
random_generator = RandomGenerator(**environment.get_confinement())
generators = [random_generator, rattle_generator]
num_candidates = {0:[10, 0], 10:[3, 7]}

acquisitor = LowerConfidenceBoundAcquisitor(model_calculator=model, 
    kappa=2, order=4)

collector = ParallelCollector(generators=generators, sampler=sampler,
    environment=environment, num_candidates=num_candidates, order=2)

relaxer = ParallelRelaxPostprocess(
    model=acquisitor.get_acquisition_calculator(), 
    constraints=environment.get_constraints(), 
    order=3,
    start_relax=8,
    optimizer_run_kwargs={'steps':50, 'fmax':0.5}
)

evaluator = LocalOptimizationEvaluator(calc, 
                                       gets={'get_key':'prioritized_candidates'}, 
                                       optimizer_kwargs={'logfile':None},#, use_all_traj_info=True,
                                       optimizer_run_kwargs={'fmax':0.05, 'steps':1}, order=5)

##############################################################################
# Let get the show running! 
##############################################################################
    
agox = AGOX(collector, acquisitor, relaxer, database, evaluator, seed=1)
print(np.random.randint(0, 1000))
agox.run(N_iterations=25)
print(np.random.randint(0, 1000))
