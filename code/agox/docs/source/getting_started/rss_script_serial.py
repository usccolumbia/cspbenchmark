import matplotlib
matplotlib.use('Agg')

import numpy as np
from agox import AGOX
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RandomGenerator
from ase import Atoms

for run_idx in range(10):

    ##############################################################################
    # Calculator
    ##############################################################################

    from ase.calculators.emt import EMT

    calc = EMT()

    ##############################################################################
    # General settings:
    ##############################################################################

    template = Atoms('', cell=np.eye(3)*12)
    confinement_cell = np.eye(3) * 6
    confinement_corner = np.array([3, 3, 3])
    environment = Environment(template=template, symbols='Au5Ni', 
        confinement_cell=confinement_cell, confinement_corner=confinement_corner)

    # Database
    db_path = 'db{}.db'.format(run_idx)
    database = Database(filename=db_path, order=3)

    ##############################################################################
    # Search Settings:
    ##############################################################################

    random_generator = RandomGenerator(**environment.get_confinement(), 
        environment=environment, order=1)

    evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
        optimizer_kwargs={'logfile':None}, store_trajectory=False,
        optimizer_run_kwargs={'fmax':0.05, 'steps':400}, order=2)

    ##############################################################################
    # Let get the show running! 
    ##############################################################################

    agox = AGOX(random_generator, database, evaluator)

    agox.run(N_iterations=50)