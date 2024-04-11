import matplotlib
matplotlib.use('Agg')

import numpy as np
from agox import AGOX
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RandomGenerator
from ase import Atoms

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
db_path = 'db1.db'
database = Database(filename=db_path, order=3)

##############################################################################
# Search Settings:
##############################################################################

random_generator = RandomGenerator(**environment.get_confinement(), 
    environment=environment, order=1)

evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
    optimizer_kwargs={'logfile':None}, store_trajectory=False,
    optimizer_run_kwargs={'fmax':0.05, 'steps':400}, 
    order=2)

##############################################################################
# Let get the show running! 
##############################################################################

from agox.writer import Writer, agox_writer
from agox.observer import Observer

class GetsBasicObserver(Observer, Writer):

    name = 'LessBasicObserver'

    def __init__(self, order=3, gets={'get_key':'evaluated_candidates'}):
        Observer.__init__(self, gets=gets, order=order)
        Writer.__init__(self)
        self.add_observer_method(self.basic_observer_method, order=self.order[0], 
            sets=self.sets[0], gets=self.gets[0])

    @agox_writer
    @Observer.observer_method
    def basic_observer_method(self, state):
        self.writer(f'Iteration: {self.get_iteration_counter()}: I AM A MODULE.')

        candidates = state.get_from_cache(self, self.get_key)

        for candidate in candidates:
            E = candidate.get_potential_energy()
            self.writer(f'Candidate with energy {E}')

basic_observer = GetsBasicObserver()

agox = AGOX(random_generator, database, basic_observer, evaluator)

agox.run(N_iterations=1)