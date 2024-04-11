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

class TwoMethodObserver(Observer, Writer):

    name = 'TwoMethodObserver'

    def __init__(self, order=[2, 4], gets=[{'get_generated':'candidates'},  
        {'get_evaluated':'evaluated_candidates'}]):
        Observer.__init__(self, gets=gets, order=order, sets={})
        Writer.__init__(self)
        self.add_observer_method(self.observer_method_1, order=self.order[0], 
            sets=self.sets[0], gets=self.gets[0])
        self.add_observer_method(self.observer_method_2, order=self.order[1], 
            sets=self.sets[0], gets=self.gets[1])

        self.mean = 0.5

    @agox_writer
    @Observer.observer_method
    def observer_method_1(self, state):
        self.writer(f'Iteration: {self.get_iteration_counter()}: I AM A MODULE.')

        candidates = state.get_from_cache(self, self.get_generated)

        # Very advanced prediction algorithm: 
        # 1. First look at the candidates. 
        number = 0
        for candidate in candidates:
            number += np.sum(candidate.positions)

        # 2. Then choose a random number regardless of that. 
        chance = np.random.normal(loc=self.mean, scale=1, size=1)

        if chance > 0.5:
            self.prediction = 'Good'
        else:
            self.prediction = 'Bad'

        self.writer(f'I predict the candidate will be: {self.prediction}')

    @agox_writer
    @Observer.observer_method
    def observer_method_2(self, state):

        candidates = state.get_from_cache(self, self.get_evaluated)
        
        for candidate in candidates:
            E = candidate.get_potential_energy()

            self.writer(f'I predicted candidate would be {self.prediction} and it has energy E = {E}.')

basic_observer = TwoMethodObserver()

agox = AGOX(random_generator, database, basic_observer, evaluator)

agox.run(N_iterations=1)