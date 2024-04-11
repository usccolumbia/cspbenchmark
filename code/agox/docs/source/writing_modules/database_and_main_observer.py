from cgitb import handler
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
    optimizer_run_kwargs={'fmax':0.05, 'steps':2}, 
    order=2)

##############################################################################
# Let get the show running! 
##############################################################################

from agox.writer import Writer, agox_writer
from agox.observer import Observer

class DatabaseAndMainObserver(Observer, Writer):

    name = 'DatabaseAndMainObserver'

    def __init__(self, order=[0, 0]):
        Observer.__init__(self, order=order)
        Writer.__init__(self)
        self.add_observer_method(self.database_observer_method, order=self.order[0], 
            sets={}, gets={}, handler_identifier='database')
        self.add_observer_method(self.main_observer_method, order=self.order[0], 
            sets={}, gets={}, handler_identifier='AGOX')
        
    @agox_writer
    @Observer.observer_method
    def database_observer_method(self, database, state):
        size_of_database = len(database)
        self.writer(f'Database size: {size_of_database}')
        self.writer(f'I am run from the database, I see the database {database} and {state}.')

    @agox_writer
    @Observer.observer_method
    def main_observer_method(self, state):
        self.writer(f'I am run from main, I see only the state: {state}')

database_and_main_observer = DatabaseAndMainObserver()
database_and_main_observer.attach(database)

agox = AGOX(random_generator, database, evaluator, database_and_main_observer)

agox.run(N_iterations=1)

# database.observer_reports()
# database.print_observers()