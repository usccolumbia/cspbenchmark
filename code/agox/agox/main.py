# Copyright 2021-2022, Mads-Peter V. Christiansen, Bjørk Hammer, Nikolaj Rønne. 
# This file is part of AGOX.
# AGOX is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# AGOX is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should 
# have received a copy of the GNU General Public License along with AGOX. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from agox.observer import ObserverHandler, FinalizationHandler
from agox.logger import Logger
from agox.writer import Writer, agox_writer, ICON, header_print

class AGOX(ObserverHandler, FinalizationHandler, Writer):
    """
    AGO-X
    Atomistic Global Optimization X
    """

    name = 'AGOX'

    def __init__(self, *args, **kwargs):
        """
        Observers are supplied through *args.

        Supported **kwargs:
            - seed: Random seed for numpy.
            - use_log: Boolean - use logger or not.
        """
        ObserverHandler.__init__(self, handler_identifier='AGOX', dispatch_method=self.run)
        FinalizationHandler.__init__(self)
        Writer.__init__(self, verbose=True, use_counter=False, prefix='')

        print(ICON)
        header_print('Initialization starting')

        self.elements = args
        
        from agox.candidates.standard import StandardCandidate
        candidate_instanstiator = kwargs.pop('candidate_instanstiator', StandardCandidate)
        self.candidate_instanstiator = candidate_instanstiator

        seed = kwargs.pop('seed', None)
        if seed is not None:
            np.random.seed(seed)
            self.writer('Numpy random seed: {}'.format(seed))

        # This attaches all Observers
        self._update()

        # This happens after because the Logger looks at the current observers.
        use_log = kwargs.pop('use_log', None)
        if use_log is not False:
            logger = Logger()
            logger.attach(self)
        
        unused_keys = False
        for key, value in kwargs.items():
            self.writer("Unused kwarg '{}' given with value {}".format(key, value))
            unused_keys = True
        if unused_keys:
            self.writer('Stopping due to unused keys as behavior may not be as expected')
            exit()
    
        self.print_observers(hide_log=True)
        self.observer_reports(hide_log=True)
        header_print('Initialization finished')


    def set_candidate_instanstiator(self, candidate_instanstiator):
        self.candidate_instanstiator = candidate_instanstiator
 
    def _update(self):
        """
        Calls 'attach' on all Observer-objects in 'self.elements' and updates 
        the 'candidate_instan
        """
        for element in self.elements:
            if hasattr(element, 'attach'):
                element.attach(self)
            if hasattr(element, 'set_candidate_instanstiator'):
                element.set_candidate_instanstiator(self.candidate_instanstiator)

    def run(self, N_iterations, verbose=True, hide_log=True):
        """
        Function called by runscripts that starts the actual optimization procedure. 

        This function is controlled by modules attaching themselves as observers to this module. 
        The order system ensures that modules with lower order are executed first, but no gurantee within each order, 
        so if two modules attach themselves both with order 0 then their individual execution order is not guranteed. 
        However, an observer with order 0 will always be executed before an observer with order 1. 

        The default ordering system is: 
        order = 0: Execution order

        All modules that intend to attach themselves as observers MUST take the order as an argument (with a default value(s)), 
        so that if a different order is wanted that can be controlled from runscripts. Do NOT change order default values!
        """

        # Main-loop calling the relevant observers.   
        state = State()
        while state.get_iteration_counter() <= N_iterations and not state.get_convergence_status():
            print('\n')
            self.header_print('Iteration: {}'.format(state.get_iteration_counter()))
            for observer in self.get_observers_in_execution_order():
                observer(state)

            self.header_print('Iteration finished')
            state.clear()
            state.advance_iteration_counter()

        # Some things may want to perform some operation only at the end of the run. 
        for method in self.get_finalization_methods():
            method()
        
class State:

    def __init__(self):
        """
        State object.

        Attributes
        ----------
        cache: dict
            Data communicated between modules is stored in the cache. 
        iteration_counter: int
            Keeps track of the number of iterations. 
        convergence: bool
            Convergence status, if True the iteration-loop is halted. 
        """

        self.cache = {}    
        self.iteration_counter = 1
        self.converged = False

    def get_iteration_counter(self):
        """
        Returns
        -------
        int
            The current iteration number. 
        """
        return self.iteration_counter

    def set_iteration_counter(self, count):
        """_summary_

        Parameters
        ----------
        count : int
            Iteration count
        """
        self.iteration_counter = count

    def advance_iteration_counter(self):
        """
        Adds one to the iteration counter. 
        """
        self.iteration_counter += 1

    def get_from_cache(self, observer, key):
        """

        Gets from the cache with the given key. The observed is passed along 
        aswell in order to ensure that the observer is allowed to get with 
        that key. 

        Parameters
        ----------
        observer : class object
            An AGOX Observer object, e.g. an instance of a Sampler. 
        key : str
            The key with which to get something from the cache. 

        Returns
        -------
        list
            List of things stored with the given key. 
        """
        # Makes sure the module has said it wants to get with this key.
        assert key in observer.get_values  
        return self.cache.get(key)

    def add_to_cache(self, observer, key, data, mode):
        """        
        Add data to the cache.

        Parameters
        ----------
        observer : class object
            An AGOX Observer object, e.g. an instance of a Sampler. 
        key : str
            The key with which to get something from the cache. 
        data : list
            List of data to store in the cache.
        mode : str
            Determines the mode in which the data is added to the cache:
            w: Will overwrite existing data with the same key. 
            a: Will append to existing data (if there is existing data). 
        """
        assert(type(data) == list)
        # Makes sure the module has said it wants to get with this key.
        assert key in observer.set_values
        assert mode in ['w', 'a']

        if key in self.cache.keys() and mode != 'w':
            self.cache[key] += data
        else:
            self.cache[key] = data

    def clear(self):
        """
        Clears the current cachce. Called at the end of each iteration. 
        """
        self.cache = {}

    def get_convergence_status(self):
        """
        Returns the convergence status.

        Returns
        -------
        bool
            If True convergence has been reached and the main iteration-loop 
            will halt. 
        """
        return self.converged

    def set_convergence_status(self, state):
        """
        Set the convergence status. 

        Parameters
        ----------
        state : bool
            If True convergence has been reached and the main iteration-loop 
            will halt. 
        """
        self.converged = state

    