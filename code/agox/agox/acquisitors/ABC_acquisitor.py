from abc import ABC, abstractmethod
import numpy as np
from agox.observer import Observer
from agox.writer import Writer, agox_writer

class AcquisitorBaseClass(ABC, Observer, Writer):

    """
    Acquisition Base Class. 

    See: Observer-class for information on gets, sets, order. 

    verbose: bool
        Controls how much printing the module does. 

    Abstract methods:

    calculate_acquisition_function: 
        Calculates acquisition function for given candidates.         
    """

    def __init__(self, order=4, gets={'get_key':'candidates'}, sets={'set_key':'prioritized_candidates'}, verbose=True, 
                use_counter=True, prefix='', surname=''):
        Observer.__init__(self, gets=gets, sets=sets, order=order, surname=surname)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)

        self.add_observer_method(self.prioritize_candidates,
                                 sets=self.sets[0], gets=self.gets[0], order=self.order[0],
                                 handler_identifier='AGOX')

    ########################################################################################
    # Required methods
    ########################################################################################

    @abstractmethod
    def calculate_acquisition_function(self, candidates):
        """
        Implements the acquisition function. 

        Parameters
        -----------
        candidates: list
            List of candidates that the acquisition function will be evaluated for. 
        
        Returns
        --------
        np.array
            Numpy array of acquisition function values. 

        """
        return acquisition_values

    @agox_writer
    @Observer.observer_method
    def prioritize_candidates(self, state):
        """
        Method that is attached to the AGOX iteration loop as an observer - not intended for use outside of that loop. 

        The method does the following: 
        1. Gets candidates from the cache using 'get_key'.
        2. Removes 'None' from the candidate list. 
        3. Calculates and sorts according to acquisition function. 
        4. Adds the sorted candidates to cache with 'set_key'
        5. Prints information.         
        """

        # Get data from the iteration data dict. 
        candidate_list = state.get_from_cache(self, self.get_key)
        candidate_list = list(filter(None, candidate_list))

        # Calculate acquisition function values and sort:
        if self.do_check():
            candidate_list, acquisition_values = self.sort_according_to_acquisition_function(candidate_list)
        else:
            acquisition_values = np.zeros(len(candidate_list))
        # Add the prioritized candidates to the iteration data in append mode!
        state.add_to_cache(self, self.set_key, candidate_list, mode='a')

        self.print_information(candidate_list, acquisition_values)
            
    ########################################################################################
    # Default methods
    ########################################################################################
        
    def sort_according_to_acquisition_function(self, candidates):        
        """
        Calculates acquisiton-function based on the implemeneted version calculate_acquisition_function. 

        Note: Sorted so that the candidate with the LOWEST acquisition function value is first. 

        Parameters
        ------------
        candidates: list
            List of candidate objects. 
        
        Returns: 
        -----------
        list
            List of candidates sorted according to increasing acquisition value (lowest first).
        np.array
            Array of acquisition function values in the same order as the sorted list, i.e. increasing. 

        """
        acquisition_values = self.calculate_acquisition_function(candidates)
        sort_idx = np.argsort(acquisition_values)
        sorted_candidates = [candidates[i] for i in sort_idx]
        acquisition_values = acquisition_values[sort_idx]
        [candidate.add_meta_information('acquisition_value', acquisition_value) for candidate, acquisition_value in zip(sorted_candidates, acquisition_values)]          
        return sorted_candidates, acquisition_values

    def get_random_candidate(self):
        DeprecationWarning('Will be removed in the future. Please use collector.get_random_candidate')
        return self.collector.get_random_candidate()

    def print_information(self, candidates, acquisition_values):
        """
        Printing function for analysis/debugging/sanity checking. 

        Parameters
        -----------
        candidates: list
            List of candidate objects. 
        acquisition_values: np.array
            Acquisition function value for each candidate in 'candidates'. 

        """

    def get_acquisition_calculator(self):
        """
        Creates a calculator for the acquisiton function that can be used for e.g. relaxation. 

        Returns
        --------
        ASE Calculator
            ASE calculator where the energy is the acquisition function and forces are the forces of the acquisition forces. 
        """

        raise NotImplementedError("'get_acqusition_calculator' is not implemented for this acquisitor")

from ase.calculators.calculator import Calculator, all_changes
from agox.module import Module

class AcquisitonCalculatorBaseClass(Calculator, Module):

    name = 'AcqusitionCalculator'

    def __init__(self, model_calculator, **kwargs):
        super().__init__(**kwargs)
        self.model_calculator = model_calculator
    
    @property
    def verbose(self):
        return self.model_calculator.verbose

    def get_model_parameters(self):
        parameters = self.model_calculator.get_model_parameters()
        return parameters

    def set_model_parameters(self, parameters):
        self.model_calculator.set_model_parameters(parameters)

    def set_verbosity(self, verbose):
        self.model_calculator.set_verbosity(verbose)

    def get_iteration_number(self):
        if hasattr(self, 'get_iteration_counter'):
            return self.get_iteration_counter()
        else:
            return self.iteration

    @property
    def ready_state(self):
        return self.model_calculator.ready_state
