from abc import ABC, abstractmethod
from ase.calculators.singlepoint import SinglePointCalculator
#from agox import Observer, Writer, agox_writer, Module, ObserverHandler
from agox.observer import Observer, ObserverHandler
from agox.writer import Writer, agox_writer


from ase import Atoms

class DatabaseBaseClass(ABC, ObserverHandler, Observer, Writer):

    def __init__(self, gets={'get_key':'evaluated_candidates'}, sets={}, order=6, verbose=True, use_counter=True, 
        prefix='', surname=''):
        Observer.__init__(self, gets=gets, sets=sets, order=order, surname=surname)
        ObserverHandler.__init__(self, handler_identifier='database', dispatch_method=self.store_in_database)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)
        self.candidates = []

        self.objects_to_assign = []

        self.add_observer_method(self.store_in_database,
                                 sets=self.sets[0], gets=self.gets[0], order=self.order[0],
                                 handler_identifier='AGOX')

    ########################################################################################
    # Required methods                                                          
    ########################################################################################

    @abstractmethod
    # def write(self, positions, energy, atom_numbers, cell, **kwargs):
    def write(self, grid):
        """
        Write stuff to database
        """

    @abstractmethod
    def store_candidate(self, candidate_object):
        pass

    @abstractmethod
    def get_all_candidates(self):
        pass

    @property
    @abstractmethod
    def name(self):
        return NotImplementedError
    
    ########################################################################################
    # Default methods
    ########################################################################################

    def __len__(self):
        return len(self.candidates) 

    @agox_writer
    @Observer.observer_method
    def store_in_database(self, state):
        
        evaluated_candidates = state.get_from_cache(self, self.get_key)
        anything_accepted = False
        for j, candidate in enumerate(evaluated_candidates):

            # Dispatch to observers only when adding the last candidate. 
            dispatch = (j+1) == len(evaluated_candidates)

            if candidate: 
                self.writer('Energy {:06d}: {}'.format(self.get_iteration_counter(), candidate.get_potential_energy()), flush=True)
                self.store_candidate(candidate, accepted=True, write=True, dispatch=False)
                anything_accepted = True

            elif candidate is None:
                dummy_candidate = self.candidate_instanstiator(template=Atoms())
                dummy_candidate.set_calculator(SinglePointCalculator(dummy_candidate, energy=float('nan')))

                # This will dispatch to observers if valid data has been added but the last candidate is None. 
                self.store_candidate(candidate, accepted=False, write=True, dispatch=False)

        if anything_accepted:
            self.dispatch_to_observers(self, state)
