import numpy as np
from abc import ABC, abstractmethod
from agox.module import Module
from agox.observer import Observer
from agox.writer import Writer, agox_writer

class CollectorBaseClass(ABC, Observer, Writer):

    def __init__(self, generators=None, sampler=None, environment=None, order=2,
        sets={'set_key':'candidates'}, gets={}, verbose=True, use_counter=True,
        prefix='', surname=''):
        """
        Quite simple, except for lines 50-54 that remove generators as observers,
        as they are now part of a Collector. This may now yield wanted behaviour 
        for generators that need observers_methods that are not just for 
        generation, if that becomes the case an update to those lines are 
        probably necesary. 

        Parameters
        ----------
        generators : list
            List of AGOX generator instances. 
        sampler : Sampler instance
            An instance of an AGOX sampler
        environment : Environment object.
            An instance of an AGOX environment object. 
        """
        Observer.__init__(self, sets=sets, gets=gets, order=order, surname=surname)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)

        assert generators is not None
        assert environment is not None

        self.generators = generators
        self.sampler = sampler
        self.environment = environment
        self.candidates = []
        self.plot_confinement()

        self.add_observer_method(self.generate_candidates,
                                 sets=self.sets[0], gets=self.gets[0], order=self.order[0],
                                 handler_identifier='AGOX')

        for generator in self.generators:            
            observer_methods = [observer_method for observer_method in generator.observer_methods.values()]
            for observer_method in observer_methods:
                if observer_method.method_name == 'generate':
                    generator.remove_observer_method(observer_method)

    @abstractmethod
    def make_candidates(self):
        """
        Method that calls generators to produce a collection of candidates. 

        Returns
        -------
        list 
           List of candidate objects. 
        """
        return candidates
    
    @abstractmethod
    def get_number_of_candidates(self, iteration):
        """
        Method  that computes the number of candidates for the given iteration.

        Parameters
        ----------
        iteration : int
            Iteration count.

        Returns
        -------
        list
            List of integers corresponding to the self.generators list dictating 
            how many candidates are generated (or really how many times each 
            generator is called.)
        """
        # Number of candidates to generate.
        return list_of_integers

    @agox_writer
    @Observer.observer_method
    def generate_candidates(self, state):
        # Make the candidates - this is the method that differs between versions of the class.
        if self.do_check():
            candidates = self.make_candidates()

        # Add to the iteration_cache:
        state.add_to_cache(self, self.set_key, candidates, 'a')
        self.writer('Number of candidates this iteration: {}'.format(len(candidates)))

########################################################################################################################
# Other methods
########################################################################################################################

    def attach(self, main):
        super().attach(main)
        for generator in self.generators:
            generator.attach(main)

    def plot_confinement(self):
        for generator in self.generators: 
            generator.plot_confinement(self.environment)
