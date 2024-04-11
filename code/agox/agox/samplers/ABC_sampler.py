from abc import ABC, abstractmethod
import numpy as np
from ase.io import write
from agox.candidates.ABC_candidate import CandidateBaseClass
from agox.module import Module
from agox.observer import Observer
from agox.writer import Writer, agox_writer

class SamplerBaseClass(ABC, Observer, Writer):

    def __init__(self, database=None, sets={}, gets={}, order=1, verbose=True, use_counter=True, prefix='', 
        use_transfer_data=True, surname=''):
        Observer.__init__(self, sets=sets, gets=gets, order=order, surname=surname)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)
        self.sample = []
        self.use_transfer_data = use_transfer_data
        self.transfer_data = []

        self.add_observer_method(self.setup_sampler,
                                 sets=self.sets[0], gets=self.gets[0], order=self.order[0],
                                 handler_identifier='database')

        if database is not None:
            self.attach_to_database(database)

    ########################################################################################
    # Required properties
    ########################################################################################    

    @property
    @abstractmethod
    def name(self):
        return NotImplementedError

    ########################################################################################
    # Required methods
    ########################################################################################
    
    @abstractmethod
    def get_random_member(self):
        """

        Get a random member of the sampler. Not allowed to return directly from 
        the database. Can utilize anything else set during the call to 
        self.setup

        Returns
        -------
        candidate object
            Candidate object describing the sample member.
        """
        return sample_member
    
    @abstractmethod
    def setup(self, all_candidates):
        """

        Function that does the setup for the Sampler, e.g. filters candidates 
        from the database or calculates probabilities. 

        Must set self.sample.

        Parameters
        ----------
        all_candidates : list
            List of all candidates that will be considered by the sampler. If 
            'setup_sampler' is not overwritten this will be all candidates 
            currently in the database during in an AGOX run.
        """
        return None

    ########################################################################################
    # Default methods
    ########################################################################################

    @agox_writer
    @Observer.observer_method
    def setup_sampler(self, database, state):
        """
        Observer function that is attached to the database. The database passes 
        itself and the state. 

        The state is not passed to the 'self.setup' function that is called, so 
        if any data on there is required it can be set as an attribute, e.g. 
            self.something = state.get(...)
        or (probably preferably) the this function can be overwritten and more 
        things can be parsed to sampler.setup(). 

        Parameters
        ----------
        database : database object. 
            An instance of an AGOX database class. 
        state : state object
            An instance of the AGOX State object. 

        Returns
        -------
        state object
            The state object with any changes the sampler has made. 
        """
        all_candidates = database.get_all_candidates() + self.transfer_data
        if self.do_check():
            self.setup(all_candidates)
        
    def get_random_member(self):
        if len(self.sample) == 0:
            return None
        index = np.random.randint(low=0, high=len(self.sample))
        member = self.sample[index].copy()
        member.add_meta_information('sample_index', index)
        return member

    def get_all_members(self):
        return [member.copy() for member in self.sample]

    def get_random_member_with_calculator(self):
        if len(self.sample) == 0:
            return None
        index = np.random.randint(low=0, high=len(self.sample))
        member = self.sample[index].copy()
        member.add_meta_information('sample_index', index)
        self.sample[index].copy_calculator_to(member)
        return member    

    def add_transfer_data(self, data):
        """
        Add structures to be considered by the Sampler, such that these structures 
        are passed to 'self.setup' when it is called.
        
        The added structures need a few things:
        1. To be Candidate objects.
        2. To have a energies.

        Feature-update: Should check that the data matches a given environment.

        Parameters
        ----------
        data : list
            List of candidate or atoms objects to be considered by the sampler
        """
        correct_type = np.array([isinstance(dat, CandidateBaseClass) for dat in data]).all()
        if not correct_type:
            raise TypeError('''Only candidate objects can be specified as transfer data, you probably gave ASE Atoms objects.''')
        self.transfer_data = data

    def set_sample(self, data):
        """
        Sets the sample to the given list of candidates, can be used to
        initialize the sample to something specific - e.g. for basin-hopping
        with the MetropolisSampler.
        When using other samplers (such as KMeansSampler) these may be
        overwritten and not considered when making the next sample - if that is
        not the wanted behaviour then use 'add_transfer_data' instead/aswell.

        The added structures need a few things:
        1. To be Candidate objects.
        2. To have a energies.    

        Parameters
        ----------
        data : list
            List of candidate objects.
        """
        correct_type = np.array([isinstance(dat, CandidateBaseClass) for dat in data]).all()
        if not correct_type:
            raise TypeError('''Only candidate objects can be specified as transfer data, you probably gave ASE Atoms objects.''')
        self.sample = data

    def __len__(self):
        return len(self.sample)

    def attach_to_database(self, database):
        from agox.databases.ABC_database import DatabaseBaseClass
        assert isinstance(database, DatabaseBaseClass)
        print(f'{self.name}: Attaching to database: {database}')
        self.attach(database)
