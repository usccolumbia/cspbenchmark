import os
from time import sleep
import numpy as np
from ase.io import write, read
from agox.writer import agox_writer
from agox.samplers.metropolis import MetropolisSampler

from ase.calculators.singlepoint import SinglePointCalculator

from agox.observer import Observer 

def safe_write(filename, atoms):
    write(filename=filename+'.lock', images=atoms, format='traj')
    os.rename(filename+'.lock', filename)

class ParallelTemperingSampler(MetropolisSampler):
    """
    Written to work with ConcurrentDatabase, no guarantee that it works in any other setting. 
    """
    
    name = 'ParallelTemperingSampler'

    def __init__(self, temperatures=[], database=None, swap_frequency=10, gets=[{'get_key':'evaluated_candidates'}, {}], 
                sets=[{}, {}], swap_order=2, **kwargs):
        super().__init__(gets=gets, sets=sets, **kwargs)
        self.swap_frequency = swap_frequency
        self.temperatures = temperatures
        self.temperature = temperatures[database.worker_number]
        self.swap_order = swap_order
        self.swap_func = self.metropolis_hastings_swap
        self.most_recently_accepted = {worker_number:None for worker_number in range(database.total_workers)}
        self.most_recently_accepted_iteration = {worker_number:0 for worker_number in range(database.total_workers)}

        self.add_observer_method(self.swap_candidates, sets=self.sets[1], gets=self.gets[1], order=self.swap_order, handler_identifier='database')

        self.attach_to_database(database)

    @agox_writer
    @Observer.observer_method
    def swap_candidates(self, database, state):
        """
        Assumes that the database is synced. 
        """
        
        if self.decide_to_swap(database):
            for candidate in database.candidates:

                worker_number = candidate.get_meta_information('worker_number')
                iteration = candidate.get_meta_information('iteration')
                accepted = candidate.get_meta_information('accepted')
                
                if accepted:
                    if iteration > self.most_recently_accepted_iteration[worker_number]:
                        self.most_recently_accepted[worker_number] = candidate
                        self.most_recently_accepted_iteration[worker_number] = candidate.get_meta_information('iteration')

            if self.verbose:
                energies = [self.most_recently_accepted[c].get_potential_energy() \
                            if self.most_recently_accepted[c] is not None else 0 for c in self.most_recently_accepted]
                if database.worker_number == 0:
                    text = 'PARALLEL TEMPERING:'  + f'{self.temperature:8.3f}, ' + ','.join([f'{e:8.3f}' for e in energies])
                else:
                    text = 'PARALLEL TEMPERING (not main worker):'
                self.writer(text)

            self.swap_func(database)
        
    def metropolis_hastings_swap(self, database):
        worker_number = database.worker_number
        total_workers = database.total_workers
        iteration = self.get_iteration_counter()

        # I abuse the filenames a bit here: 
        filename = database.filename[:-3] + '_swap_iteration_{}_worker_{}.traj'

        if worker_number == 0:
            # This one does the calculation and 'broadcasts' to the others over disk. 

            # Starting from the bottom: 
            for i in range(total_workers-1):                
                C_i = self.most_recently_accepted[i]
                C_j = self.most_recently_accepted[i+1]

                E_i = C_i.get_potential_energy()
                E_j = C_j.get_potential_energy()

                beta_i = 1/self.temperatures[i]
                beta_j = 1/self.temperatures[i+1]

                P = np.min([1, np.exp((beta_i-beta_j)*(E_i-E_j))])

                r = np.random.rand()

                if r < P:
                    self.writer('Swapped {} with {}'.format(i, i+1))
                    self.most_recently_accepted[i], self.most_recently_accepted[i+1] = self.most_recently_accepted[i+1], self.most_recently_accepted[i]
                    self.most_recently_accepted_iteration[i], self.most_recently_accepted_iteration[i+1] = iteration, iteration
                    
                else:
                    self.writer('Did not swap {} for {}'.format(i, i+1))

            # Write the candidates:
            for wn in range(1, total_workers):
                atoms = self.most_recently_accepted[wn]
                atoms.info = atoms.meta_information
                safe_write(filename.format(iteration, wn), atoms)
            self.chosen_candidate = self.most_recently_accepted[worker_number]

        else: 
            while not os.path.exists(filename.format(iteration, worker_number)):
                sleep(1)
            
            chosen_atoms = read(filename.format(iteration, worker_number))
            
            self.sample[0] = self.convert_to_candidate_object(chosen_atoms, self.sample[0].template)
            self.sample[0].meta_information = chosen_atoms.info
            scp = SinglePointCalculator(self.sample[0], energy=chosen_atoms.get_potential_energy(), forces=chosen_atoms.get_forces())
            self.sample[0].set_calculator(scp)

            os.remove(filename.format(iteration, worker_number))

    def convert_to_candidate_object(self, atoms_type_object, template):
        candidate =  self.candidate_instanstiator(template=template, positions=atoms_type_object.positions, numbers=atoms_type_object.numbers, 
                                          cell=atoms_type_object.cell)
        return candidate

    def decide_to_swap(self, database):
        return (self.get_iteration_counter() % self.swap_frequency == 0) * (database.total_workers > 1)    
