import numpy as np
from ase.data import covalent_radii
from abc import ABC, abstractmethod
from agox.candidates import StandardCandidate
from ase.geometry import get_distances                         
from agox.observer import Observer
from agox.writer import Writer, agox_writer

from agox.helpers.confinement import Confinement

dimensionality_angles = {
                        3:{'theta':[0, 2*np.pi], 'phi':[0, np.pi]},
                        2:{'theta':[0, 2*np.pi], 'phi':[np.pi/2, np.pi/2]},
                        1:{'theta':[0, 0], 'phi':[np.pi/2, np.pi/2]}
                        }

class GeneratorBaseClass(ABC, Observer, Writer, Confinement):

    def __init__(self, confinement_cell=None, confinement_corner=None, c1=0.75, c2=1.25, 
                use_mic=True, environment=None, sampler=None, gets={}, sets={'set_key':'candidates'}, order=2, 
                verbose=True, use_counter=True, prefix='', surname=''):
        """
        use_mic: Whether to use minimum image convention when checking distances. 
                 If using a periodic cell but a smaller confinement cell set this to False to speed
                 up generation time! 
        sampler/enivronment: Needs to be set when not used with a collector!
        """
        Observer.__init__(self, sets=sets, gets=gets, order=order, surname=surname)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)
        Confinement.__init__(self, confinement_cell=confinement_cell, confinement_corner=confinement_corner)

        self.c1 = c1 # Lower limit on bond lengths. 
        self.c2 = c2 # Upper limit on bond lengths. 
        self.use_mic = use_mic
        self.environment = environment
        self.sampler = sampler
        self.candidate_instanstiator = StandardCandidate

        if self.environment is not None:
            self.plot_confinement(environment)

        self.add_observer_method(self.generate,
                                 sets=self.sets[0], gets=self.gets[0], order=self.order[0],
                                 handler_identifier='AGOX')

    @abstractmethod
    def get_candidates(self, sampler, environment):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    def __call__(self, sampler, environment):
        return self.get_candidates(sampler, environment)
    
    ####################################################################################################################
    # Convenience methods:
    ####################################################################################################################

    def convert_to_candidate_object(self, atoms_type_object, template):
        candidate =  self.candidate_instanstiator(template=template, positions=atoms_type_object.positions, numbers=atoms_type_object.numbers, 
                                          cell=atoms_type_object.cell)

        candidate.add_meta_information('generator', self.name)

        return candidate

    def check_new_position(self, candidate, new_position, number, skipped_indices=[]):
        """
        Checks if new positions is not too close or too far to any other atom. 

        Probably not be the fastest implementation, so may be worth it to optimize at some point. 
        """
        state = True
        succesful = False
        for i in range(len(candidate)):
            if i in skipped_indices:
                continue

            covalent_dist_ij = covalent_radii[candidate[i].number] + covalent_radii[number]
            rmin = self.c1 * covalent_dist_ij
            rmax = self.c2 * covalent_dist_ij

            if self.use_mic:
                vector, distance = get_distances(new_position, candidate.positions[i], cell=candidate.cell, pbc=candidate.pbc)
            else:
                distance = np.linalg.norm(candidate.positions[i]-new_position)
            if distance < rmin: # If a position is too close we should just give up. 
                return False
            elif not distance > rmax: # If at least a single position is not too far we have a bond.
                succesful = True
        return succesful * state

    def get_sphere_vector(self, atomic_number_i, atomic_number_j):
        """
        Get a random vector on the sphere of appropriate radii. 

        Behaviour changes based on self.dimensionality: 
        3: Vector on sphere. 
        2: Vector on circle (in xy)
        1: Vector on line (x)
        """
        covalent_dist_ij = covalent_radii[atomic_number_i] + covalent_radii[atomic_number_j]
        rmin = self.c1 * covalent_dist_ij
        rmax = self.c2 * covalent_dist_ij
        r = np.random.uniform(rmin**self.dimensionality, rmax**self.dimensionality)**(1/self.dimensionality)
        return self.get_displacement_vector(r)

    def get_displacement_vector(self, radius):
        theta = np.random.uniform(*dimensionality_angles[self.dimensionality]['theta'])
        phi = np.random.uniform(*dimensionality_angles[self.dimensionality]['phi'])
        displacement = radius * np.array([np.cos(theta)*np.sin(phi),
                                          np.sin(theta)*np.sin(phi),
                                          np.cos(phi)])
        return displacement

    def get_box_vector(self):
        return self._get_box_vector(self.confinement_cell, self.confinement_corner) # From confinement-class.

    ####################################################################################################################
    # Observer functions
    ####################################################################################################################

    def start_candidate(self):
        """
        This method generates a candidate using the start generator, which allows other generators 
        to kick-start the sampler. 
        """
        from agox.generators import RandomGenerator
        return RandomGenerator(confinement_cell=self.confinement_cell, confinement_corner=self.confinement_corner, 
                c1=self.c1, c2=self.c2, use_mic=self.use_mic)(self.sampler, self.environment)

    @agox_writer
    @Observer.observer_method
    def generate(self, state):
        candidates = self.get_candidates(self.sampler, self.environment)
        if candidates[0] is None and self.sampler is not None and len(self.sampler) == 0:
            candidates = self.start_candidate()
            self.writer('Fall-back to start generator, generated {} candidate '.format(len(candidates)))
        state.add_to_cache(self, self.set_key, candidates, mode='a')
        
    def plot_confinement(self, environment):
        from agox.helpers.plot_confinement import plot_confinement
        import matplotlib.pyplot as plt
        from agox.utils.matplotlib_utils import use_agox_mpl_backend; use_agox_mpl_backend()

        if self.confined:
            fig, ax = plot_confinement(environment.get_template(), self.confinement_cell, self.confinement_corner)
            plt.savefig(f'confinement_plot_{self.name}.png')
            plt.close()
