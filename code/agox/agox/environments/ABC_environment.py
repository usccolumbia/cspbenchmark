import numpy as np
from abc import ABC, abstractmethod

from agox.utils.constraints.box_constraint import BoxConstraint
from ase.constraints import FixAtoms

from agox.module import Module

class EnvironmentBaseClass(ABC, Module):
    """
    The Environment contains important properties about the envrionment (or conditions) of the global atomisation problem. 
    These are at least: 

    - numbers: The atomic numbers of missings atoms e.g. C2H4 is [1, 1, 1, 1, 6, 6]. 
    - template: An ASE Atoms object that describes the *static* atoms, e.g a surface or part of a molecule.
    """

    def __init__(self, confinement_cell=None, confinement_corner=None, constraints=[], 
                 use_box_constraint=True, box_constraint_pbc=[False]*3, fix_template=True, surname=''):
        Module.__init__(self, surname=surname)

        self.confinement_cell = confinement_cell
        self.confinement_corner = confinement_corner
        self.constraints = constraints 
        self.use_box_constraint = use_box_constraint
        self.box_constraint_pbc = box_constraint_pbc
        self.fix_template = fix_template
        
    @abstractmethod
    def get_template(self, **kwargs):
        pass

    @abstractmethod    
    def get_numbers(self, numbers, **kwargs):
        pass
        
    @abstractmethod
    def environment_report(self):
        pass 

    def get_missing_indices(self):
        return np.arange(len(self._template), len(self._template)+len(self._numbers))

    def get_confinement_cell(self, distance_to_edge=3):
        if self.confinement_cell is not None:
            confinement_cell = self.confinement_cell
        elif self._template.pbc.all() is False:
            confinement_cell = self._template.get_cell().copy() - np.eye(3) * distance_to_edge * 2
        else: 
            # Find the directions that are not periodic: 
            confinement_cell = self._template.get_cell().copy() - np.eye(3) * distance_to_edge * 2
            directions = np.argwhere(self._template.pbc == True)
            for d in directions:
                confinement_cell[d, :] = self._template.get_cell()[d, :]

        return confinement_cell

    def get_confinement_corner(self, distance_to_edge=3):
        if self.confinement_cell is not None:
            confinement_corner = self.confinement_corner
        elif self._template.pbc.all() is False:
            confinement_corner = np.ones(3) * distance_to_edge
        else: 
            # Find the directions that are not periodic: 
            confinement_corner = np.ones(3) * distance_to_edge
            directions = np.argwhere(self._template.pbc == True)
            for d in directions:
                confinement_corner[d] = 0
        return confinement_corner

    def get_confinement(self):
        return {'confinement_cell':self.confinement_cell, 'confinement_corner':self.confinement_corner}

    def get_box_constraint(self):
        confinement_cell = self.get_confinement_cell()
        confinement_corner = self.get_confinement_corner()
        return BoxConstraint(confinement_cell=confinement_cell, confinement_corner=confinement_corner,
                             indices=self.get_missing_indices(), pbc=self.box_constraint_pbc)

    def get_constraints(self):
        constraints = []
        if self.use_box_constraint:
            constraints += [self.get_box_constraint()]
        if self.fix_template:
            constraints += [self.get_template_constraint()]
        return constraints + self.constraints 

    def get_template_constraint(self):
        return FixAtoms(indices=np.arange(len(self.get_template())))

