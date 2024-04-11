from types import new_class
import numpy as np
from ase.geometry import wrap_positions
from agox.helpers.confinement import Confinement

class BoxConstraint(Confinement):

    def __init__(self, confinement_cell=None, confinement_corner=None, indices=None, 
        pbc=[False]*3, **kwargs):
        super().__init__(confinement_cell=confinement_cell, confinement_corner=confinement_corner, 
            indices=indices, pbc=pbc)

        # Soft boundary & force decay.
        self.lower_soft_boundary = 0.05; self.lower_hard_boundary = 0.001
        self.al, self.bl = np.polyfit([self.lower_soft_boundary, self.lower_hard_boundary], [1, 0], 1)
        self.upper_soft_boundary = 0.95; self.upper_hard_boundary = 0.999
        self.au, self.bu = np.polyfit([self.upper_soft_boundary, self.upper_hard_boundary], [1, 0], 1)
    
    def linear_boundary(self, x, a, b):
        return a*x+b

    def adjust_positions(self, atoms, newpositions):
        inside = self.check_confinement(newpositions[self.indices])
        #newpositions[not inside, :] = atoms.positions[not inside]        
        # New positions of those atoms that are not inside (so outside) the box are set inside the box.

        if np.invert(inside).any():
            newpositions[self.indices[np.invert(inside)], :] = wrap_positions(newpositions[self.indices[np.invert(inside)], :],
                                                                              cell=self.confinement_cell, pbc=self.pbc)
            for idx in self.indices[np.invert(inside)]:
                newpositions[idx, self.hard_boundaries] = atoms.positions[idx, self.hard_boundaries]
            
    def adjust_forces(self, atoms, forces):
        C = self.get_projection_coefficients(atoms.positions[self.indices])
        # Because adjust positions does not allow the atoms to escape the box we know that all atoms are witihn the box. 
        # Want to set the forces to zero if atoms are close to the box, this happens if any component of C is close to 0 or 1. 
        for coeff, idx in zip(C, self.indices):
            coeff = np.array([0.5 if p else c for c, p in zip(coeff, self.pbc)])
            if ((coeff < 0) * (coeff > 1)).any():
                forces[idx] = 0 # Somehow the atom is outside, so it is just locked. 
            if (coeff > self.upper_soft_boundary).any():
                forces[idx, self.hard_boundaries] = self.linear_boundary(np.max(coeff), self.au, self.bu) * forces[idx, self.hard_boundaries]
            elif (coeff < self.lower_soft_boundary).any():
                forces[idx, self.hard_boundaries] = self.linear_boundary(np.min(coeff), self.al, self.bl) * forces[idx, self.hard_boundaries]

    def adjust_momenta(self, atoms, momenta):
        self.adjust_forces(atoms, momenta)

    def get_removed_dof(self, atoms):
        return 0
        
    def todict(self):
        return {'name':'BoxConstraint', 
                'kwargs':{'confinement_cell':self.confinement_cell.tolist(), 'confinement_corner':self.confinement_corner.tolist(), 'indices':self.indices.tolist()}}
        
# To work with ASE read/write we need to do some jank shit. 
from ase import constraints 
constraints.__all__.append('BoxConstraint')
constraints.BoxConstraint = BoxConstraint

if __name__ == '__main__':
    from ase import Atoms
    from ase.io import write
    
    B = np.eye(3) * 1
    c = np.array([0, 0, 0])

    atoms = Atoms('H3', positions=[[0.5, 0.5, 0.5], [0.1, 0.1, 0.9], [0.02, 0.9, 0.1]], cell=B)
    write('initial.traj', atoms)

    print(atoms.positions)
    BC = BoxConstraint(B, c, indices=np.array([0, 1, 2]), pbc=[False, False, True])
    atoms.set_constraint([BC])

    print(BC.check_if_inside_box(atoms.positions))
    forces = np.tile(np.array([[1, 1, 1]]), 3).reshape(3, 3).astype(float)
    print(forces)    
    BC.adjust_forces(atoms, forces)
    print(forces)

    atoms.set_positions([[0.5, 0.5, 0.5], [0.1, 0.1, 0.9], [0.02, 1.1, -0.1]])
    print(atoms.positions)
    atoms.set_constraint([])
    write('moved.traj', atoms)
    
    #print(BC.upper_boundary_factor(0.999))
    




        
