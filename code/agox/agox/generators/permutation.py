from agox.generators.ABC_generator import GeneratorBaseClass
from ase.data import covalent_radii
from ase.geometry import get_distances
from ase.io import write
import numpy as np


class PermutationGenerator(GeneratorBaseClass):
    """
    Swap atoms with different chemical symbols
    """
    name='PermutationGenerator'

    def __init__(self, acquisitor=None, max_number_of_swaps=1, rattle_strength=0., 
        use_xy_only=False, ignore_H=False, write_candidates_to_disk=False, 
        **kwargs):
        super().__init__(**kwargs)
        self.acquisitor = acquisitor
        self.max_number_of_swaps = max_number_of_swaps
        self.rattle_strength = rattle_strength
        self.ignore_H = ignore_H
        self.use_xy_only = use_xy_only
        self.write_candidates_to_disk = write_candidates_to_disk

        if self.verbose:
            self.counter = -1

    def get_candidates(self, sampler, environment):
        if self.verbose:
            self.counter +=1
            
        if self.acquisitor is not None:
            candidate = self.acquisitor.get_random_candidate()
        else:
            candidate = sampler.get_random_member()

        if candidate is None:
            return [None]

        if self.verbose and self.write_candidates_to_disk:
            write(f'candidate_{self.counter}.traj', candidate)
            
        template = candidate.get_template()

        number_of_atoms = len(candidate)
        number_of_template_atoms = len(candidate.get_template())
        number_of_non_template_atoms = number_of_atoms - number_of_template_atoms

        symbols = candidate.get_atomic_numbers()[number_of_template_atoms:]
        unique_symbols = np.unique(symbols)
        
        if self.ignore_H:
            unique_symbols = np.array([s for s in unique_symbols if s !=1])
            
        number_of_unique_symbols = len(unique_symbols)
        assert number_of_unique_symbols > 1, 'Cannot be used for single component systems'

        number_of_swaps = np.random.randint(self.max_number_of_swaps) + 1

        for n in range(number_of_swaps):
            symbol_i = unique_symbols[np.random.randint(number_of_unique_symbols)]
            remaining_symbols = np.delete(unique_symbols, [idx for idx in range(number_of_unique_symbols) if unique_symbols[idx]==symbol_i])

            symbol_j = remaining_symbols[np.random.randint(number_of_unique_symbols-1)]

            idx_symbol_i = np.argwhere(symbols==symbol_i).reshape(-1) + number_of_template_atoms
            idx_symbol_j = np.argwhere(symbols==symbol_j).reshape(-1) + number_of_template_atoms

            combinations_ij = np.array(np.meshgrid(idx_symbol_i, idx_symbol_j)).T.reshape(-1,2)

            for row in np.random.permutation(combinations_ij):
                swap_idx_i = row[0]
                swap_idx_j = row[1]
                
                new_positions = candidate.get_positions()
                new_positions[[swap_idx_i, swap_idx_j]] = new_positions[[swap_idx_j, swap_idx_i]]
                if self.use_xy_only:
                    new_positions[swap_idx_i] +=self.pos_add_disk(self.rattle_strength)
                    new_positions[swap_idx_j] +=self.pos_add_disk(self.rattle_strength)
                else:
                    new_positions[swap_idx_i] +=self.pos_add_sphere(self.rattle_strength)
                    new_positions[swap_idx_j] +=self.pos_add_sphere(self.rattle_strength)


                swap_successfull = True
                near_enough_to_other_atoms = False
                for i in (swap_idx_i, swap_idx_j):
                    for other_atom_idx in range(len(candidate)):
                        if other_atom_idx in [swap_idx_i, swap_idx_j]:
                            continue
                        other_atom = candidate[other_atom_idx]
                        
                        covalent_dist = covalent_radii[candidate[i].number] + covalent_radii[other_atom.number]
                        
                        rmin = 0.85 * covalent_dist
                        rmax = 1.15 * covalent_dist
                        tmp = np.linalg.norm(other_atom.position - new_positions[i])
                        if np.linalg.norm(other_atom.position - new_positions[i]) < rmin:
                            swap_successfull = False
                            break
                        if np.linalg.norm(other_atom.position - new_positions[i]) < rmax:
                            near_enough_to_other_atoms = True
                    if not swap_successfull or not near_enough_to_other_atoms:
                        break

                if not swap_successfull or not near_enough_to_other_atoms:
                    continue
                
                candidate.set_positions(new_positions)
                if self.verbose and self.write_candidates_to_disk:
                    write(f'candidate_swap_{n}_{self.counter}.traj', candidate)
                break
            else:
                self.writer('No swaps possible')
                return [None]

        candidate = self.convert_to_candidate_object(candidate, template)

        candidate.add_meta_information('description', self.name)
            
        return [candidate]

                    
    def pos_add_disk(self,rattle_strength):
        """Help function for rattling within a disk
        """
        r = rattle_strength * np.random.rand()**(1/2)
        theta = np.random.uniform(low=0, high=2*np.pi)
        pos_add = r * np.array([np.cos(theta),
                                np.sin(theta),
                                0])
        return pos_add

    def pos_add_sphere(self,rattle_strength):
        """Help function for rattling within a sphere
        """
        r = rattle_strength * np.random.rand()**(1/3)
        theta = np.random.uniform(low=0, high=2*np.pi)
        phi = np.random.uniform(low=0, high=np.pi)
        pos_add = r * np.array([np.cos(theta)*np.sin(phi),
                                np.sin(theta)*np.sin(phi),
                                np.cos(phi)])
        return pos_add        

        
