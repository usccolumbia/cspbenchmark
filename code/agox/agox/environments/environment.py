import numpy as np
from agox.environments.ABC_environment import EnvironmentBaseClass
from ase.atoms import symbols2numbers
from ase.symbols import Symbols

class Environment(EnvironmentBaseClass):

    def __init__(self, template, numbers=None, symbols=None, **kwargs):
        super().__init__(**kwargs)

        # Both numbers and symbols cannot be specified:
        assert (numbers is not None) is not (symbols is not None) # XOR

        if numbers is not None:
            self._numbers = numbers
        elif symbols is not None:
            self._numbers = symbols2numbers(symbols)
        
        self._template = template

        self.environment_report()


    def get_template(self):
        return self._template.copy()

    def set_template(self, template):
        self._template = template

    def set_numbers(self, numbers):
        self._numbers = numbers

    def get_numbers(self):
        return self._numbers.copy()

    def get_missing_types(self):
        return np.sort(np.unique(self.get_numbers()))

    def get_all_types(self):
        return list(set(list(self._template.numbers) + list(self.get_numbers())))

    def get_identifier(self):
        return self.__hash__()

    def get_missing_indices(self):
        return np.arange(len(self._template), len(self._template)+len(self._numbers))

    def get_all_numbers(self):
        all_numbers = np.append(self.get_numbers(), self._template.get_atomic_numbers())
        return all_numbers

    def get_all_species(self):
        return list(Symbols(self.get_all_types()).species())

    def match(self, candidate):
        cand_numbers = candidate.get_atomic_numbers()
        env_numbers = self.get_all_numbers()

        stoi_match = (np.sort(cand_numbers) == np.sort(env_numbers)).all() # Not very efficient, but does it matter? Should pobably only use this function for debugging.
        template_match = (candidate.positions[0:len(candidate.template)] == self._template.positions).all()
        return stoi_match*template_match

    def __hash__(self):
        feature = tuple(self.get_numbers()) + tuple(self._template.get_atomic_numbers()) + tuple(self._template.get_positions().flatten().tolist())
        return hash(feature)    

    def environment_report(self):
        from agox.writer import pretty_print, header_print

        header_print('Environment Properties')
        tab = '    '

        missing_numbers = self.get_numbers()

        pretty_print('Atoms in search:')  
        for number in np.unique(missing_numbers):
            symbols_object = Symbols([number])
            specie = symbols_object.species().pop()
            count = np.count_nonzero(missing_numbers == number)
            pretty_print(tab + f'{specie} = {count}')

        total_symbols = Symbols(self.get_all_numbers())
        pretty_print(f'Template formula: {self._template.get_chemical_formula()}')
        pretty_print(f'Full formula: {total_symbols.get_chemical_formula()}')

        pretty_print('Cell:')
        for cell_vec in self._template.get_cell():
            pretty_print(tab + '{:4.2f} {:4.2f} {:4.2f}'.format(*cell_vec))
        pretty_print('Periodicity:')
        pretty_print(tab + '{} {} {}'.format(*self._template.pbc))

        pretty_print(f'Box constraint: {self.use_box_constraint}')
        if self.use_box_constraint:
            assert self.confinement_cell is not None
            assert self.confinement_corner is not None
            pretty_print('Confinement corner')
            pretty_print(tab + '{:4.2f} {:4.2f} {:4.2f}'.format(*self.confinement_corner))
            pretty_print('Confinement cell:')
            for cell_vec in self.confinement_cell:
                pretty_print(tab + '{:4.2f} {:4.2f} {:4.2f}'.format(*cell_vec))

        header_print('')
    