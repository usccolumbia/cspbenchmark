import pickle
import numpy as np 
from time import sleep
from ase import Atoms
import sqlite3
import os
import sys
import os.path

from .database_utilities import *
from .ABC_database import DatabaseBaseClass
from copy import deepcopy
from collections import OrderedDict

# Should match init_statements!!
from agox.candidates import StandardCandidate

class Database(DatabaseBaseClass):
    """ Database module """

    init_statements = ["""create table structures (
    id integer primary key autoincrement,
    ctime real,
    positions blob,
    energy real,
    type blob,
    cell blob,
    forces blob, 
    pbc blob,
    template_indices blob,
    iteration int
    )""", 

    """CREATE TABLE text_key_values (
    key TEXT,
    value TEXT,
    id INTEGER,
    FOREIGN KEY (id) REFERENCES systems(id))""",

    """CREATE TABLE float_key_values (
    key TEXT,
    value REAL,
    id INTEGER,
    FOREIGN KEY (id) REFERENCES systems(id))""", 

    """CREATE TABLE int_key_values (
    key TEXT,
    value INTEGER,
    id INTEGER,
    FOREIGN KEY (id) REFERENCES systems(id))""", 
    
    """CREATE TABLE boolean_key_values (
    key TEXT,
    value INTEGER,
    id INTEGER,
    FOREIGN KEY (id) REFERENCES systems(id))""", 
    
    """CREATE TABLE other_key_values (
    key TEXT,
    value BLOB,
    id INTEGER,
    FOREIGN KEY (id) REFERENCES systems(id))"""]

    #unpack_functions = [nothing, nothing, deblob, nothing, deblob, deblob, deblob, deblob, deblob, nothing]
    #pack_functions = [nothing, nothing, blob, nothing, blob, blob, blob, blob, nothing]
    
    # Pack: Positions, energy, type, cell, forces, pbc, template_indices, iteration
    # Unpack: ID, time, -//-
    pack_functions = [blob, nothing, blob, blob, blob, blob, blob, nothing]
    unpack_functions = [nothing, nothing, deblob, nothing, deblob, deblob, deblob, deblob, deblob, nothing]

    name = 'Database'

    def __init__(self, filename='db.db', initialize=False, verbose=False, write_frequency=1, call_initialize=True, 
                store_meta_information=True, **kwargs):
        super().__init__(**kwargs)

        # File-based stuff:
        self.filename = filename
        if initialize and os.path.exists(filename):
            os.remove(filename)
        self.con = sqlite3.connect(filename, timeout=600)
        self.write_frequency = write_frequency

        # Important that this matches the init_statements list. 
        self.storage_keys = ['positions', 'energy', 'type', 'cell', 'forces', 'pbc', 'template_indices', 'iteration']

        # Memory-based stuff:        
        self.candidate_instanstiator = StandardCandidate
        self.candidate_energies = []
        
        # Whether or not to save and retrieve meta information from the database.
        self.store_meta_information = store_meta_information
        self.meta_dict_list = []

        if call_initialize:
            self._initialize()

    ####################################################################################################################
    # Memory-based methods:
    ####################################################################################################################

    def store_candidate(self, candidate, accepted=True, write=True, dispatch=True):
        # Needs some way of handling a dummy candidate, probably boolean argument.
        if accepted:
            self.candidates.append(candidate)
            self.candidate_energies.append(candidate.get_potential_energy())
        if write:
            self.write(candidate)
        # if dispatch:
        #     self.dispatch_to_observers(self, state)

    def get_all_candidates(self, **kwargs):
        all_candidates = []
        for candidate in self.candidates:
            all_candidates.append(candidate)
        return all_candidates

    def get_most_recent_candidate(self):
        if len(self.candidates) > 0:
            candidate = deepcopy(self.candidates[-1])
        else:
            candidate = None
        return candidate

    def get_recent_candidates(self, number):
        return [deepcopy(candidate) for candidate in self.candidates[-number:]]

    def get_best_energy(self):
        return np.min(self.candidate_energies)
        
    ####################################################################################################################
    # File-based methods:
    ####################################################################################################################

    def _init_storage(self):
        self.positions = []
        self.atom_numbers = []
        self.energies = []
        self.cells = []
        self.forces = []
        self.pbc = []
        self.template_indices = []

        self.storage_dict = OrderedDict()
        for key in self.storage_keys:
            self.storage_dict[key] = []
        
        self.number_of_rows = 0
        self.meta_dict_list = []
        
    def _initialize(self):
        # Check if structures are in the database, otherwise initialize tables
        cur = self.con.execute(
            'select count(*) from sqlite_master where name="structures"')
        try:
            if cur.fetchone()[0] == 0:
                for statement in self.init_statements:
                    self.con.execute(statement)
                self.con.commit()
        except sqlite3.OperationalError:
            pass
        self._init_storage()
        self.con.row_factory = sqlite3.Row

    def store_information(self, candidate):
        if candidate is not None:            

            e = candidate.calc.results.get('energy', 0)
            f = candidate.calc.results.get('forces', np.zeros(candidate.positions.shape))
            
            self.storage_dict['energy'].append(e)
            self.storage_dict['forces'].append(f)
            self.storage_dict['positions'].append(np.array(candidate.positions, dtype=np.float64))
            self.storage_dict['type'].append(np.array(candidate.numbers, dtype=np.float64))
            self.storage_dict['cell'].append(np.array(candidate.cell, dtype=np.float64))
            self.storage_dict['pbc'].append(np.array(candidate.pbc.astype(int), dtype=np.float64))
            self.storage_dict['template_indices'].append(np.array(candidate.get_template_indices(), dtype=np.float64))
            self.storage_dict['iteration'].append(self.get_iteration_counter())
            self.number_of_rows += 1

            if self.store_meta_information:
                self.meta_dict_list.append(candidate.meta_information)

    def get_row_to_write(self, index):
        row = [now()]
        for (key, value), func in zip(self.storage_dict.items(), self.pack_functions):
            row.append(func(value[index]))

        return tuple(row)

    def get_last_id(self, cur):
        cur.execute('SELECT seq FROM sqlite_sequence WHERE name="structures"')
        result = cur.fetchone()
        if result is not None:
            id = result[0]
            return id
        else:
            return 0

    def write(self, candidate=None, force_write=False, already_stored=False):

        if not already_stored: 
            self.store_information(candidate)

        if self.number_of_rows == self.write_frequency or force_write:
            try:
                for row_index in range(self.number_of_rows):
                    cur = self.con.cursor()
                    row = self.get_row_to_write(row_index)
                    q = 'NULL,' + ', '.join('?' * len(row))

                    cur.execute('insert into structures values ({})'.format(q), row)
                    
                    self.last_used_id = self.get_last_id(cur)

                    if self.store_meta_information:
                        cur = self.con.cursor()
                        id = self.last_used_id
                        self.write_key_value_pairs(cur, self.meta_dict_list[row_index], id=id)
                    self.con.commit()                    

            except sqlite3.OperationalError as error:
                print('Encountered error: {}'.format(error))
                sleep(1)
                self.write(candidate=candidate, force_write=force_write, already_stored=True)
            self._init_storage()          

    def db_to_atoms(self, structure):
        """

        Converts a database representation (dictionary) of a structure to an ASE atoms object

        Parameters
        ----------
        structure :  database representation of a structure

        Returns
        -------
        struc : ASE Atoms object

        """
        e = structure['energy']
        f = structure.get('forces', 0).reshape(-1, 3)
        pos = structure['positions']
        num = structure['type']
        cell = structure['cell']
        pbc = structure.get('pbc', None)

        atoms = Atoms(symbols = num,
                    positions = pos,
                    cell = cell, 
                    pbc=pbc)        

        calc = SinglePointCalculator(atoms, energy=e, forces=f)
        atoms.set_calculator(calc)
        return atoms

    def db_to_candidate(self, structure, meta_dict=None):
        e = structure['energy']
        f = structure.get('forces', 0).reshape(-1, 3)
        pos = structure['positions']
        num = structure['type']
        cell = structure['cell']
        pbc = structure.get('pbc', None)
        template_indices = structure.get('template_indices', None)
        
        if hasattr(self, 'template') and template_indices is None:
            template = self.template
        else:
            template = None            

        candidate = self.candidate_instanstiator(symbols = num, positions = pos, cell = cell, pbc=pbc, template=template, template_indices=template_indices)        
        calc = SinglePointCalculator(candidate, energy=e, forces=f)
        candidate.set_calculator(calc)

        if self.store_meta_information:
            if meta_dict is None:
                restored_dict = self.read_key_value_pairs(id=structure['id'])
                for key, value in restored_dict.items():
                    candidate.add_meta_information(key, value)
            else:
                candidate.meta_information.update(meta_dict)

        candidate.add_meta_information('iteration', structure.get('iteration', 0))
        candidate.add_meta_information('database_id', structure['id'])

        return candidate

    def get_data_from_row(self, row):
        d = {}
        for key, value, func in zip(row.keys(), row, self.unpack_functions):
            d[key] = func(value)

        d['positions'] = d['positions'].reshape(-1, 3)
        d['cell'] = d['cell'].reshape(3, 3)

        return d

    def get_all_structures_data(self):
        cursor = self.con.execute("SELECT * from structures")
        structures = []
        for row in cursor.fetchall():
            d = self.get_data_from_row(row)
            structures.append(d)
        return structures

    def get_structure_data(self, i):
        t = (str(int(i)),)
        cursor = self.con.execute('SELECT * from structures WHERE id=?',t)
        row = cursor.fetchone()
        d = self.get_data_from_row(row)    
        return d        

    def get_all_energies(self):
        cursor = self.con.execute("SELECT energy from structures")
        energies = []
        for row in cursor.fetchall():
            energies.append(row['energy'])
        return np.array(energies)

    def get_best_structure(self):
        structure_data = self.get_all_structures_data()
        energies = [c['energy'] for c in structure_data]
        idx = energies.index(min(energies))
        best_struc = self.db_to_atoms(structure_data[idx])   
        #print('the code comes here ...................................')     
        return best_struc
        
    def get_structure(self, index):
        cand = self.get_structure_data(index)
        struc = self.db_to_atoms(cand)

        return struc

    ####################################################################################################################
    # Meta-information related.
    ####################################################################################################################

    def restore_to_memory(self):
        strucs = self.get_all_structures_data()
        if self.store_meta_information:
            all_meta_info = self.read_all_key_value_pairs()
        else:
            all_meta_info = {}

        candidates = []
        for struc in strucs:            
            candidates.append(self.db_to_candidate(struc, meta_dict=all_meta_info.get(struc['id'], None)))
        self.candidates = candidates
        self.candidate_energies = [atoms.get_potential_energy() for atoms in candidates]
        print('asdffgqegeg')

    def restore_to_trajectory(self):
        strucs = self.get_all_structures_data()
        atoms_objects = []
        for structure in strucs:
            atoms_objects.append(self.db_to_atoms(structure))
        return atoms_objects

    ####################################################################################################################
    # Meta-information related.
    ####################################################################################################################

    def write_key_value_pairs(self, cur, dictionary, id=0):            
        text_pairs = []
        float_pairs = []
        int_pairs = []
        bool_pairs = []
        other_pairs = []

        for key, value in dictionary.items():
            if type(value) == str:
                text_pairs += [(key, value, id)]
            elif type(value) == float:
                float_pairs += [(key, value, id)]
            elif type(value) == int:
                int_pairs += [(key, value, id)]
            elif type(value) == bool:
                bool_pairs += [(key, value, id)]
            else:
                value_converted =  pickle.dumps(value, pickle.HIGHEST_PROTOCOL)
                other_pairs += [(key, value_converted, id)]
        
        cur.executemany('INSERT INTO text_key_values VALUES (?, ?, ?)', text_pairs)
        cur.executemany('INSERT INTO float_key_values VALUES (?, ?, ?)', float_pairs)
        cur.executemany('INSERT INTO int_key_values VALUES (?, ?, ?)', int_pairs)
        cur.executemany('INSERT INTO boolean_key_values VALUES (?, ?, ?)', bool_pairs)        
        cur.executemany('INSERT INTO other_key_values VALUES (?, ?, ?)', other_pairs)

    def read_key_value_pairs(self, id=0):
        dict_recreation = {}
        tables = ['text_key_values', 'float_key_values', 'int_key_values', 'boolean_key_values', 'other_key_values']
        functions = [nothing, nothing, nothing, bool, pickle.loads]

        cur = self.con.cursor()
        for table, func in zip(tables, functions):
            cur.execute(f'SELECT * FROM {table} WHERE id=?', (id,))
            A = cur.fetchall()
            for key, value, _ in A:
                try:
                    converted_value = func(value)
                except:
                    converted_value = deblob(value)

                dict_recreation[key] = converted_value

        return dict_recreation

    def read_all_key_value_pairs(self):
        cursor = self.con.cursor()

        tables = ['text_key_values', 'float_key_values', 'int_key_values', 'boolean_key_values', 'other_key_values']
        functions = [nothing, nothing, nothing, bool, pickle.loads]

        dict_of_dicts = {} # Uses the id as the key to the meta information dict for each candidate. (Id from structures table).

        for table, func in zip(tables, functions):
            cursor.execute(f'SELECT * FROM {table}')
            rows = cursor.fetchall()

            for key, value, id in rows:
                try:
                    converted_value = func(value)
                except:
                    converted_value = deblob(value)

                if id in dict_of_dicts.keys():
                    dict_of_dicts[id][key] = converted_value
                else:
                    dict_of_dicts[id] = {key:converted_value}

        return dict_of_dicts

    def update_meta_information(self, meta_dict, database_index, memory_index):
        """

        Update meta-information of the candidate at database_index 'id' in the structures
        table. 

        Parameters
        ----------
        meta_dict : dict
            Dict containing the information to update - if keys are present that 
            are already present in the database they are overwritten with the new
            values and if new keys are present they are added to the database. 
        index : int
            The index (id) of the structure in the structures table. 
        """
        if database_index is not None:
            cur = self.con.cursor()
            self.write_key_value_pairs(cur, meta_dict, id=database_index)
            self.con.commit()
        if memory_index is not None:
            self.candidates[memory_index].meta_information.update(meta_dict)
