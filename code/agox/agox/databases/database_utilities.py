import numpy as np 
from time import time
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import sqlite3
import os
import sys
import os.path
if sys.version >= '3':
    buffer = memoryview

T2000 = 946681200.0  # January 1. 2000
YEAR = 31557600.0  # 365.25 days


def now():
    """Return time since January 1. 2000 in years."""
    return (time() - T2000) / YEAR

def nothing(a):
    return a

def blob(a):
    return buffer(np.ascontiguousarray(a))


def deblob(buf, dtype=np.float64, shape=None):
    """Convert blob/buffer object to ndarray of correct dtype and shape.
    (without creating an extra view)."""

    if buf is None:
        return None
    if len(buf) == 0:
        array = np.zeros(0, dtype)
    else:
        if len(buf) % 2 == 1 and False:
            # old psycopg2:                                                                                     
            array = np.fromstring(str(buf)[1:].decode('hex'), dtype)
        else:
            array = np.frombuffer(buf, dtype)
        if not np.little_endian:
            array.byteswap(True)
    if shape is not None:
        array.shape = shape
    return array


def export_candidates(dbs, name='run', save=False):
    ebests = []
    emin = float(1E10)
    es = []
    best_structure = None
    for i, d in enumerate(dbs):
        e = []
        ebest = []
        cands = d.get_all_structures_data()
        for c in cands:
            # Extract candidate energy:
            E = c['energy'] if c['energy'] is not None else np.nan
            e.append(E)
            if len(ebest) == 0:
                ebest.append(E)
            else:
                ebest.append(np.nanmin([ebest[-1], e[-1]]))
            if e[-1] < emin:
                emin = e[-1]
                cell = c['cell']
                num = c['type']
                pos = c['positions']
                best_structure = Atoms(symbols = num,
                                       positions = pos,
                                       cell = cell)
                calc = SinglePointCalculator(best_structure, energy = e[-1])
                best_structure.set_calculator(calc)                
        ebests.append(ebest)
        es.append(e)
        if save:
            np.save(name + str(i) + '_Energies.npy',(e, ebest))

    return es, ebests, best_structure


def db_to_ase(cand):
    """

    Converts a database representation (dictionary) of a structur to an ASE atoms object

    Parameters
    ----------
    cand :  database representation of a structure

    Returns
    -------
    struc : ASE Atoms object

    """

    e = cand['energy']
    try:
        f = cand['forces']
    except:
        f = 0
    #f = cand.get('forces',None)
    pos = cand['positions']
    num = cand['type']
    cell = cand['cell']
    struc = Atoms(symbols = num,
                  positions = pos,
                  cell = cell)
    calc = SinglePointCalculator(struc, energy=e, forces=f)
    struc.set_calculator(calc)

    return struc
