import numpy as np
from ase.io.utils import cell_to_lines
import matplotlib.pyplot as plt
from ase.data.colors import jmol_colors
from ase.data import covalent_radii, atomic_numbers
from matplotlib.patches import Circle, Patch
from matplotlib.collections import PatchCollection

def plot_atoms(atoms, ax, dim1=0, dim2=1, repeats=[], edgecolor='black', 
    jmol_edgecolor=False, use_facecolor=True, alpha_repeat=0.75, **kwargs):

    # Should sort according to the third dimension:
    sort_dim = [i for i in range(3) if i != dim1 and i != dim2][0]
    dim_array = atoms.positions[:, sort_dim]
    sort_indices = np.argsort(dim_array).flatten()

    for i in sort_indices:
        atom = atoms[i]

        if use_facecolor:
            facecolor = jmol_colors[atom.number]
        else:
            facecolor = 'none'

        if jmol_edgecolor:
            edgecolor = jmol_colors[atom.number]

        c = Circle((atom.position[dim1], atom.position[dim2]), radius=covalent_radii[atom.number], facecolor=facecolor, edgecolor=edgecolor, **kwargs)
        ax.add_patch(c)
    
    ax.axis('equal')
    xlim = ax.get_xlim(); ylim = ax.get_ylim()

    if len(repeats): 
        atoms = make_super_cell(atoms, repeats)
        sort_dim = [i for i in range(3) if i != dim1 and i != dim2][0]
        dim_array = atoms.positions[:, sort_dim]
        sort_indices = np.argsort(dim_array).flatten()
        for i in sort_indices:
            atom = atoms[i]
            c = Circle((atom.position[dim1], atom.position[dim2]), radius=covalent_radii[atom.number], facecolor=jmol_colors[atom.number], edgecolor=edgecolor, zorder=-10, alpha=alpha_repeat)
            ax.add_patch(c)

        ax.set_xlim(xlim); ax.set_ylim(ylim)

def make_super_cell(atoms, repeats):
    for i, r in enumerate(repeats):
        translation_vector = atoms.cell[0, :] * r[0]+ atoms.cell[1, :] * r[1] + atoms.cell[2, :] * r[2]
        temp = atoms.copy()
        temp.positions += translation_vector
        if i == 0:
            supercell = temp
        else:
            supercell += temp
    return supercell

def plot_cell(ax, cell, corner=None, dim1=0, dim2=1, use_combinations=True, **kwargs):

    if corner is None:
        corner = np.array([0, 0, 0])


    combinations = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]
    
    if use_combinations:
        combinations += [(None, None, None), (1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1), # Top
                        (None, None, None), (0, 0, 0), (0, 0, 1),
                        (None, None, None), (1, 0, 0), (1, 0, 1),
                        (None, None, None), (0, 1, 0), (0, 1, 1),
                        (None, None, None), (1, 1, 1), (1, 1, 0)]

    coords = []
    for i, j, k in combinations:
        if i is not None:            
            #p1 = i*cell[0, :] + j*cell[1, :] + k * cell[2, :] + corner
            p = np.dot(np.array([i, j, k]), cell) + corner
            coords.append(p)
        else:
            coords.append((None, None, None))

    coords = np.array(coords)
    ax.plot(coords[:, dim1], coords[:, dim2], **kwargs)

def plot_confinement(atoms, confinement_cell=None, cell_corner=None, axis=None, tight=True):

    if axis == None: 
        sz = 5
        fig, axis = plt.subplots(1, 3, figsize=(3*sz, 1*sz))

    # xy:
    for i, (dim1, dim2) in enumerate([(0, 1), (0, 2), (1, 2)]):

        plot_atoms(atoms, axis[i], dim1=dim1, dim2=dim2)
        plot_cell(axis[i], atoms.cell, np.array([0, 0, 0]), dim1=dim1, dim2=dim2, color='black', linestyle=':')    
        if confinement_cell is not None:
            plot_cell(axis[i], confinement_cell, cell_corner, dim1=dim1, dim2=dim2, color='red', linestyle='--')

    for ax in axis:
        ax.axis('equal')    

    if tight:
        plt.tight_layout()

    return fig, axis




