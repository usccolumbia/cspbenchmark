"""
Automagic ASLA analysis notebook generator. 


To-do: 
Add animation for histogram/structure interaction. DONE
Quickview: Structure, succes, energy, histogram in one figure ??
Easy saving of figures.
Over-write protection.
Run-details: Interactive to see the structure, energy etc, slider for iteration and restart, dropdown for directory. 
Greedy builds: Interactive plot that shows 10 greedy-policy builds with slider for restart and iteration and dropdown for directory.
Interactive succes: Selector for which directories to plot (on/off tick box)

"""

import nbformat as nbf
import pathlib
from argparse import ArgumentParser
import subprocess

# Argument Parser stuff:
parser = ArgumentParser()
parser.add_argument('-d', '--directories', nargs='+', type=str) # List - Directories
parser.add_argument('-n', '--name', type=str, default='automagic_analysis') # Name of notebook file (without .ipynb)
parser.add_argument('-ar', '--auto_run', action='store_true') # Autorun the notebook so all cells have been run when first opened.
parser.add_argument('-ao', '--auto_open', action='store_false') # Autooepn notebook in VS code


args = parser.parse_args()
directories = args.directories
auto_run = args.auto_run
auto_open = args.auto_open
name = args.name

# Need to massage the input of directories a little bit:
for i in range(len(directories)):
    directories[i] = "'" + directories[i] + "'"

# Convenience settings:
figsize = 'figsize = (7, 7)'

# Create notebook object
nb = nbf.v4.new_notebook()

# Add functions:
markdown = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# List to hold cell contents
cells = nb['cells'] = []

def add_cell(content_string, add_function):
    cells.append(add_function(content_string))

# Create initial text
this_file = pathlib.Path(__file__).absolute()
headline = """
# Automagic AGOX Analysis
This is an automatically generated AGOX analysis notebook.\n 
Notebook generation is defined in this file: {}
""".format(this_file)
#cells.append(markdown(headline))
add_cell(headline, markdown)


# Import statements % basic settings
import_block = """import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from agox.utils.batch_analysis import Analysis
from agox.utils.jupyter_interactive import InteractiveStructureHistogram, InteractiveSuccessStats, InteractiveEnergy, sorted_species"""
#cells.append(code(import_block))
add_cell(import_block, code)

# Load block: 
load_headline = """## Scan and read db-files in the directories and calculate CDF.
This is the block that takes the most times, avoid rerunning unless changes have been made that require re-reading the 
directories, such as adding additional ones. """
add_cell(load_headline, markdown)

load_block = """analysis = Analysis()
force_reload = False
"""
for directory in directories:
    load_block += 'analysis.add_directory({}, force_reload=force_reload)\n'.format(directory)
load_block += 'analysis.compile_information()\n'
load_block += 'analysis.calculate_CDF()'

#cells.append(code(load_block))
add_cell(load_block, code)

# Success statistics block:
succes_headline = """
## Analysis
"""
add_cell(succes_headline, markdown)


success_block = """ISS = InteractiveSuccessStats(analysis)
out = widgets.interactive_output(ISS.update_plot, ISS.widget_dict)
widgets.HBox([widgets.VBox(ISS.widget_list), out])""".format(figsize)
add_cell(success_block, code)

energy_int_block = """IE = InteractiveEnergy(analysis)
out = widgets.interactive_output(IE.update_plot, IE.widget_dict)
widgets.HBox([widgets.VBox(IE.widget_list), out])"""
add_cell(energy_int_block, code)


interactive_struct_hist = """ISH = InteractiveStructureHistogram(analysis)
num_structures = np.sum(analysis.restarts)
index = widgets.IntSlider(min=0, max=num_structures-1, value=0, description='Index')
widgets.interactive(ISH.update_plot, index=index)"""
add_cell(interactive_struct_hist, code)


structure_viewer = """from ase.visualize import view
structures, energies = analysis.get_best_structures()
structures = [sorted_species(atoms) for atoms in structures]
view(structures, viewer='ngl')"""
add_cell(structure_viewer, code)

# Write notebook:
notebook_path = name + '.ipynb'
nbf.write(nb, notebook_path)

if auto_run:
    out = subprocess.check_output('jupyter nbconvert --execute --inplace {}'.format(notebook_path), shell=True)


# This will only work with appropriately configured VS-code installation.
if auto_open:
    subprocess.run('code {}'.format(notebook_path), shell=True)



# Structure plots:
# structure_headline = """
# ## Found structures
# """
# add_cell(structure_headline, markdown)

# structure_block = """
# sz = 3
# nrows = 2
# ncols = 5
# structure_fig, structure_axes = plt.subplots(nrows, ncols, figsize = (ncols*sz, nrows*sz))
# structure_axes = structure_axes.flatten()
# best_structures, best_energies = asla_analysis.get_best_structures()

# # Weak filtering, so do deeper analysis based on grid features if you need highly certain results. 
# unique_energies, unique_idx = np.unique(best_energies, return_index=True)


# for j in range(nrows*ncols):
#     i = unique_idx[j]
#     asla_analysis.plot_structure_nice(structure_axes[j], best_structures[i])

#     structure_axes[j].set_title('{:2d}: E = {:6.2f}'.format(j, best_energies[j]), fontsize=15)


# plt.tight_layout()
# plt.show()"""
# add_cell(structure_block, code)