import numpy as np
from ase.calculators.calculator import Calculator
import os
from ase.io import read, write, Trajectory
import pickle

"""
If the GPAW Calculator throws unexpected errors it is likely because Ray has somehow 
been imported on its process, look at the imports that are happening for it and 
make sure Ray is not present in any file that is imported (even indirectly through 
__init__.py files).
"""

calc_script = """
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from agox.utils.constraints.box_constraint import BoxConstraint
from agox.main import AGOX

from ase.io import read, write, Trajectory
from gpaw import GPAW, PW
import sys
import argparse
import pickle

# Input stuff:
parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('output_file')
parser.add_argument('settings_file')

# Read the input file: 
args = parser.parse_args()

t = read(args.input_file)

# Setup the GPAW calculator:
with open(args.settings_file, 'rb') as f:
    settings = pickle.load(f)

#if 'modules' in settings.keys():
modules = settings.pop('modules')
for module in modules:
    exec(module)

# Built settings: 
built_settings = dict()
for key, setting in settings.items():
    try: # Some settings are functions.
        built_settings[key] = eval(setting)
    except: # Some are just strings that wont eval
        built_settings[key] = setting

calc = GPAW(**built_settings)

# Write the output file:
t.set_calculator(calc)
t.get_forces(apply_constraint=False)
t.get_potential_energy(apply_constraint=False)
Trajectory(args.output_file, mode='a').write(t)
"""

test_calc_script="""
print('I HAVE BEEN SUMMONED')
"""

class GPAW_IO(Calculator):
    """
    GPAW IO Calculator

    parexe: bool
        Run in parallel or not.
    script_name: str
        Name of the script file that will be written and execute. 
    settings_name: str
        Name of pickle file that settings will be written to. 
        If more than one calculator is used at the same time, each should have a distinct settings_name. 
    par_command: str
        Shell command used to invoke parallelization. Eg. mpiexec, mpirun or on a Slurm system srun.
    modules: list
        List of strings for additional imports like for example:
        'from gpaw.utilities import h2gpts'
    **kwargs: 
        The kwargs will be interpreted as settings for the GPAW calculation, BUT, they MUST be specified as strings 
        rather than the Python objects that GPAW expects. This is because actually importing these modules from gpaw 
        somehow breaks os.command (and also subprocess.Popen). So for example if you want to use plane-wave mode, you 
        supply mode='PW(300)', NOT mode=PW(300)! 
        DO NOT LOAD ANYTHING FROM THE GPAW LIBRARY IN THE AGOX SCRIPT PARALLELIZATION WILL NOT WORK AS EXPECTED OR AT ALL.     

    The trickery with modules and **kwargs is handled using exec and eval, which is not the safest thing in the world 
    as it will execute all supplied string as Python code. So be careful with that!
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, parexe=True, par_command='mpiexec', script_name='gpaw_calc.py', settings_name='gpaw_settings.pckl', modules=[], **kwargs):
        Calculator.__init__(self)
        
        self.parexe = parexe
        self.par_command = par_command

        self.script_name = script_name
        self.settings_name = settings_name
    
        # All kwargs are assumed to be settings!
        self.settings_for_gpaw = kwargs
        self.settings_for_gpaw['modules'] = modules

        self.input_file = 'gpaw_io_input.traj'
        self.output_file = 'gpaw_io_output.traj'

        self.prepare()

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms=atoms, properties=properties, system_changes=system_changes)

        write(self.input_file, atoms)
        write(self.output_file, [])

        commandline = ''
        if self.parexe: 
            commandline += self.par_command + ' '

        commandline += 'python ' + self.script_name + ' ' + self.input_file + ' ' + self.output_file + ' ' + self.settings_name

        # Execute the command-line:
        #print(commandline)
        s = os.system(commandline)

        # Read the results:
        try:
            tnew = read(self.output_file)
        except Exception as e:
            self.error_handling(e)

        # Assign energies:
        self.results['energy'] = tnew.get_potential_energy(apply_constraint=False)
        try: # Not sure when this can happen? 
            self.results['forces'] = tnew.get_forces(apply_constraint=False)
        except Exception as e:
           self.error_handling(e)

        # Delete the files:
        os.remove(self.output_file)
        os.remove(self.input_file)        


    def error_handling(self, exception):
         # Catch the error and print it
        print('Error: ', exception)
        # Delete files to avoid possibility of errors. 
        os.remove(self.output_file)
        os.remove(self.input_file)
        # Raise an exception that is caught by the gauge.
        # Unlike the error produced by GPAW this should only be printed once so much easier to search for.. 
        raise Exception('Energy calculation failed')

    def prepare(self):
        # Write the pickle settings file:
        with open(self.settings_name, 'wb') as f:
            pickle.dump(self.settings_for_gpaw, f)

        # Write the script:
        with open(self.script_name, 'w') as f:
            print(calc_script, file=f)        



