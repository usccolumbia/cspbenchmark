from agox.postprocessors.ABC_postprocess import PostprocessBaseClass
import subprocess
from ase.io import read, write
from time import sleep
import pickle
import os
import numpy as np

from timeit import default_timer as dt

from ase.optimize import BFGS
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.constraints import FixAtoms

from ase.optimize import BFGS, BFGSLineSearch

optimizers = {'BFGS':BFGS, 'BFGSLineSearch':BFGSLineSearch}

SETTINGS_FILE_PATH = 'io_relax_settings.pckl'
SECONDARY_FILE_PATH = 'secondary_file.py'
MODEL_FILE_PATH = 'model.pckl'
OPTIMIZER_RUN_KWARGS_PATH = 'optimizer_run_kwargs.pckl'
MODEL_PARAMETERS_PATH = 'model_parameters.pckl'
GET_TO_WORK_FILE_PATH='READY'

SECONDARY_FILE=f"""import pickle
from agox.postprocessors.mpi_relax import SecondaryPostProcessMPIRelax
from agox.models.gaussian_process.featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
from agox.models.gaussian_process.delta_functions_multi.delta import delta
from agox.utils.constraints.box_constraint import BoxConstraint

import os
from mpi4py import MPI
world = MPI.COMM_WORLD
comm = world.Dup()

if comm.rank == 0:
    print(comm.size)

with open('{SETTINGS_FILE_PATH}', 'rb') as f:
    settings = pickle.load(f)

with open('{MODEL_FILE_PATH}', 'rb') as f:
    model = pickle.load(f)

with open('{OPTIMIZER_RUN_KWARGS_PATH}', 'rb') as f:
    opt_run_kwargs = pickle.load(f)

relaxer = SecondaryPostProcessMPIRelax(model, opt_run_kwargs, **settings)


relaxer.main_method(comm)
"""

########################################################################################################################
# Convenience functions:
########################################################################################################################

def safe_write(file_path, images):
    write('TEMP'+file_path, images)
    os.rename('TEMP'+file_path, file_path)

def save_model_parameters(model):
    model_parameters = model.get_model_parameters() # dict
    with open(MODEL_PARAMETERS_PATH, 'wb') as f:
        pickle.dump(model_parameters, f)

def load_model_parameters(model):
    with open(MODEL_PARAMETERS_PATH, 'rb') as f:
        model_parameters = pickle.load(f)
    model.set_model_parameters(model_parameters)

########################################################################################################################
# Primary 
########################################################################################################################

class MPIRelaxPostprocess(PostprocessBaseClass):

    name = 'MPI_relax'

    def __init__(self, model, database, optimizer='BFGS', start_relax=5, sleep_timing=1, output_file='output_candidates.traj', 
                input_file='input_candidates.traj', training_file='training_data.traj', start_sleep_time=0.01,
                optimizer_run_kwargs={'fmax':0.5, 'steps':200}, optimizer_kwargs={'logfile':None}, fix_template=True, constraints=[], 
                model_training_mode='primary', mpi_command='mpiexec', **kwargs):
        super().__init__(**kwargs)

        DeprecationWarning('MPIRelaxPostProcess is deprecated and will be removed in a future release. Use one of the Ray parallel relaxers.')

        assert type(optimizer) == str, 'Give optimizer as a string of the name, not an instance of the class'
        assert optimizer in optimizers.keys(), 'Invalid optimizer name given'

        self.model = model

        self.input_file = input_file
        self.output_file = output_file
        self.training_file = training_file
        self.sleep_timing = sleep_timing
        self.database = database

        # Controls how the model is trained:
        if model_training_mode in ['both', 'secondary', 'primary']:
            self.model_training_mode = model_training_mode
        else:
            ValueError('model_training_mode: {} is not any of the valid choices: both, secondary or primary'.format(model_training_mode))
        self.previous_database_length = 0

        # Kwargs for the ASE optimizer.
        self.optimizer_run_kwargs = optimizer_run_kwargs
        self.start_relax = start_relax

        # For constraints:
        self.fix_template = fix_template
        self.constraints = constraints

        self.settings = {'sleep_timing':sleep_timing, 'output_file':output_file, 'input_file':input_file, 
                         'model_training_mode':model_training_mode, 'optimizer_key':optimizer, 
                         'optimizer_kwargs':optimizer_kwargs}

        self.mpi_command = mpi_command
        self.relaxation_process = None
        self.start_sleep_time = start_sleep_time

        self.write_secondary_file()
        self.write_settings_file()
        self.save_model_pickle()

    @PostprocessBaseClass.immunity_decorator_list
    def process_list(self, candidates):
        """
        The basic scheme here is to: 

        1. Write the candidates to a trajectory file. 
        2. Wait for them to be read and processed by the secondary process
        3. Read results and return them.
        """

        if self.relaxation_process is None:
            self.start_secondary_process()

        t0 = dt()
        if self.get_iteration_counter() < self.start_relax:
            return candidates

        # Handling of training stuff:
        if self.model_training_mode == 'both' or self.model_training_mode == 'secondary':
            self.save_training_data()
        elif self.model_training_mode == 'primary':
            # Should maybe do something here so that this is ignored if the model is unchanged since previous call.            
            save_model_parameters(self.model)

        # Write the candidates that should be processed:
        [self.apply_constraints(candidate) for candidate in candidates]
        safe_write(self.input_file, candidates)

        # All input should have been written successfully at this point, the secondary process is looking for this file, 
        # and will start as soon at it see it. 
        open(GET_TO_WORK_FILE_PATH, 'x').close()

        relaxation_finished = False
        while not relaxation_finished:
            sleep(self.sleep_timing)
            successful_read = False
            if os.path.exists(self.output_file):
                try:
                    output = read(self.output_file, index=':')
                    successful_read = True
                except Exception as e:
                    successful_read = False
            if successful_read:
                relaxation_finished = True

        os.remove(self.output_file)

        relaxed_candidates = []
        for atoms, input_candidate in zip(output, candidates):
            relaxed_candidates.append(self.convert_to_candidate_object(atoms, input_candidate.template))
            relaxed_candidates[-1].meta_information = input_candidate.meta_information
            for key, val in zip(atoms.info.keys(), atoms.info.values()):
                relaxed_candidates[-1].add_meta_information(key, val)         

        if self.model_training_mode == 'secondary':
            load_model_parameters(self.model)

        t1 = dt()
        self.writer(f'Candidate relaxation time: {t1-t0}')
        average_steps = np.mean([relaxed_candidate.get_meta_information('optimizer_steps') for relaxed_candidate in relaxed_candidates])
        self.writer(f'Relaxed for an average of {average_steps} steps')

        return relaxed_candidates
            
    def postprocess(self, candidate):
        return self.process_list([candidate])[0]
    
    def apply_constraints(self, candidate):
        constraints = [] + self.constraints
        if self.fix_template:
            constraints.append(self.get_template_constraint(candidate))

        for constraint in constraints:
            if hasattr(constraint, 'reset'):
                constraint.reset()

        candidate.set_constraint(constraints)

    def get_template_constraint(self, candidate):
        return FixAtoms(indices=np.arange(len(candidate.template)))

    ####################################################################################################################
    # For starting the secondary process:
    ####################################################################################################################

    def write_secondary_file(self):
        with open(SECONDARY_FILE_PATH, 'w') as f:
            print(SECONDARY_FILE, file=f)

        with open(OPTIMIZER_RUN_KWARGS_PATH, 'wb') as f:
            pickle.dump(self.optimizer_run_kwargs, f)
            
    def write_settings_file(self):
        with open(SETTINGS_FILE_PATH, 'wb') as f:
            pickle.dump(self.settings, f)

    def start_secondary_process(self):
        self.relaxation_process = subprocess.Popen([self.mpi_command, 'python', SECONDARY_FILE_PATH])
        sleep(self.start_sleep_time) # Required to make sure the model starts up in time (for network models, can be set to a small value for GPR)

    def terminate_secondary_process(self):
        try:
            outs, errs = self.relaxation_process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            self.relaxation_process.kill()
            outs, errs = self.relaxation_process.communicate()
        self.relaxation_process = None
        print('Good chance the secondary relaxation process terminated succesfully')        


    def restart_secondary_process(self, model):
        self.terminate_secondary_process()
        if os.path.exists(GET_TO_WORK_FILE_PATH):
            os.remove(GET_TO_WORK_FILE_PATH)
        if os.path.exists(MODEL_FILE_PATH):
            os.remove(MODEL_FILE_PATH)
        self.model = model
        self.save_model_pickle()
        self.start_secondary_process()

        
    def save_model_pickle(self):
        verbose_before = self.model.verbose
        with open(MODEL_FILE_PATH, 'wb') as f:
            self.model.set_verbosity(False)
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.model.set_verbosity(verbose_before)

    def save_training_data(self):        
        if len(self.database) > self.previous_database_length:
            safe_write(self.training_file, self.database.get_all_candidates())
            self.previous_database_length = len(self.database)
    
    ####################################################################################################################
    # Misc.
    ####################################################################################################################

    def attach(self, main):
        super().attach(main)
        main.attach_finalization('kill_relax_mpi', self.terminate_secondary_process)

    def convert_to_candidate_object(self, atoms_type_object, template):
        candidate =  self.candidate_instanstiator(template=template, positions=atoms_type_object.positions, numbers=atoms_type_object.numbers, 
                                          cell=atoms_type_object.cell)

        candidate.set_calculator(SPC(candidate, energy=atoms_type_object.get_potential_energy()))

        return candidate
    
########################################################################################################################
# Secondary
########################################################################################################################

class SecondaryPostProcessMPIRelax:

    def __init__(self, model, optimizer_run_kwargs, optimizer_kwargs={}, optimizer_key='BFGS', model_training_mode='both', sleep_timing=1, output_file='output_candidates.traj', input_file='input_candidates.traj', training_file='training_data.traj'):
        self.input_file = input_file
        self.output_file = output_file 
        self.sleep_timing = sleep_timing
        self.training_file = training_file
        self.optimizer_run_kwargs = optimizer_run_kwargs
        self.model_training_mode = model_training_mode
        self.model = model
        self.optimizer_instanstiator = optimizers[optimizer_key]
        self.optimizer_kwargs = optimizer_kwargs

    def single_relaxation(self, candidate, comm):            
        candidate.set_calculator(self.model)
        dyn = self.optimizer_instanstiator(candidate, **self.optimizer_kwargs)
        try: 
            dyn.run(**self.optimizer_run_kwargs)
            candidate.info['optimizer_steps'] = dyn.get_number_of_steps()
            candidate.info['succesful_model_relaxation'] = True
        except Exception as exception:
            print('Relaxation ran into an exception: {}'.format(exception))
            candidate.info['succesful_model_relaxation'] = False
            candidate.info['optimizer_steps'] = np.nan

        candidate.set_calculator(SPC(candidate, energy=candidate.get_potential_energy()))
        candidate.set_constraint([]) # Remove constraints because otherwise stuff breaks.
        return candidate

    def parallel_relaxation(self, comm, candidates):
        Njobs = len(candidates)
        task_split = split(Njobs, comm.size)
        def dummy_func():
            calculations = [self.single_relaxation(candidates[i], comm) for i in task_split[comm.rank]]
            return calculations
        relaxed_candidates = parallel_function_eval(comm, dummy_func)
        comm.Barrier()
        return relaxed_candidates

    def main_method(self, comm):

        # if comm.rank == 0:
        #     print('Model on secondary:', self.model)

        C = 0
        state = True
        while state:
            sleep(self.sleep_timing) # Start by sleep for awhile: 
            
            if comm.rank == 0:
                work_to_do = os.path.exists(GET_TO_WORK_FILE_PATH)
            else: 
                work_to_do = False
            work_to_do = comm.bcast(work_to_do, root=0)

            if work_to_do:
                # Train
                self.model_training(comm)                                    

                # Relax
                self.model_relaxation(comm)

    def model_relaxation(self, comm):
        #if os.path.exists(self.input_file):
        candidates = read(self.input_file, index=':', parallel=False)
        comm.Barrier()                
        relaxed_candidates = self.parallel_relaxation(comm, candidates)
        if comm.rank == 0:
            safe_write(self.output_file, relaxed_candidates)
            os.remove(self.input_file)
            os.remove(GET_TO_WORK_FILE_PATH)

        comm.Barrier()

    def model_training(self, comm):
        # Look for training data & train model.
        if self.model_training_mode == 'secondary' or self.model_training_mode == 'both':
            #if os.path.exists(self.training_file):
            if comm.rank == 0:
                t0 = dt()
            
            data = read(self.training_file, index=':', parallel=False) # This is somewhat inefficient I think, but had trouble with better ways. 
            comm.Barrier()
            if comm.rank == 0:
                os.remove(self.training_file)

            #data = comm.bcast(data, root=0)
            energies = [atoms.get_potential_energy() for atoms in data]                
            self.model.train_model(data, energies)

            if comm.rank == 0:
                t1 = dt()
                print('Training time on master rank: {}'.format(t1-t0))

            if self.model_training_mode == 'secondary':
                if comm.rank == 0:
                    save_model_parameters(self.model)

        elif self.model_training_mode == 'primary':
            if comm.rank == 0:
                t0 = dt()
            load_model_parameters(self.model)
            if comm.rank == 0:
                t1 = dt()
                #print('Loaded model time: {}'.format(t1-t0))

def split(Njobs, comm_size):
    """Splits job indices into simmilar sized chuncks
    to be carried out in parallel.
    """
    Njobs_each = Njobs // comm_size * np.ones(comm_size, dtype=int)
    Njobs_each[:Njobs % comm_size] += 1
    Njobs_each = np.cumsum(Njobs_each)
    split = np.split(np.arange(Njobs), Njobs_each[:-1])
    return split

def parallel_function_eval(comm, func):
    """Distributes the results from parallel evaluation of func()
    among all processes.
    
    comm: mpi communicator

    func: function to evaluate. Should return a list of results.
    """
    results = func()
    results_all = comm.gather(results, root=0)
    results_all = comm.bcast(results_all, root=0)
    results_list = []
    for results in results_all:
        results_list += results
    return results_list
