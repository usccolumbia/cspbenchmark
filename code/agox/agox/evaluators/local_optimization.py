from .ABC_evaluator import EvaluatorBaseClass
import numpy as np
from agox.observer import Observer
from ase.calculators.singlepoint import SinglePointCalculator
from agox.writer import agox_writer

from ase.optimize.bfgs import BFGS
from ase.constraints import FixAtoms

from pymatgen.io.ase import AseAtomsAdaptor

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  ### for disabling tensorflow warnings

# from m3gnet.models import M3GNet

class LocalOptimizationEvaluator(EvaluatorBaseClass):

    name = 'LocalOptimizationEvaluator'

    def __init__(self, calculator, optimizer=BFGS, optimizer_run_kwargs={'fmax':0.25, 'steps':200},
                 optimizer_kwargs={'logfile':None}, fix_template=True, constraints=[],
                 store_trajectory=True, **kwargs): 
        super().__init__(**kwargs)
        self.calculator = calculator
        self.store_trajectory = store_trajectory
        
        # Optimizer stuff:
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_run_kwargs = optimizer_run_kwargs

        # Constraints:
        self.constraints = constraints
        self.fix_template = fix_template

        # Energy function
        self.aseToPymatgen = AseAtomsAdaptor()
        # self.m3gnet_e_form = M3GNet.load()

        if self.name == 'LocalOptimizationEvaluator':
            if optimizer_run_kwargs['steps'] == 0:
                print(f'{self.name}: Running with steps = 0 is equivalent to SinglePointEvaluator, consider using that instead to simplify your runscript.')
                self.store_trajectory = False
    
    def evaluate_candidate(self, candidate):
        candidate.calc = self.calculator
        try:
            if self.optimizer_run_kwargs['steps'] > 0:
                print('more than 1 steps ............')
                self.apply_constraints(candidate)
                optimizer = self.optimizer(candidate, **self.optimizer_kwargs)
                # print(candidate)
                # optimizer = self.optimizer(candidate)

                # The observer here stores all the steps if 'store_trajectory' is True.
                optimizer.attach(self._observer, interval=1, candidate=candidate, steps=optimizer.get_number_of_steps)
        
                optimizer.run(**self.optimizer_run_kwargs)
                # optimizer.run()                
                candidate.add_meta_information('relax_index', optimizer.get_number_of_steps())

                # If 'store_trajectory' is False we manually store the last step.
                if not self.store_trajectory:
                    self.evaluated_candidates.append(candidate)
            else:
                print('no steps ...............')
                # E = candidate.get_potential_energy(apply_constraint=False)
                candidate_pymatgen = self.aseToPymatgen.get_structure(candidate)
                E = candidate.calc.predict_structure(candidate_pymatgen)
                E = E[0][0].numpy().astype(float) / len(candidate_pymatgen)
                #F = candidate.get_forces(apply_constraint=False)
                self.evaluated_candidates.append(candidate)
            
        except Exception as e:
            self.writer('Energy calculation failed with exception: {}'.format(e))
            return False

        # E = candidate.get_potential_energy(apply_constraint=False)
        candidate_pymatgen = self.aseToPymatgen.get_structure(candidate)
        E = candidate.calc.predict_structure(candidate_pymatgen)
        E = E[0][0].numpy().astype(float) / len(candidate_pymatgen)
        print(f'final energy: {E}')
        #F = candidate.get_forces(apply_constraint=False)
        self.writer(f'Final energy of candidate = {E:5.3f}')
        #calc = SinglePointCalculator(candidate, energy=E, forces=F)
        calc = SinglePointCalculator(candidate, energy=E)
        # print(f'calc: {calc}')
        candidate.calc = calc

        return True


    def _observer(self, candidate, steps):
        # E = candidate.get_potential_energy(apply_constraint=False)
        # aseToPymatgen = AseAtomsAdaptor()
        candidate_pymatgen = self.aseToPymatgen.get_structure(candidate)
        # print(f'candidate_pymatgen: {candidate_pymatgen}')
        # m3gnet_e_form = M3GNet.load()
        E = candidate.calc.predict_structure(candidate_pymatgen)
        E = E[0][0].numpy().astype(float) / len(candidate_pymatgen)
        print(f'energy: {E}')
        #F = candidate.get_forces(apply_constraint=False)
        
        traj_candidate = candidate.copy()
        #calc = SinglePointCalculator(traj_candidate, energy=E, forces=F)
        calc = SinglePointCalculator(traj_candidate, energy=E)
        traj_candidate.set_calculator(calc)
        traj_candidate.add_meta_information('relax_index', steps())
        traj_candidate.add_meta_information('final', False)
        self.writer(f'Step {steps()}: {E:.3f}')

        if self.store_trajectory:
            self.evaluated_candidates.append(traj_candidate)

        #self.add_to_cache(self.set_key, [traj_candidate], mode='a')
    

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
