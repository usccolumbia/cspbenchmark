import ray
import numpy as np

from ase.optimize import BFGS
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from agox.postprocessors.ABC_postprocess import PostprocessBaseClass
from agox.utils.ray_utils import RayBaseClass, ray_kwarg_keys

@ray.remote
def ray_relaxation_function(structure, calculator, 
        optimizer, optimizer_kwargs, optimizer_run_kwargs):
    """
    Remote function that performs the actual relaxation. 

    Parameters
    ----------
    structure : Candidate
        AGOX Candidate object to relax.
    calculator : AGOX Model calculator.
        Calculator to use.
    optimizer : ASE Optimizer
        ASE Optimizer class - not an instance of the class!
    optimizer_kwargs : dict
        Settings with which intialize the ASE optimizer.
    optimizer_run_kwargs : dict
        Settings with which to start the relaxation. 

    Returns
    -------
    AGOX candidate
        Relaxed candidate.
    """
    structure_copy=structure.copy()
    structure_copy.set_calculator(calculator)
    optimizer = optimizer(structure_copy,**optimizer_kwargs)
    try:
        optimizer.run(**optimizer_run_kwargs)
    except Exception as e:
        print('Relaxation failed with exception: {}'.format(e))
        structure.add_meta_information('relaxation_steps', -1)
        return structure
    structure_copy.add_meta_information('relaxation_steps', 
        optimizer.get_number_of_steps())
    return structure_copy

class ParallelRelaxPostprocess(PostprocessBaseClass, RayBaseClass):

    name = 'RayPostprocessRelax'
    
    def __init__(self, model=None, optimizer=None,
        optimizer_run_kwargs={'fmax':0.05, 'steps':200}, fix_template=True,
        optimizer_kwargs={'logfile':None}, constraints=[], start_relax=1, 
        **kwargs):
        ray_kwargs = {key:kwargs.pop(key, None) for key in ray_kwarg_keys}
        PostprocessBaseClass.__init__(self, **kwargs)
        RayBaseClass.__init__(self, **ray_kwargs)

        self.optimizer = BFGS if optimizer is None else optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_run_kwargs = optimizer_run_kwargs
        self.start_relax = start_relax
        self.model = model
        self.constraints = constraints
        self.fix_template = fix_template

        self.ray_startup()

        # Put things taht are constant:
        self.optimizer_kwargs_re = ray.put(self.optimizer_kwargs)
        self.optimizer_run_kwargs_re = ray.put(self.optimizer_run_kwargs)
        self.optimizer_re = ray.put(self.optimizer)

    def process_list(self, list_of_candidates):
        """
        Relaxes the given candidates in paralle using Ray. 

        Parameters
        ----------
        list_of_candidates : listn
            List of AGOX candidates to relax. 

        Returns
        -------
        list
            List of relaxed candidates.
        """
        # Apply constraints. 
        [self.apply_constraints(candidate) for candidate in list_of_candidates]

        # Put the model:
        model_re = ray.put(self.model)

        # Make futures:
        post_processed_candidates = [ray_relaxation_function.remote(candidate, 
                model_re, self.optimizer_re, self.optimizer_kwargs_re, 
                self.optimizer_run_kwargs_re) for candidate in list_of_candidates]

        # Get the candidates:
        post_processed_candidates = ray.get(post_processed_candidates)

        # Copy properties over to the input:
        for cand, processed_cand in zip(list_of_candidates, post_processed_candidates):
            cand.positions = processed_cand.positions.copy()
            results = {prop: val for prop, val in processed_cand.calc.results.items() if prop in all_properties}
            cand.calc = SinglePointCalculator(cand, **results)
            cand.add_meta_information('relaxation_steps', processed_cand.get_meta_information('relaxation_steps'))
        
        # Remove constraints.
        [self.remove_constraints(candidate) for candidate in list_of_candidates]

        average_steps = np.mean([candidate.get_meta_information('relaxation_steps') 
            for candidate in list_of_candidates])
        self.writer(f'{len(list_of_candidates)} candidates relaxed for an average of {average_steps} steps.')

        return list_of_candidates

    def postprocess(self, candidate):
        raise NotImplementedError('"postprocess"-method is not implemented, use postprocess_list.')

    def do_check(self, **kwargs):
        return self.check_iteration_counter(self.start_relax) * self.model.ready_state

    ####################################################################################################################
    # Constraints
    ####################################################################################################################

    def apply_constraints(self, candidate):
        constraints = [] + self.constraints
        if self.fix_template:
            constraints.append(self.get_template_constraint(candidate))

        for constraint in constraints:
            if hasattr(constraint, 'reset'):
                constraint.reset()

        candidate.set_constraint(constraints)

    def remove_constraints(self, candidate):
        candidate.set_constraint([])

    def get_template_constraint(self, candidate):
        return FixAtoms(indices=np.arange(len(candidate.template)))
