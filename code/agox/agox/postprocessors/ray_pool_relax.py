import numpy as np
from agox.postprocessors.ABC_postprocess import PostprocessBaseClass
from agox.utils.ray_utils import RayPoolUser
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import BFGS
from ase.constraints import FixAtoms

def relax(model, candidate, optimizer, optimizer_kwargs, optimizer_run_kwargs):
    candidate = candidate.copy()
    candidate.set_calculator(model)
    optimizer = optimizer(candidate,**optimizer_kwargs)
    try:
        optimizer.run(**optimizer_run_kwargs)
    except Exception as e:
        print('Relaxation failed with exception: {}'.format(e))
        candidate.add_meta_information('relaxation_steps', -1)
        return candidate
    candidate.add_meta_information('relaxation_steps', 
        optimizer.get_number_of_steps())
    return candidate

class ParallelRelaxPostprocess(PostprocessBaseClass, RayPoolUser):

    name = 'PoolRelaxer'
    
    def __init__(self, model=None, optimizer=None,
        optimizer_run_kwargs={'fmax':0.05, 'steps':200}, fix_template=True,
        optimizer_kwargs={'logfile':None}, constraints=[], start_relax=1, 
        **kwargs):
        pool_user_kwargs = {key:kwargs.pop(key, None) for key in RayPoolUser.kwargs if kwargs.get(key, None) is not None}
        RayPoolUser.__init__(self, **pool_user_kwargs)
        PostprocessBaseClass.__init__(self, **kwargs)

        self.optimizer = BFGS if optimizer is None else optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_run_kwargs = optimizer_run_kwargs
        self.start_relax = start_relax
        self.model = model
        self.constraints = constraints
        self.fix_template = fix_template
        self.model_key = self.pool_add_module(model)

    def process_list(self, candidates):
        """
        Relaxes the given candidates in parallel using Ray. 

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
        [self.apply_constraints(candidate) for candidate in candidates]

        # Make args, kwargs and modules lists:
        N_jobs = len(candidates)
        modules = [[self.model_key]] * N_jobs
        args = [[candidate] + [self.optimizer, self.optimizer_kwargs, self.optimizer_run_kwargs] for candidate in candidates]
        kwargs = [{} for _ in range(N_jobs)]
        relaxed_candidates = self.pool_map(relax, modules, args, kwargs)
        
        # Remove constraints & move relaxed positions to input candidates:
        # This is due to immutability of the candidates coming from pool_map.
        [self.remove_constraints(candidate) for candidate in candidates]
        for cand, relax_cand in zip(candidates, relaxed_candidates):
            cand.set_positions(relax_cand.positions)
            results = {prop: val for prop, val in relax_cand.calc.results.items() if prop in all_properties}
            cand.calc = SinglePointCalculator(cand, **results)

        average_steps = np.mean([candidate.get_meta_information('relaxation_steps') 
            for candidate in relaxed_candidates])
        self.writer(f'{len(relaxed_candidates)} candidates relaxed for an average of {average_steps} steps.')

        return candidates

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
