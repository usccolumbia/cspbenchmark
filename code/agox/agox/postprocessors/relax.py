import numpy as np
from agox.postprocessors.ABC_postprocess import PostprocessBaseClass
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.optimize.bfgslinesearch import BFGSLineSearch


class RelaxPostprocess(PostprocessBaseClass):

    name = 'PostprocessRelax'
    
    def __init__(self, model=None, optimizer=None, optimizer_run_kwargs={'fmax':0.05, 'steps':200}, optimizer_kwargs={'logfile':None}, constraints=[], fix_template=True, start_relax=1, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = BFGSLineSearch if optimizer is None else optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_run_kwargs = optimizer_run_kwargs
        self.start_relax = start_relax
        self.model = model

        # Constraints:
        self.constraints = constraints
        self.fix_template = fix_template
    
    def postprocess(self, candidate):
        initial_candidate = candidate.copy()
        candidate.set_calculator(self.model)
        self.apply_constraints(candidate)
        optimizer = self.optimizer(candidate, **self.optimizer_kwargs)

        try: 
            optimizer.run(**self.optimizer_run_kwargs)
        except Exception as e:
            print('Relaxation failed with exception: {}'.format(e))
            return initial_candidate
        
        candidate.add_meta_information('relaxation_steps', optimizer.get_number_of_steps())

        results = {prop: val for prop, val in candidate.calc.results.items() if prop in all_properties}
        candidate.calc = SinglePointCalculator(candidate, **results)

        print(f'Relaxed for {optimizer.get_number_of_steps()} steps')

        self.remove_constraints(candidate)
        return candidate

    def do_check(self, **kwargs):
        return (self.get_iteration_counter() > self.start_relax) * self.model.ready_state

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
