from agox.evaluators.local_optimization import LocalOptimizationEvaluator

class SinglePointEvaluator(LocalOptimizationEvaluator):

    name = 'SinglePointEvaluator'

    def __init__(self, calculator, **kwargs): 
        super().__init__(calculator, optimizer_run_kwargs=dict(steps=0), **kwargs)