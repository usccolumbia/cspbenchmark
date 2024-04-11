import numpy as np
from agox.acquisitors.ABC_acquisitor import AcquisitorBaseClass, AcquisitonCalculatorBaseClass
from ase.calculators.calculator import all_changes

class LowerConfidenceBoundAcquisitor(AcquisitorBaseClass):

    name = 'LCBAcquisitor'

    def __init__(self, model_calculator, kappa=1, **kwargs):
        super().__init__(**kwargs)
        self.kappa = kappa
        self.model_calculator = model_calculator

    def calculate_acquisition_function(self, candidates):
        fitness = np.zeros(len(candidates))
        
        # Attach calculator and get model_energy
        for i, candidate in enumerate(candidates):
            candidate.set_calculator(self.model_calculator)
            E = candidate.get_potential_energy()
            sigma = candidate.get_uncertainty()
            fitness[i] = self.acquisition_function(E, sigma)

            # For printing:
            candidate.add_meta_information('model_energy', E)
            candidate.add_meta_information('uncertainty', sigma)

        return fitness

    def print_information(self, candidates, acquisition_values):
        if self.model_calculator.ready_state:
            for i, candidate in enumerate(candidates):
                fitness = acquisition_values[i]
                Emodel = candidate.get_meta_information('model_energy')
                sigma = candidate.get_meta_information('uncertainty')
                self.writer('Candidate: E={:8.3f}, s={:8.3f}, F={:8.3f}'.format(Emodel, sigma, fitness))

    def get_acquisition_calculator(self):
        return LowerConfidenceBoundCalculator(self.model_calculator, self.acquisition_function, self.acquisition_force)

    def acquisition_function(self, E, sigma):
        return E - self.kappa * sigma

    def acquisition_force(self, E, F, sigma, sigma_force):
        return F - self.kappa*sigma_force

    def do_check(self, **kwargs):
        return self.model_calculator.ready_state

class LowerConfidenceBoundCalculator(AcquisitonCalculatorBaseClass):

    implemented_properties = ['energy', 'forces']

    def __init__(self, model_calculator, acquisition_function, acquisition_force, **kwargs):
        super().__init__(model_calculator, **kwargs)
        self.acquisition_function = acquisition_function
        self.acquisition_force = acquisition_force

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if 'forces' in properties:
            E, sigma = self.model_calculator.predict_energy(atoms, return_uncertainty=True)
            F, sigma_force = self.model_calculator.predict_forces(atoms, return_uncertainty=True, acquisition_function=self.acquisition_function)
            self.results['forces'] = self.acquisition_force(E, F, sigma, sigma_force)
        else:
            E, sigma = self.model_calculator.predict_energy(atoms, return_uncertainty=True)
            
        self.results['energy'] = self.acquisition_function(E, sigma)


        

