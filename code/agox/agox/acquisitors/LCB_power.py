import numpy as np
from agox.acquisitors import LowerConfidenceBoundAcquisitor, LowerConfidenceBoundCalculator

class PowerLowerConfidenceBoundAcquisitor(LowerConfidenceBoundAcquisitor):

    name = 'PowerLowerConfindenceBoundAcquisitor'

    def __init__(self, power=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.power = power

    def acquisition_function(self, E, sigma):
        return E - self.kappa * sigma ** self.power

    def acquisition_force(self, E, F, sigma, sigma_force):
        return F - self.kappa * self.power * sigma**(self.power-1) * sigma_force

