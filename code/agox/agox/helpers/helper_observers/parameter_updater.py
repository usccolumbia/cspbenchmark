import numpy as np
from agox.observer import Observer
from abc import ABC, abstractmethod

class ParameterUpdaterBaseClass(ABC, Observer):

    def __init__(self, module, parameter, gets={}, sets={}, order=0):
        Observer.__init__(self, gets=gets, sets=sets, order=order)
        self.module = module
        self.parameter = parameter
    
    def update_parameter(self):
        print('Update')
        print(self.get_parameter())
        if self.decide_to_update():
            value = self.get_updated_parameter_value()
            self.set_parameter(value)

        print(self.get_parameter())

    @abstractmethod
    def get_updated_parameter_value(self):
        pass

    @abstractmethod
    def decide_to_update(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    def attach(self, main):
        main.attach_observer(self.name+'.update_parameter', self.update_parameter, order=self.order)

    def get_parameter(self):
        return self.module.__dict__[self.parameter]
    
    def set_parameter(self, value):
        self.module.__dict__[self.parameter] = value


class RattleAmplitudeUpdater(ParameterUpdaterBaseClass):

    name = 'RattleAmplitudeUpdater'

    def __init__(self, module, parameter, gets={}, sets={}, order=0, **kwargs):
        super().__init__(module, parameter, gets=gets, sets=sets, order=order, **kwargs)

    def get_updated_parameter_value(self):
        return np.random.rand() * 3

    def decide_to_update(self):
        return True

        
