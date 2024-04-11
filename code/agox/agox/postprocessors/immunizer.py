import numpy as np
from agox.postprocessors.ABC_postprocess import PostprocessBaseClass

class ImmunizerPostprocess(PostprocessBaseClass):

    name = 'TheImmunizer'

    def __init__(self, probability=1, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability

    def postprocess(self, candidate):
        if np.random.rand() < self.probability:
            candidate.set_postprocess_immunity(True)
        return candidate
        
class DeimmunizerPostprocess(PostprocessBaseClass):

    name = 'TheDeimmunizer'

    def postprocess(self, candidate):        
        candidate.set_postprocess_immunity(False)
        return candidate
        