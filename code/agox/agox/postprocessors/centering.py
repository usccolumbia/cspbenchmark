import numpy as np
from agox.postprocessors.ABC_postprocess import PostprocessBaseClass

class CenteringPostProcess(PostprocessBaseClass):

    name = 'CenteringPostProcess'

    @PostprocessBaseClass.immunity_decorator
    def postprocess(self, candidate):
        """
        Centers a candidate object to the middle of the cell. 
        """
        com = candidate.get_center_of_mass()
        cell_middle = np.sum(candidate.get_cell(), 0) / 2
        candidate.positions = candidate.positions - com + cell_middle
        return candidate



