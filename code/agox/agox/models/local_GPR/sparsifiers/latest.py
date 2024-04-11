import numpy as np
from agox.models.sparsifiers.ABC_sparsifier import SparsifierBaseClass

class LatestSparsifier(SparsifierBaseClass):
    name = 'LatestSparsifier'

    def __init__(self, n_latest=100, **kwargs):
        super().__init__(**kwargs)
        self.n_latest = n_latest

    def _sparsify(self, list_of_candidates):
        if len(list_of_candidates) < self.n_latest:
            return list_of_candidates
        else:
            return list_of_candidates[-self.n_latest:]
            
