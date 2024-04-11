from abc import ABC, abstractmethod

class SparsifierBaseClass(ABC):
    def __init__(self, n_max=250, **kwargs):
        super().__init__(**kwargs)
        self.n_max = n_max

    def __call__(self, list_of_candidates):
        return True, self._sparsify(list_of_candidates)
        
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def _sparsify(self, list_of_candidates):
        pass

