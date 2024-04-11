import numpy as np
from agox.models.local_GPR.LSGPR import LSGPRModel
from scipy.linalg import svd

class LSGPRModelCUR(LSGPRModel):
    name = 'LSGPRModelCUR'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    

    def _train_sparse(self, atoms_list, **kwargs):
        if self.Xn.shape[0] < self.m_points:
            self.Xm = self.Xn
            return True

        U, _, _ = svd(self.Xn)
        score = np.sum(U[:,:self.m_points]**2, axis=1)/self.m_points
        sorter = np.argsort(score)[::-1]
        self.Xm = self.Xn[sorter, :][:self.m_points, :]
            
        return True




    
