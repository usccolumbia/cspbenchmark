import numpy as np
from ase.data import covalent_radii
from agox.postprocessors.ABC_postprocess import PostprocessBaseClass

class MinimumDistPostProcess(PostprocessBaseClass):

    name = 'MinimumDistance'
    def __init__(self, c1=0.75, c2=1.25, fragmented_check=True, backbone_check=False, **kwargs):
        super().__init__(**kwargs)
        self.c1 = c1
        self.c2 = c2
        self.backbone_check = backbone_check
        self.fragmented_check = fragmented_check
    
    def postprocess(self, candidate):
        if candidate is None:
            return None

        distances = self.get_distances(candidate)
        if np.any(distances[np.tril_indices(distances.shape[0], k=-1)] < self.c1):
            print('Candidate minimum distance violated')
            return None

        # check max distance through laplacian spectrum
        if self.fragmented_check:
            L = self.get_laplacian_matrix(distances)
            w,_ = np.linalg.eig(L)
            if np.sum(np.abs(w) < 1e-12) > 1:
                print('Candidate is fragmented')
                return None


        if self.backbone_check:
            candidate_no_H = candidate.copy()
            indices = np.argwhere(candidate_no_H.numbers == 1)
            del candidate_no_H[indices]

            distances = self.get_distances(candidate_no_H)
            L = self.get_laplacian_matrix(distances)
            w,_ = np.linalg.eig(L)

            if np.sum(np.abs(w) < 1e-12) > 1:
                print('Candidate backbone is fragmented')
                return None
                                       
        self.writer('Candidate dist OK')
        return candidate


    def get_laplacian_matrix(self, relative_dist):
        A = np.logical_and(relative_dist > self.c1, relative_dist < self.c2).astype(int)
        D = np.diag(np.sum(A, axis=1))
        return D-A

    def get_distances(self, candidate):
        dist = candidate.get_all_distances(mic=True)
        cov_dist = np.array([covalent_radii[n] for n in candidate.numbers])
        relative_dist = dist / np.add.outer(cov_dist, cov_dist)
        return relative_dist

    
if __name__ == '__main__':
    from ase.build import molecule
    from ase import Atoms
    from ase.visualize import view
    h20 = molecule('H2O')
    h20 += Atoms('H', positions=[[0.,.7,.7]])
    # h20 += Atoms('C', positions=[[0.,1.4,1.4]])
    view(h20)
    print(h20.positions)
    min_dist = MinimumDistPostProcess(backbone_check=True)
    min_dist.postprocess(h20)
        
