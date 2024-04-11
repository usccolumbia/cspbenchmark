import numpy as np

class Confinement:

    def __init__(self, confinement_cell=None, confinement_corner=None, indices=None, pbc=[False]*3):
        self.confinement_cell = np.array(confinement_cell)
        self.confinement_corner = np.array(confinement_corner)

        if indices is None:
            indices = np.array([])
        self.indices = np.array(indices).flatten()

        self.periodicity_setup(pbc)
        self.confined = self.confinement_cell is not None and self.confinement_corner is not None

    def periodicity_setup(self, pbc):
        if isinstance(pbc, bool):
            self.pbc = [pbc]*3
        elif len(pbc) == 3:
            self.pbc = list(pbc)
        else:
            print('pbc should be list or bool! Setting pbc=False.')
            self.pbc = [False]*3

        self.hard_boundaries = [not p for p in self.pbc]

        if np.any(self.pbc):
            periodic_cell_vectors = self.confinement_cell[:, self.pbc]
            non_periodic_cell_vectors = self.confinement_cell[:, self.hard_boundaries]
            if np.any(np.abs(np.matmul(periodic_cell_vectors.T, non_periodic_cell_vectors)) > 0):
                print('---- BOX CONSTRAINT ----')
                print('Periodicity does not work for non-square non-periodic directions!')
                print('------------------------')

        if np.all(self.confinement_cell[:, 2] == 0):
            self.dimensionality = 2
            self.effective_confinement_cell = self.confinement_cell[0:2, 0:2]
            if np.all(self.confinement_cell[:, 1] == 0):
                self.effective_confinement_cell = self.confinement_cell[0:1, 0:1]
                self.dimensionality = 1
        else:
            self.effective_confinement_cell = self.confinement_cell
            self.dimensionality = 3

    def get_projection_coefficients(self, positions):
        positions = positions.reshape(-1, 3)
        return np.linalg.solve(self.effective_confinement_cell.T, (positions-self.confinement_corner)[:, 0:self.dimensionality].T).T.reshape(-1, self.dimensionality)

    def check_confinement(self, positions):
        """
        Finds the fractional coordinates of the atomic positions in terms of the box defined by the constraint. 
        """
        if self.confined:        
            C = self.get_projection_coefficients(positions)
            inside = np.all((C > 0) * (C < 1), axis=1)
            return inside
        else:
            return np.ones(positions.shape[0]).astype(bool)

    def get_confinement_limits(self):
        """
        This returns the confinement-limit lists which always assumes a square box. 
        """
        conf = [self.confinement_corner, self.confinement_cell @ np.array([1, 1, 1]) + self.confinement_corner]
        return conf

    def _get_box_vector(self, cell, corner):
        return cell.T @ np.random.rand(3) + corner

    def set_confinement_cell(self, cell, confinement_corner):        
        self.confinement_cell = cell
        self.confinement_corner = confinement_corner
        self.confined = True
        self.confined = self.confinement_cell is not None and self.confinement_corner is not None

    def set_dimensionality(self, dimensionality):
        self.dimensionality = dimensionality

    
