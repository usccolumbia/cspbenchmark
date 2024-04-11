import numpy as np
from agox.generators import ReplaceGenerator
from scipy.spatial.distance import cdist

class CenterOfGeometryGenerator(ReplaceGenerator):

    name = 'CenterOfGeometryGenerator'

    def __init__(self, selection_percentages={'low':0.25, 'high':0.5}, extra_radius_amplitude=1, extra_radius_params={'low':-0.5, 'high':3}, **kwargs):
        super().__init__(**kwargs)
        self.selection_percentages = selection_percentages
        self.extra_radius_amplitude = extra_radius_amplitude
        self.extra_radius_params = extra_radius_params

    def get_indices_to_move(self, candidate):
        # Calculate the distance to the COG: 
        cog = candidate.get_center_of_geometry()        
        cluster_indices = candidate.get_optimize_indices()
        cog_distances = cdist(candidate.positions[cluster_indices], cog).flatten()

        # Determine how many atoms to move:
        cluster_size = len(cluster_indices)
        atoms_picked = np.floor(np.random.uniform(**self.selection_percentages) * cluster_size).astype(int)

        # Pick the 'atoms_picked' number of atoms with the furthest distance to the COG. 
        picked_cluster_indices = list(cluster_indices[np.flip(np.argsort(cog_distances))[0:atoms_picked]])

        return picked_cluster_indices

    def get_new_position_center(self, candidate, index, indices_to_move):
        return candidate.get_center_of_geometry()
            
    def get_radius(self, candidate):
        cluster_indices = candidate.get_optimize_indices()
        cog = candidate.get_center_of_geometry()        
        cog_distances = cdist(candidate.positions[cluster_indices], cog)
        return np.mean(cog_distances) + self.extra_radius_amplitude * np.random.uniform(**self.extra_radius_params)

    

