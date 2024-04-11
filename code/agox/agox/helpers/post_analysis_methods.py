import os
import glob
import numpy as np
from ase.io import read, write
from agox.databases import Database
from agox.models.descriptors.spectral_graph_descriptor import SpectralGraphDescriptor

def load_structures_from_directory(directory_path, reload_files=False):


    traj_path = directory_path + 'all_structures.traj'

    if not os.path.exists(traj_path) or reload_files:
        database_paths = glob.glob(directory_path + '*.db')

        all_structures = []
        for database_path in database_paths:
            database = Database(filename=database_path)
            all_structures += database.restore_to_trajectory()

        write(traj_path, all_structures)
    else:
        all_structures = read(traj_path, ':')
    
    return all_structures

def energy_filtering(all_structures, threshold):
    E = [atoms.get_potential_energy() for atoms in all_structures]
    best_E = np.min(E)
    valid = E < (best_E + threshold)
    filtered_structures = [struc for struc, state in zip(all_structures, valid) if state]
    energies = [atoms.get_potential_energy() for atoms in all_structures]
    return filtered_structures, energies

class Cluster: 

    def __init__(self):
        self.structures = []
        self.energies = []
        self.features = []

        self.representative_feature = None
    
    def add_member(self, atoms, energy, feature):
        self.structures.append(atoms)
        self.energies.append(energy)
        self.features.append(feature)

    def get_best_member(self):
        idx = np.argmin(self.energies)

        self.structures[idx].info['count'] = len(self.structures)
        return self.structures[idx]

    def merge_cluster(self, cluster):
        self.structures += cluster.structures
        self.energies += cluster.energies
        self.features += cluster.features

    def get_best_energy(self):
        return np.min(self.energies)

    def set_representative_feature(self, feature):
        # This is weird, but it is not neccesarily the same type of feature 
        # as is present in self.features.
        self.representative_feature = feature

    def get_representative_feature(self):
        return self.representative_feature
    
    def reset_representative_feature(self):
        self.representative_feature = None

    def __len__(self):
        return len(self.structures)

class ClusterHandler:

    def __init__(self):
        pass

    def sort_clusters_according_to_energy(self, clusters):
        sort_indices = np.argsort([cluster.get_best_energy() for cluster in clusters])
        clusters = [clusters[i] for i in sort_indices]
        return clusters

class Catagorizer(ClusterHandler):

    def __init__(self, descriptor, distance_threshold):
        self.descriptor = descriptor
        self.distance_threshold = distance_threshold

    def catagorize_structures(self, data):
        clusters = []
        F = self.get_all_features(data)
        energies = [atoms.get_potential_energy() for atoms in data]
        for i, (atoms, f, e) in enumerate(zip(data, F, energies)):
            added_to_cluster = False
            for cluster in clusters:
                cluster_feature = cluster.features[0]
                if np.linalg.norm(f-cluster_feature) < self.distance_threshold:
                    cluster.add_member(atoms, e, f)
                    added_to_cluster = True
                    break
            
            if not added_to_cluster:
                new_cluster = Cluster()
                new_cluster.add_member(atoms, e, f)
                clusters.append(new_cluster)

        clusters = self.sort_clusters_according_to_energy(clusters)
        return clusters

    def get_all_features(self, data):
        F = []
        for atoms in data: 
            F.append(self.descriptor.get_feature(atoms))

        F = np.array(F)
        return F

class ClusterMerger(ClusterHandler):

    def __init__(self, descriptor, distance_threshold):
        self.descriptor = descriptor
        self.distance_threshold = distance_threshold

    def merge_clusters(self, clusters):
        merged_cluster_indices = {}
        index_to_key = {}
        clusters_that_will_be_merged = []
        for a, cluster_A in enumerate(clusters):

            # Get the feature for A: 
            feature_A = cluster_A.get_representative_feature()
            if feature_A is None:
                representative_A = cluster_A.get_best_member()
                feature_A = self.descriptor.get_feature(representative_A)
                cluster_A.set_representative_feature(feature_A)

            # Compare A to all other clusters. 
            for b, cluster_B in enumerate(clusters):
                
                # Don't compare to the same. 
                if b == a: 
                    continue

                # See if B is part of a merged cluster.
                key_for_a = index_to_key.get(a, None)
                key_for_b = index_to_key.get(b, None)
                
                # If both A and B are both part of a cluster already we should just stop. 
                if key_for_b is not None and key_for_a is not None:
                    continue
                
                # Get feature for B:
                feature_B = cluster_B.get_representative_feature()
                if feature_B is None:
                    representative_B = cluster_B.get_best_member()
                    feature_B = self.descriptor.get_feature(representative_B)
                    cluster_B.set_representative_feature(feature_B)

                # See if they are close enough:
                if np.linalg.norm(feature_A-feature_B) < self.distance_threshold:
                    if key_for_a is not None:
                        merged_cluster_indices[key_for_a].append(b)
                        index_to_key[b] = key_for_a
                        clusters_that_will_be_merged.append(b)
                    elif key_for_b is not None:
                        merged_cluster_indices[key_for_b].append(a)
                        index_to_key[a] = key_for_b
                        clusters_that_will_be_merged.append(a)
                    else:
                        new_key = len(merged_cluster_indices)
                        merged_cluster_indices[new_key] = [a, b]
                        index_to_key[a] = new_key
                        index_to_key[b] = new_key
                        clusters_that_will_be_merged.append(a)
                        clusters_that_will_be_merged.append(b)

        print(clusters_that_will_be_merged)
        print(merged_cluster_indices)
        merge_indices = [index_list for index_list in merged_cluster_indices.values()]
        non_merged = [i for i in range(len(clusters)) if i not in clusters_that_will_be_merged]
        final_clusters = [clusters[i] for i in non_merged]

        for merge_list in merge_indices:

            c = [clusters[j] for j in merge_list]
            new_cluster = Cluster()
            for cluster in c:
                new_cluster.merge_cluster(cluster)

            final_clusters.append(new_cluster)

        final_clusters = self.sort_clusters_according_to_energy(final_clusters)
        return final_clusters
