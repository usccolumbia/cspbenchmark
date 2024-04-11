from ase.io import write

from agox.writer import agox_writer
from .ABC_sampler import SamplerBaseClass
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class KMeansSampler(SamplerBaseClass):
    name = 'SamplerKMeans'
    parameters = {}

    def __init__(self, descriptor=None, feature_calculator=None, model_calculator=None, sample_size=10, max_energy=5, 
                        use_saved_features=False, **kwargs):
        super().__init__(**kwargs)

        if descriptor is not None and feature_calculator is None:
            self.descriptor = descriptor
        if feature_calculator is not None and descriptor is None:
            print(DeprecationWarning("'feature_calculator'-argument will be removed in a future release, please use the descriptor argument instead."))
            self.descriptor = feature_calculator
        if feature_calculator is not None and descriptor is not None:
            print(DeprecationWarning("Both feature_calculator and descriptor arguments have been specified, please use only 'descriptor'"))

        self.sample_size = sample_size
        self.max_energy = max_energy
        self.sample = []
        self.sample_features = []
        self.model_calculator = model_calculator
        self.use_saved_features = use_saved_features
        self.debug = False

    def setup(self, all_finished_structures):

        if len(all_finished_structures) < 1:
            self.sample = []
            return
        
        e_all = np.array([s.get_potential_energy() for s in all_finished_structures])
        e_min = min(e_all)

        for i in range(5):
            filt = e_all <= e_min + self.max_energy * 2**i
            if np.sum(filt) >= 2*self.sample_size:
                break
        else:
            filt = np.ones(len(e_all), dtype=bool)
            index_sort = np.argsort(e_all)
            filt[index_sort[2*self.sample_size:]] = False

        structures = [all_finished_structures[i] for i in range(len(all_finished_structures)) if filt[i]]
        e = e_all[filt]

        f = self.get_features(structures)
        #f = np.array(self.get_global_features(structures))

        n_clusters = 1 + min(self.sample_size-1, int(np.floor(len(e)/5)))

        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10).fit(f)
        labels = kmeans.labels_

        indices = np.arange(len(e))
        sample = []
        sample_features = []
        for n in range(n_clusters):
            filt_cluster = labels == n
            cluster_indices = indices[filt_cluster]
            if len(cluster_indices) == 0:
                continue
            min_e_index = np.argmin(e[filt_cluster])
            index_best_in_cluster = cluster_indices[min_e_index]
            sample.append(structures[index_best_in_cluster])
            sample_features.append(f[index_best_in_cluster])

        sample_energies = [t.get_potential_energy() for t in sample]
        sorting_indices = np.argsort(sample_energies)

        self.sample = [sample[i] for i in sorting_indices]
        self.sample_features = [sample_features[i] for i in sorting_indices]
        
        sample_energies = [t.get_potential_energy() for t in self.sample]
        #self.writer('SAMPLE_DFT_ENERGY  ', '[',','.join(['{:8.3f}'.format(e) for e in sample_energies]),']')
        for i, sample_energy in enumerate(sample_energies):
            self.writer(f'{i}: Sample DFT Energy {sample_energy:8.3f}')
        
        if self.model_calculator is not None and self.model_calculator.ready_state:
            for s in self.sample:
                t = s.copy()
                t.set_calculator(self.model_calculator)
                E = t.get_potential_energy()
                sigma = self.model_calculator.get_property('uncertainty')
                s.add_meta_information('model_energy',E)
                s.add_meta_information('uncertainty',sigma)
            self.writer('SAMPLE_MODEL_ENERGY', \
                  '[',','.join(['{:8.3f}'.format(t.get_meta_information('model_energy')) for t in self.sample]),']')
            self.writer('SAMPLE_MODEL_SIGMA', \
                  '[',','.join(['{:8.3f}'.format(t.get_meta_information('uncertainty')) for t in self.sample]),']')

        if self.debug:
            write(f'all_strucs_iteration_{self.get_iteration_counter()}.traj', all_finished_structures)
            write(f'filtered_strucs_iteration_{self.get_iteration_counter()}.traj', structures)
            write(f'sample_iteration_{self.get_iteration_counter()}.traj', self.sample)
         
    def assign_to_closest_sample(self,candidate_object):

        if len(self.sample) == 0:
            return None
        
        # find out what cluster we belong to
        f_this = np.array(self.descriptor.get_global_features(candidate_object))
        distances = cdist(f_this, self.sample_features, metric='euclidean').reshape(-1)


        self.writer('cdist [',','.join(['{:8.3f}'.format(e) for e in distances]),']')

        d_min_index = int(np.argmin(distances))

        cluster_center = self.sample[d_min_index]
        cluster_center.add_meta_information('index_in_sampler_list_used_for_printing_purposes_only',d_min_index)
        return cluster_center

    def adjust_sample_cluster_distance(self, chosen_candidate, closest_sample):
        # See if this structure had a lower energy than the lowest energy in the cluster it belongs to
        chosen_candidate_energy = chosen_candidate.get_potential_energy()
        closest_sample_energy = closest_sample.get_potential_energy()
        d_min_index = closest_sample.get_meta_information('index_in_sampler_list_used_for_printing_purposes_only')
        self.writer('CLUST_RES {:06d}'.format(self.get_iteration_counter()),
              '[',','.join(['{:8.3f}'.format(t.get_potential_energy()) for t in self.sample]),
              '] {:8.3f} {:8.3f}'.format(closest_sample_energy,chosen_candidate_energy),
              'NEW {:d} {}'.format(d_min_index,chosen_candidate.get_meta_information('description')) \
              if closest_sample_energy > chosen_candidate_energy else '')

    def get_features(self, structures):
        if self.use_saved_features:
            features = []
            for candidate in structures:
                F = candidate.get_meta_information('kmeans_feature')
                if F is None:
                    F = self.descriptor.get_global_features(candidate)[0]
                    candidate.add_meta_information('kmeans_feature', F)
                features.append(F)
            features = np.array(features)
        else:
            features = np.array(self.descriptor.get_global_features(structures))
        return features
