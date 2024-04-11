from .ABC_sampler import SamplerBaseClass
import numpy as np
from ase.io import write

class KernelSimSampler(SamplerBaseClass):

    name = 'KernelSimSampler'

    def __init__(self, descriptor, model_calculator, log_scale=False,
        similarity_criterion=0.95, sample_size=10, use_saved_features=False,
        **kwargs):
        super().__init__(**kwargs)
        self.feature_calculator = descriptor
        if log_scale:
            # 0.995 becomes: 0.995, 0.990, 0.980, 0.960, ...
            self.similarity_criteria = [1 - (1 - similarity_criterion) * 2**i for i in range(sample_size)]
        else:
            self.similarity_criteria = [similarity_criterion] * sample_size
        self.writer('SAMPLE_SIMILARITY_CRITERIA', self.similarity_criteria)
        self.sample_size = sample_size
        self.model_calculator = model_calculator
        self.sample = []
        self.use_saved_features = use_saved_features

    def setup(self, all_finished_structures):
        if self.verbose:
            self.writer('len(all_finished_structures):',len(all_finished_structures))
        if len(all_finished_structures) == 0:
            return
        
        all_finished_structures.sort(key=lambda x: x.get_potential_energy())

        self.f_all = self.get_features(all_finished_structures)

        if self.model_calculator.ready_state:
            self.K0 = np.exp(np.array(self.model_calculator.model.kernel_.theta))[0]
            self.sample_indices = [0]
            for i, f_i in enumerate(self.f_all):
                this_is_a_new_structure = True
                for j,idx in enumerate(self.sample_indices):
                    similarity = self.model_calculator.model.kernel_.get_kernel(self.f_all[i],self.f_all[idx]) / self.K0
                    if similarity > self.similarity_criteria[j]:
                        this_is_a_new_structure = False
                        break
                if this_is_a_new_structure:
                    self.sample_indices.append(i)
                    if len(self.sample_indices) >= self.sample_size:
                        break
        else:
            self.sample_indices = list(range(min(len(all_finished_structures),self.sample_size)))

        self.sample = [all_finished_structures[i] for i in self.sample_indices]
        if self.verbose:
            write('sample_{:06d}.traj'.format(self.get_iteration_counter()),self.sample)
        sample_energies = [t.get_potential_energy() for t in self.sample]
        self.writer('SAMPLE_DFT_ENERGY  ', '[',','.join(['{:8.3f}'.format(e) for e in sample_energies]),']')

        if hasattr(self.model_calculator,'ever_trained') and self.model_calculator.ever_trained:
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

    def assign_to_best_sample(self,candidate_object):

        if len(self.sample) == 0:
            return None
        
        best_index = 0

        cluster_center = self.sample[best_index]
        cluster_center.add_meta_information('index_in_sampler_list_used_for_printing_purposes_only',best_index)

        return cluster_center

    def assign_to_closest_sample(self,candidate_object):

        if len(self.sample) == 0:
            return None
        
        # find out what cluster we belong to
        f_this = self.get_features([candidate_object])

        # assume that this is a new cluster that should be compared in energy to the highest one so far
        d_min_index = len(self.sample_indices) - 1

        # run through existing clusters in ascending energy order and stop first time we belong to one
        #   if no match, we already were assigned to the highest energy one
        for li,idx in enumerate(self.sample_indices):
            similarity = self.model_calculator.model.kernel_.get_kernel(self.f_all[idx],f_this) / self.K0
            if similarity > self.similarity_criteria[li]:
                d_min_index = li
                break

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
                F = candidate.get_meta_information('kernel_sim_sampler_feature')
                if F is None:
                    F = self.feature_calculator.get_feature(candidate)
                    candidate.add_meta_information('kernel_sim_sampler_feature', F)
                features.append(F)
            features = np.array(features)
        else:
            features = np.array(self.feature_calculator.get_global_features(structures))
        return features
