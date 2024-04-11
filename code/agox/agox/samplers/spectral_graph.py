from .ABC_sampler import SamplerBaseClass
import numpy as np
from ase.io import write
from ase.data import covalent_radii

class SpectralGraphSampler(SamplerBaseClass):

    
    name = 'SpectralGraphSampler'

    def __init__(self, model_calculator, indices=None, debug=False, 
        sample_size=10, covalent_bond_scale_factor=1.3, **kwargs):
        super().__init__(**kwargs)
        self.indices = indices
        self.sample_size = sample_size
        self.covalent_bond_scale_factor = covalent_bond_scale_factor
        self.debug = debug
        self.model_calculator = model_calculator

    def setup(self, all_finished_structures):
        if self.debug:
            write('all_finished_structures_{:06d}.traj'.format(self.get_iteration_counter()),all_finished_structures)
        if self.verbose:
            print('len(all_finished_structures):',len(all_finished_structures))
        if len(all_finished_structures) == 0:
            return

        self.template = all_finished_structures[0].template.copy()
        all_finished_structures.sort(key=lambda x: x.get_potential_energy())

        sample_indices = []
        self.sample_eigen_strings = []
        for i,t in enumerate(all_finished_structures):
            s = self.get_eigen_value_string(t,allow_read_from_meta_information=True)
            if s not in self.sample_eigen_strings:
                sample_indices.append(i)
                self.sample_eigen_strings.append(s)
                print('SAMPLE index',i,s)
            if len(sample_indices) >= self.sample_size:
                break
        
        self.sample = [all_finished_structures[i] for i in sample_indices]
        if self.verbose:
            write('sample_{:06d}.traj'.format(self.get_iteration_counter()),self.sample)
        sample_energies = [t.get_potential_energy() for t in self.sample]
        print('SAMPLE_DFT_ENERGY  ', '[',','.join(['{:8.3f}'.format(e) for e in sample_energies]),']')

        if hasattr(self.model_calculator,'ever_trained') and self.model_calculator.ever_trained:
            for s in self.sample:
                t = s.copy()
                t.set_calculator(self.model_calculator)
                E = t.get_potential_energy()
                sigma = self.model_calculator.get_property('uncertainty')
                s.add_meta_information('model_energy',E)
                s.add_meta_information('uncertainty',sigma)
            print('SAMPLE_MODEL_ENERGY', \
                  '[',','.join(['{:8.3f}'.format(t.get_meta_information('model_energy')) for t in self.sample]),']')
            print('SAMPLE_MODEL_SIGMA', \
                  '[',','.join(['{:8.3f}'.format(t.get_meta_information('uncertainty')) for t in self.sample]),']')

    def assign_to_closest_sample(self,candidate_object):

        if len(self.sample) == 0:
            return None
        
        # find out what cluster we belong to
        s = self.get_eigen_value_string(candidate_object)

        if s in self.sample_eigen_strings:
            i = self.sample_eigen_strings.index(s)
        else:
            i = len(self.sample) - 1

        cluster_center = self.sample[i]
        cluster_center.add_meta_information('index_in_sampler_list_used_for_printing_purposes_only',i)
        return cluster_center

    def adjust_sample_cluster_distance(self, chosen_candidate, closest_sample):
        # See if this structure had a lower energy than the lowest energy in the cluster it belongs to
        chosen_candidate_energy = chosen_candidate.get_potential_energy()
        closest_sample_energy = closest_sample.get_potential_energy()
        d_min_index = closest_sample.get_meta_information('index_in_sampler_list_used_for_printing_purposes_only')
        print('CLUST_RES {:06d}'.format(self.get_iteration_counter()),
              '[',','.join(['{:8.3f}'.format(t.get_potential_energy()) for t in self.sample]),
              '] {:8.3f} {:8.3f}'.format(closest_sample_energy,chosen_candidate_energy),
              'NEW {:d} {}'.format(d_min_index,chosen_candidate.get_meta_information('description')) \
              if closest_sample_energy > chosen_candidate_energy else '')

    def get_distances(self,original_candidate,indices):
        candidate = original_candidate.copy()
        del candidate[[i for i in range(len(candidate)) if i not in indices]]
        distances_abs = candidate.get_all_distances(mic=True)

        numbers = candidate.get_atomic_numbers()
        r = [covalent_radii[number] for number in numbers]
        x,y = np.meshgrid(r,r)
        optimal_distances = x+y
    
        distances_rel = distances_abs / optimal_distances
    
        return distances_rel

    def get_bond_matrix(self,candidate):
        dist = self.get_distances(candidate,self.indices)
        matrix = np.logical_and(dist > 1e-3, dist < self.covalent_bond_scale_factor).astype(int)
        matrix += np.diag(candidate.get_atomic_numbers()[self.indices])
        return matrix

    def convert_matrix_to_eigen_value_string(self,matrix):
        w,_ = np.linalg.eig(matrix)
        w = np.real(w)
        w.sort()
        s = '[' + ','.join(['{:8.3f}'.format(e) for e in w]) + ']'
        s = s.replace('-0.000',' 0.000')
        return s

    def get_eigen_value_string(self, t, allow_read_from_meta_information=False):
        # this is conditioned that the meta information is trusted (corrupted when Candidates are copied and changed)
        if allow_read_from_meta_information:
            if t.has_meta_information('eigen_string'):
                return t.get_meta_information('eigen_string')

        if self.indices is None:
            self.indices = list(range(len(self.template),len(t)))
            print('SETTING INDICES:',self.indices)

        m = self.get_bond_matrix(t)
        if self.debug:
            print(m)
        s = self.convert_matrix_to_eigen_value_string(m)
        #if t.has_meta_information('eigen_string'):
        #    assert s == t.get_meta_information('eigen_string'),'eigen string has changed: A:{}, B:{}, C:{}'.format(s,t.get_meta_information('eigen_string'),m)
        t.add_meta_information('eigen_string',s)
        return s