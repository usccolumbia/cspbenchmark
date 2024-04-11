import numpy as np
from agox.writer import agox_writer
from agox.samplers import SamplerBaseClass
from agox.observer import Observer
from ase.io import read, write
from copy import deepcopy

class GeneticSampler(SamplerBaseClass):

    name = 'GeneticSampler'

    def __init__(self, comparator=None, population_size=10, gets={'get_key':'evaluated_candidates'}, order=6, 
                using_add_all=True, **kwargs):        
        super().__init__(gets=gets, order=order, **kwargs)

        assert comparator is not None
        self.comparator = comparator
        self.population = []
        self.population_size = population_size
        self.sample = self.population
        self.idx = 0
        self.using_add_all = using_add_all

    def get_random_member(self):
        if len(self.sample) == 0:
            return None

        P = self.get_selection_probabilities()
        idx = np.random.choice(np.arange(len(self.population)), 1, p=P)[0]
        member = self.population[idx]
        self.increase_times_used(member)
        return member.copy()

    def get_random_member_with_calculator(self):
        if len(self.sample) == 0:
            return None

        P = self.get_selection_probabilities()
        idx = np.random.choice(np.arange(len(self.population)), 1, p=P)[0]
        member = self.population[idx]
        self.increase_times_used(member)
        member = member.copy()
        self.population[idx].copy_calculator_to(member)
        return member

    def increase_times_used(self, member):
        count = member.get_meta_information('used_count')
        if count is None:
            count = 0
        else:
            count += 1
        member.add_meta_information('used_count', count)

    def count_similar(self, member, all_candidates):

        count = 0
        for candidate in all_candidates:            
            if candidate.get_meta_information('final'):
                if self.comparator(member, candidate):
                    count += 1
            
        member.add_meta_information('similar_count', count)

    def get_recent_candidates(self, state):
        evaluated_candidates = state.get_from_cache(self, self.get_key)

        if not self.using_add_all:
            return evaluated_candidates
        else:
            recent = []
            for candidate in evaluated_candidates:
                if candidate.get_meta_information('final'): # This is True for the final candidate from a trajectory with LocalOpt evaluator
                    recent.append(deepcopy(candidate))

            E = [atoms.get_potential_energy() for atoms in recent]
            return recent


    @agox_writer
    @Observer.observer_method
    def setup_sampler(self, database, state):
        if self.do_check():
            possible_candidates = self.get_recent_candidates(state)
            all_candidates = database.get_all_candidates()
            self.setup(all_candidates, possible_candidates)
        self.sample = self.population # Lazy renaming, as sample-attribute is used by some methods of the base-class.

    def setup(self, all_candidates, possible_candidates):        
        # If the population is empty we just take all of them for now: 
        if len(self.population) < self.population_size:
            self.population += [possible_candidates.pop(0) for i in range(min(self.population_size, len(possible_candidates)))]
            self.sort_population()
            if len(possible_candidates) == 0:
                return 

        # Now we start deciding whether to replace candidates in the population: 
        for candidate in possible_candidates:
            state = self.consider_candidate(candidate)
            if state:
                self.sort_population()
        
        # Compare population to database:
        for member in self.population:
            self.count_similar(member, all_candidates)

        if self.get_iteration_counter() % 5 == 0:
            write('population_{}.traj'.format(self.get_iteration_counter()), self.population)

        self.print_information()

    def consider_candidate(self, candidate):        
        fitness = self.get_fitness(candidate)

        worst_fitness = self.get_fitness(self.population[-1])

        if fitness < worst_fitness and len(self.population) == self.population_size:
            return False

        for i, member in enumerate(self.population):
            if self.comparator(candidate, member):
                if fitness > self.get_fitness(member):

                    used_count = member.get_meta_information('used_count')
                    
                    del self.population[i]
                    candidate.add_meta_information('used_count', used_count) # Replaced this member so inherit its use count. 
                    self.population.append(candidate)
                    return True
                # If it looks like another member we return regardless of whether it replaces that member. 
                return False

        # If it doesn't look like anything we just replace the worst member of the population:
        del self.population[-1]
        self.population.append(candidate)
    
        return True

    def get_fitness(self, candidate):
        population_energies = [candidate.get_potential_energy() for candidate in self.population]
        e_min = np.min(population_energies)
        e_max = np.max(population_energies)

        p = (candidate.get_potential_energy() - e_min) / (e_max - e_min)

        F = 0.5 * (1 - np.tanh(2*p - 1))

        return F
    
    def sort_population(self):
        if len(self.population):
            fitness = [self.get_fitness(candidate) for candidate in self.population]
            sort_idx = np.argsort(fitness)
            self.population = [self.population[i] for i in sort_idx][::-1]


    def get_selection_probabilities(self):
        N = np.array([member.get_meta_information('used_count') for member in self.population])
        N[N == None] = 0
        N = N.astype(int)
        M = np.array([member.get_meta_information('similar_count') for member in self.population])
        M[M == None] = 0
        M = M.astype(int)
        F = np.array([self.get_fitness(member) for member in self.population])
        U = 1 / np.sqrt(M+1) * 1 / np.sqrt(N + 1)
        P = F * U 
        P = P / np.sum(P)
        return P 

    def print_information(self):
        probs = self.get_selection_probabilities()
        for i, member in enumerate(self.population):
            E = member.get_potential_energy()
            F = self.get_fitness(member)
            P = probs[i]
            self.writer('Member {}: E = {:6.4f}, F = {:6.4f}, P = {:4.2f}'.format(i, E, F, P))

class DistanceComparator:

    def __init__(self, descriptor, threshold):
        self.descriptor = descriptor
        self.threshold = threshold

    def __call__(self, candidate_A, candidate_B):
        return self.compare_candidates(candidate_A, candidate_B)

    def compare_candidates(self, candidate_A, candidate_B):
        feature_A = self.get_feature(candidate_A)
        feature_B = self.get_feature(candidate_B)

        return np.linalg.norm(feature_A-feature_B) < self.threshold

    def get_feature(self, candidate):
        feature = candidate.get_meta_information('population_feature')

        if feature is None:
            feature = self.descriptor.get_global_features(candidate)[0]
            candidate.add_meta_information('population_feature', feature)
        
        return feature


