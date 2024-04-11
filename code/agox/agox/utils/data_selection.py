import numpy as np
from abc import ABC, abstractmethod

class DataSelector(ABC):

    def __init__(self, replace=False):
        self.replace = replace

    def __call__(self, data, N, return_indices=False):
        p = self.make_probability(data, N)
            
        indices = np.random.choice(np.arange(len(data)), p=p, size=N, replace=self.replace)
        selected_data = [data[i] for i in indices]
        if return_indices:
            return selected_data, indices
        return selected_data

    @abstractmethod
    def make_probability(self, data, N):
        pass

class UniformSelector(DataSelector):

    def make_probability(self, data, N):
        return np.ones(len(data)) * 1 / len(data)

class SegmentedSelector(DataSelector):

    def make_probability(self, data, N):
        p = np.zeros(len(data))
        energies = np.array([atoms.get_potential_energy() for atoms in data])
        sorted_indices = np.argsort(energies)
        interval = int(np.floor(len(data) / N))
        p[sorted_indices[0:-1:interval]] = 1
        return p / np.sum(p)

class EnergyUniformSelector(DataSelector):

    def __init__(self, max_delta, bin_width=1, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta
        self.bin_width = bin_width

    def make_probability(self, data, N):
        energies = np.array([a.get_potential_energy() for a in data])

        e_min = energies.min()
        e_max = np.min([e_min+self.max_delta, energies.max()])
        bins = np.arange(e_min, e_max, self.bin_width) + self.bin_width / 2
        bin_count, bin_indices = self.get_bin_count(bins, energies)

        effective_nbins = np.count_nonzero(bin_count!=0)
        bin_prob = 1 / (effective_nbins - 1)

        bin_probs = np.zeros_like(bins)
        for i, bc in enumerate(bin_count):
            if bc != 0:
                p = bin_prob / bc
            else:
                p = 0
            bin_probs[i] = p

        bin_probs[-1] = 0

        probabilities = np.zeros_like(energies)
        for i, bin_index in enumerate(bin_indices):
            probabilities[i] = bin_probs[bin_index]

        return probabilities
        
    def get_bin_count(self, bins, energies):
        bin_count = np.zeros_like(bins).astype(int)
        bins = bins.copy().reshape(-1, 1)
        energies = energies.copy().reshape(1, -1)
        indices = np.argmin(np.abs(bins - energies), axis=0)
        for index in indices: 
            bin_count[index] += 1

        return bin_count, indices

def split(data, fractions):
    N_total = len(data)
    counts = [int(frac * N_total) for frac in fractions]
    counts[-1] += N_total - np.sum(counts)
    counts = [0] + counts
    counts = np.cumsum(counts)

    indices = np.arange(N_total)
    np.random.shuffle(indices)

    splits = []
    for i in range(len(fractions)):
        splits.append([data[j] for j in indices[counts[i]:counts[i+1]]])

    return splits

def exclude(data, indices):
    return [data[i] for i in range(len(data)) if i not in indices]

def filter_energy(data, max_delta):
    energies = np.array([a.get_potential_energy() for a in data])
    indices = np.argwhere(energies < energies.min()+max_delta).flatten()
    return [data[i] for i in indices]



        



