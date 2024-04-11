import numpy as np
from agox.databases.database_concurrent import ConcurrentDatabase


class OrderedDatabase(ConcurrentDatabase):
        
    def restore_to_memory(self):
        strucs = self.get_all_structures_data()
        if self.store_meta_information:
            all_meta_info = self.read_all_key_value_pairs()
        else:
            all_meta_info = {}

        all_candidates = []
        for struc in strucs:            
            all_candidates.append(self.db_to_candidate(struc, meta_dict=all_meta_info.get(struc['id'], None)))

        iteration = self.get_iteration_counter()
        
        new_candidates = [c for c in all_candidates if c.get_meta_information('iteration') > iteration - self.sync_frequency]

        old_candidates = [c for c in self.candidates if c.get_meta_information('iteration') <= iteration - self.sync_frequency]

        worker_numbers = np.array([c.get_meta_information('worker_number') != self.worker_number for c in new_candidates],
                                  dtype=int)
        iterations = np.array([c.get_meta_information('iteration') for c in new_candidates])
        relax_indices = np.array([c.get_meta_information('relax_index') for c in new_candidates])
        indices = np.lexsort((iterations, worker_numbers))

        self.candidates =  old_candidates + [new_candidates[i] for i in indices]

        if self.verbose > 1:
            for c in self.candidates:
                self.writer(f"worker number: {c.get_meta_information('worker_number')}, iteration: {c.get_meta_information('iteration')}, relax_index: {c.get_meta_information('relax_index')}")        
