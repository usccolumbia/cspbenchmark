import numpy as np
from agox.models.local_GPR.LSGPR import LSGPRModel
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.validation import check_random_state
from time import time

class LSGPRModelMBKMeans(LSGPRModel):
    name = 'LSGPRModelMBKMeans'

    def __init__(self, m_distribution=None, include_best=True, include_transfer=False, batch_size=1024,
                 sparse_update=False, full_update_interval=10, cluster_weights=None,
                 exact_points=False, seed=None, **kwargs):
        """ m_distribution must be given as dict of energy:number e.g. {0.5:100, 1:100, 2:50, 50:50}
        """
        super().__init__(**kwargs)
        self.include_best = include_best
        self.include_transfer = include_transfer
        self.batch_size = batch_size
        self.sparse_update = sparse_update
        self.full_update_interval = full_update_interval
        self.cluster_weights = cluster_weights

        # tmp:
        self.exact_points = exact_points


        self.cluster = MiniBatchKMeans(n_clusters=self.m_points, batch_size=batch_size, random_state=seed,
                                       init='k-means++', n_init=3)
    

    def _train_sparse(self, atoms_list, **kwargs):
        if self.Xn.shape[0] < self.m_points:
            self.Xm = self.Xn
            return True

        try:
            full_update = self.get_iteration_counter()%self.full_update_interval == 0
        except:
            full_update = True

        if hasattr(self.cluster, 'cluster_centers_') and self.sparsifier is None and not full_update:
            self._MB_episode()
        elif hasattr(self.cluster, 'cluster_centers_') and self.sparsifier is not None and self.sparse_update:
            self._MB_episode()
        else:
            if self.cluster_weights is not None:
                sample_weights = np.array([[self.cluster_weights[i]]*len(atoms_list[i]) for i in range(len(atoms_list))]).flatten()
                self.cluster.fit(self.Xn, sample_weight=sample_weights)
            else:
                self.cluster.fit(self.Xn)
            self.writer(f'KMeans Episodes: {self.cluster.n_steps_}')    

            

        if self.exact_points:
            dists = self.cluster.transform(self.Xn)
            min_idx = np.argmin(dists, axis=0)
            self.Xm = self.Xn[min_idx, :]
        else:
            self.Xm = self.cluster.cluster_centers_
            
        return True


    def _MB_episode(self):
        # do 1 episode
        self.writer(f'KMeans partial update')
        n_samples = self.Xn.shape[0]
        batch_size = min(self.batch_size, self.Xn.shape[0])
        steps = n_samples // batch_size
        random_state = np.random.RandomState()
        for _ in range(steps):
            minibatch_indices = random_state.randint(0, n_samples, batch_size)
            self.cluster.partial_fit(self.Xn[minibatch_indices])
            

    def sparse_plot(self, name=''):
        from sklearn.decomposition import PCA
        from agox.utils.matplotlib_utils import use_agox_mpl_backend; use_agox_mpl_backend()
        import matplotlib.pyplot as plt
        pca = PCA(n_components=2)
        pca.fit(self.Xn)
        l = pca.transform(self.Xn)
        s = pca.transform(self.Xm)
        fig, ax = plt.subplots(ncols=1, figsize=(5,5))
        ax.scatter(l[:,0], l[:,1], marker='.', alpha=0.7, color='b')
        ax.scatter(s[:,0], s[:,1], marker='x', alpha=0.5, color='r')
        plt.savefig(name+'_sparse_plot.png')
            
    @classmethod
    def default_model(cls, species, single_atom_energies):
        """ 
        - species must be list of symbols or numbers
        - single_atom_energies must be list of floats corresponding to ordering in species or array of 0's
        except at positions of numbers corresponding to species.
        """
        from agox.models.descriptors.soap import SOAP
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        from ase.atoms import symbols2numbers
        
        if len(species) == len(single_atom_energies):
            sae = np.zeros(100)
            numbers = symbols2numbers(species)
            for num, e in zip(numbers,single_atom_energies):
                sae[num] = e
        else:
            sae = single_atom_energies
            
        descriptor = SOAP(species, r_cut=3, nmax=3, lmax=2, sigma=1, normalize=False, use_radial_weighting=True, periodic=True)
        kernel = C(1)*RBF(length_scale=20)
        return cls(kernel=kernel, descriptor=descriptor, single_atom_energies=sae, noise=0.05)


    
