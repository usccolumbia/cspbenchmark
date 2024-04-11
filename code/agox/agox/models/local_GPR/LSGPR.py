import numpy as np
from os.path import join
from copy import copy
import pickle
import random

from numpy.random import default_rng
from itertools import product

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from scipy.linalg import cholesky, cho_solve, solve_triangular, qr, lstsq, LinAlgError, svd, eig, eigvals, det
from scipy.spatial.distance import cdist
from scipy.optimize import fmin_l_bfgs_b

from agox.models.ABC_model import ModelBaseClass
from time import time

from agox.writer import agox_writer
from agox.observer import Observer


class LSGPRModel(ModelBaseClass):
    name = 'LSGPRModel'
    
    implemented_properties = ['energy', 'forces']

    dynamic_attributes = ['Xm', 'K_inv', 'Kmm_inv', 'alpha']

    """ Local GPR model with uniform sparsification

    Attributes
    ----------
    kernel : Scikit-learn Kernel 
        Local kernel object.
    descriptor : DScribe descriptor
        descriptor object
    noise : float
        Noise in eV/atom
    prior : Callable
        Prior expectation for prediction. 
    single_atom_energies : np.ndarray or Dict
        If np.ndarray it must be array of number for 
        single atom energies for species e.g. index 1 must be E for hydrogen.
    m_points : int
        Number of sparse points maximally used by model.
    transfer_data : List[Candidate] or List[Atoms]
        Training data not taken from a database, when used in an active learning 
        scenario such as AGOX structure searches.
    sparsifier : Callable
        Object, which sparsifies training data on Candidate/Atoms level. 
    jitter : float
        Small number for numerical stability. If model training is not stable
        try increasing this number. 
    trainable_prior : bool
        Whether the prior can and should be trained on the model training data.
    use_prior_in_training : bool
        Wheter the prior expectation should be subtracted before model training.
    
    
    """    

    def __init__(self, kernel=None, descriptor=None, noise=0.05,
                 prior=None, single_atom_energies=None,
                 m_points=1000, transfer_data=[], sparsifier=None,
                 jitter=1e-8, trainable_prior=False, use_prior_in_training=False,
                 **kwargs):
        
        super().__init__(**kwargs)

        self.kernel = kernel
        self.descriptor = descriptor         
        self._noise = noise
        self.single_atom_energies = single_atom_energies

        self.transfer_data = transfer_data # add additional data not generated during run
        self.jitter = jitter   # to help inversion

        self.m_points = m_points
        self.sparsifier = sparsifier

        self.prior = prior
        self.trainable_prior = trainable_prior
        self.use_prior_in_training = use_prior_in_training

        self.method = 'lstsq'
        
        # Initialize model parameters
        self.alpha = None
        self.Xn = None
        self.Xm = None
        self.m_indices = None
        self.y = None

        self.K_mm = None
        self.K_nm = None
        self.K_inv = None
        self.Kmm_inv = None
        self.L = None

        
    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, s):
        self._noise = s

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, name):
        if name in ['QR', 'cholesky', 'lstsq']:
            self._method = name
        else:
            self.writer('Method not known - use: QR, cholesky or lstsq')

    @property
    def single_atom_energies(self):
        return self._single_atom_energies

    @single_atom_energies.setter
    def single_atom_energies(self, s):
        if isinstance(s, np.ndarray):
            self._single_atom_energies = s
        elif isinstance(s, dict):
            self._single_atom_energies = np.zeros(100)
            for i, val in s.items():
                self._single_atom_energies[i] = val
        elif s is None:
            self._single_atom_energies = np.zeros(100)

    @property
    def transfer_data(self):
        return self._transfer_data

    @transfer_data.setter
    def transfer_data(self, l):
        if isinstance(l, list):
            self._transfer_data = l
            self._transfer_weights = np.ones(len(l))
        elif isinstance(l, dict):
            self._transfer_data = []
            self._transfer_weights = np.array([])
            for key, val in l.items():
                self._transfer_data += val
                self._transfer_weights = np.hstack((self._transfer_weights, float(key) * np.ones(len(val)) ))
        else:
            self._transfer_data = []
            self._trasfer_weights = np.array([])

    @property
    def transfer_weights(self):
        return self._transfer_weights

    
    def set_verbosity(self, verbose):
        self.verbose = verbose
    
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)        
        if 'energy' in properties:
            e = self.predict_energy(atoms=atoms)
            self.results['energy'] = e
        
        if 'forces' in properties:
            self.results['forces'] = self.predict_forces(atoms=atoms)

    ####################################################################################################################
    # Prediction
    ####################################################################################################################

    def predict_energy(self, atoms=None, X=None, return_uncertainty=False):
        if X is None:
            X = np.vstack(self.descriptor.get_local_features(atoms))

        if self.prior is not None:
            e0 = self.prior.predict_energy(atoms)
        else:
            e0 = 0

        k = self.kernel(self.Xm, X)
        if not return_uncertainty:
            return np.sum(k.T@self.alpha) + sum(self.single_atom_energies[atoms.get_atomic_numbers()]) + e0
        else:
            unc = self.predict_uncertainty(atoms=atoms, k=k)
            return np.sum(k.T@self.alpha) + sum(self.single_atom_energies[atoms.get_atomic_numbers()]) + e0, unc


    def predict_energies(self, atoms_list):
        energies = np.array([self.predict_energy(atoms) for atoms in atoms_list])
        return energies
            

    def predict_uncertainty(self, atoms=None, X=None, k=None):
        self.writer('Uncertainty not implemented.')
        return 0.

        
    def predict_local_energy(self, atoms=None, X=None):
        """
        Calculate the local energies in the model. 
        """
        if X is None:
            X = np.vstack(self.descriptor.get_local_features(atoms))

        k = self.kernel(self.Xm, X)
        return (k.T@self.alpha).reshape(-1,) + self.single_atom_energies[atoms.get_atomic_numbers()]



    def predict_forces(self, atoms, return_uncertainty=False, **kwargs):
        """
        """
        f = self.predict_forces_central(atoms, **kwargs)

        if return_uncertainty:
            return f, np.zeros(f.shape)
        else:
            return f


    ####################################################################################################################
    # Training:
    # 3 levels:
    #     public: train_model
    #     changable in subclass: _train_sparse (sets: self.Xn, self.Xm, self.L, self.y)
    #         return: <boolean> training_nessesary
    #     private: _train_GPR (asserts that self.Xn, self.Xm, self.L, self.y is set)
    ####################################################################################################################

    @agox_writer
    def train_model(self, training_data, **kwargs):
        self.ready_state = True
        self.atoms = None

        # train prior
        if self.prior is not None and self.trainable_prior:
            data = self.transfer_data + training_data
            energies = np.array([atoms.get_potential_energy() - sum(self.single_atom_energies[atoms.get_atomic_numbers()]) \
                                 for atoms in data])
            self.prior.train_model(data)

        # prepare for training
        t1 = time()
        self.Xn, self.y = self._get_X_y(self.transfer_data + training_data)
        self.L = self._make_L(self.transfer_data + training_data)
        self.sigma_inv = self._make_sigma(self.transfer_data + training_data)
        t2 = time()

        training_nessesary = self._train_sparse(atoms_list=self.transfer_data + training_data, **kwargs)
        if self.verbose:
            self.writer('=========== MODEL INFO ===========')
            self.writer(f'Number of energies available: {len(self.y)}')
            self.writer(f'Number of local environments: {self.Xn.shape[0]}')
            self.writer(f'Number of basis points:       {self.Xm.shape[0]}')
        if training_nessesary:
            t3 = time()
            self._train_GPR()
            t4 = time()
            if self.verbose:
                self.writer('========== MODEL TIMING ==========')
                self.writer(f'Total:          {t4-t1:.3f} s')
                self.writer(f'Features:       {t2-t1:.3f} s')
                self.writer(f'Sparsification: {t3-t2:.3f} s')
                self.writer(f'Kernel:         {self.kernel_timing:.3f} s')
                self.writer(f'Training:       {t4-t3-self.kernel_timing:.3f} s')


    def _train_sparse(self, atoms_list, **kwargs):
        """
        sparsification scheme: must set self.Xm
        returns boolean: indicates if training nessesary. 
        """
        if self.m_points > self.Xn.shape[0]:
            m_indices = np.arange(0,self.Xn.shape[0])
        else:
            m_indices = np.random.choice(self.Xn.shape[0], size=self.m_points, replace=False)
        self.Xm = self.Xn[m_indices, :]
        return True
    
    def _train_GPR(self):
        """
        Fit Gaussian process regression model.
        Assert self.Xn, self.Xm, self.L and self.y is not assigned
        """
        assert self.Xn is not None, 'self.Xn must be set prior to call'
        assert self.Xm is not None, 'self.Xm must be set prior to call'
        assert self.L is not None, 'self.L must be set prior to call'
        assert self.y is not None, 'self.y must be set prior to call'

        t1 = time()
        self.K_mm = self.kernel(self.Xm)
        self.K_nm = self.kernel(self.Xn, self.Xm)
        t2 = time()
        self.kernel_timing = t2-t1
        
        LK_nm = self.L @ self.K_nm # This part actually also takes a long time to calculate - change in future
        K = self.K_mm + LK_nm.T @ self.sigma_inv @ LK_nm + self.jitter*np.eye(self.K_mm.shape[0])

        # if self.uncertainty_method != 'SR':
        #     cho_Kmm = cholesky(self.K_mm+self.jitter*np.eye(self.K_mm.shape[0]), lower=True)
        #     self.Kmm_inv = cho_solve((cho_Kmm, True), np.eye(self.K_mm.shape[0]))

        if self.method == 'cholesky':
            t1 = time()
            K = self.nearestPD(K)
            t2 = time()
            cho_low = cholesky(K, lower=True)
            t3 = time()
            self.K_inv = cho_solve((cho_low, True), np.eye(K.shape[0]))
            t4 = time()
            self.alpha = cho_solve((cho_low, True), LK_nm.T @ self.sigma_inv @ self.y.reshape(-1,1))
            t5 = time()
            
            if self.verbose > 1:
                self.writer('========= FOR DEBUGGING ==========')
                self.writer(f'PSD Kernel timing:             {t2-t1:.3f} s')
                self.writer(f'Cholesky decomposition timing: {t3-t2:.3f} s')
                self.writer(f'K inversion timing:            {t4-t3:.3f} s')
                self.writer(f'alpha solve timing:            {t5-t4:.3f} s')
                self.writer('')
                self.writer(f'Cholesky residual:  {np.linalg.norm(cho_low.dot(cho_low.T) - K):.5f}')
                residual = np.linalg.norm((K @ self.K_inv)-np.eye(K.shape[0]))
                self.writer(f'K inverse residual: {residual:.5f}')                
                residual = np.linalg.norm((K@self.alpha)-LK_nm.T @ self.sigma_inv @ self.y.reshape(-1,1))
                self.writer(f'alpha residual:     {residual:.5f}')

        elif self.method == 'QR':
            t1 = time()
            K = self.symmetrize(K)
            t2 = time()
            Q, R = qr(K)
            t3 = time()
            self.K_inv = solve_triangular(R, Q.T)
            t4 = time()
            self.alpha = solve_triangular(R, Q.T @ LK_nm.T @ self.sigma_inv @ self.y.reshape(-1,1))
            t5 = time()
            
            if self.verbose > 1:
                self.writer('========= FOR DEBUGGING ==========')
                self.writer(f'Symmetrize Kernel timing:{t2-t1:.3f} s')
                self.writer(f'QR decomposition timing: {t3-t2:.3f} s')
                self.writer(f'K inversion timing:      {t4-t3:.3f} s')
                self.writer(f'alpha solve timing:      {t5-t4:.3f} s')
                self.writer('')
                self.writer(f'QR residual: {np.linalg.norm(Q@R - K):.5f}')                
                residual = np.linalg.norm((K @ self.K_inv)-np.eye(K.shape[0]))
                self.writer(f'K inverse residual: {residual:.5f}')                
                residual = np.linalg.norm((K@self.alpha)-LK_nm.T @ self.sigma_inv @ self.y.reshape(-1,1))
                self.writer(f'alpha residual:     {residual:.5f}')

        elif self.method == 'lstsq':
            t1 = time()
            K = self.symmetrize(K)
            t2 = time()
            Q, R = qr(K)
            t3 = time()
            self.K_inv = lstsq(R, Q.T)[0]
            t4 = time()
            self.alpha = lstsq(R, Q.T @ LK_nm.T @ self.sigma_inv @ self.y.reshape(-1,1))[0]
            t5 = time()
            
            if self.verbose > 1:
                self.writer('========= FOR DEBUGGING ==========')
                self.writer(f'Symmetrize Kernel timing:{t2-t1:.3f} s')
                self.writer(f'QR decomposition timing: {t3-t2:.3f} s')
                self.writer(f'K inversion timing:      {t4-t3:.3f} s')
                self.writer(f'alpha solve timing:      {t5-t4:.3f} s')
                self.writer('')
                self.writer(f'QR residual: {np.linalg.norm(Q@R - K):.5f}')                
                residual = np.linalg.norm((K @ self.K_inv)-np.eye(K.shape[0]))
                self.writer(f'K inverse residual: {residual:.5f}')                
                residual = np.linalg.norm((K@self.alpha)-LK_nm.T @ self.sigma_inv @ self.y.reshape(-1,1))
                self.writer(f'alpha residual:     {residual:.5f}')            

        else:
            self.writer(f'method name: {self.method} unknown. Will fail shortly')
            
    @agox_writer
    def update_model(self, new_data, all_data):
        self.atoms = None
        
        t1 = time()

        self.L = self._update_L(new_data)
        self.Xn, self.y = self._update_X_y(new_data)            
        self.sigma_inv = self._make_sigma(self.transfer_data + all_data)
        t2 = time()

        training_nessesary = self._train_sparse(atoms_list=self.transfer_data + all_data)
        t3 = time()
        
        if self.verbose:
            self.writer('=========== MODEL INFO ===========')
            self.writer(f'Number of energies available: {len(self.y)}')
            self.writer(f'Number of local environments: {self.Xn.shape[0]}')
            self.writer(f'Number of basis points:       {self.Xm.shape[0]}')
            
        if training_nessesary:
            self._train_GPR()
            t4 = time()
            if self.verbose:
                self.writer('========== MODEL TIMING ==========')
                self.writer(f'Total:          {t4-t1:.3f} s')
                self.writer(f'Features:       {t2-t1:.3f} s')
                self.writer(f'Sparsification: {t3-t2:.3f} s')
                self.writer(f'Kernel:         {self.kernel_timing:.3f} s')
                self.writer(f'Training:       {t4-t3-self.kernel_timing:.3f} s')
        

                
                 
    ####################################################################################################################
    # Assignments:
    ####################################################################################################################

    @agox_writer
    @Observer.observer_method        
    def training_observer(self, database, state):
        iteration = state.get_iteration_counter()

        if iteration < self.iteration_start_training:
            return
        if (iteration % self.update_period != 0) * (iteration != self.iteration_start_training):
            return


        all_data = database.get_all_candidates()
        self.writer(f'lenght all data: {len(all_data)}')
        
        if self.sparsifier is not None:
            full_update, data_for_training = self.sparsifier(all_data)
        elif self.ready_state:
            full_update = False
            data_amount_before = len(self.y) - len(self.transfer_data)
            data_for_training = all_data
            data_amount_new = len(data_for_training) - data_amount_before
            new_data = data_for_training[-data_amount_new:] 
        else:
            full_update = True
            data_for_training = all_data

        if full_update:
            self.train_model(data_for_training)
        else:
            self.update_model(new_data, data_for_training)
        

        
    ####################################################################################################################
    # Helpers
    ####################################################################################################################


    def _get_X_y(self, atoms_list):
        X = np.vstack(self.descriptor.get_local_features(atoms_list))
        y = np.array([atoms.get_potential_energy() - sum(self.single_atom_energies[atoms.get_atomic_numbers()]) \
                      for atoms in atoms_list])
        if self.prior is not None and self.use_prior_in_training:
            y -= np.array([self.prior.predict_energy(atoms) for atoms in atoms_list])
            
        return X, y

    def _update_X_y(self, new_atoms_list):        
        number_of_new_environments = np.sum([len(atoms) for atoms in new_atoms_list])
        number_of_new_energies = len(new_atoms_list)
        X = np.zeros((self.Xn.shape[0]+number_of_new_environments, self.Xn.shape[1]))
        X[0:self.Xn.shape[0]] = self.Xn
        X[self.Xn.shape[0]:] = np.vstack(self.descriptor.get_local_features(new_atoms_list))
        y = np.zeros(self.y.shape[0]+number_of_new_energies)
        y[0:self.y.shape[0]] = self.y
        y[-number_of_new_energies:] = [atoms.get_potential_energy() - sum(self.single_atom_energies[atoms.get_atomic_numbers()]) for atoms in new_atoms_list]
        return X, y

    def _make_L(self, atoms_list):
        assert self.y is not None, 'self.y cannot be None'
        assert self.Xn is not None, 'self.Xn cannot be None'
        col = 0
        L = np.zeros((len(self.y), self.Xn.shape[0]))
        for i, atoms in enumerate(atoms_list):
            L[i,col:col+len(atoms)] = 1.
            col += len(atoms)
        return L

    def _update_L(self, new_atoms_list):
        new_lengths = [len(atoms) for atoms in new_atoms_list]
        size = len(new_lengths)
        new_total_length = np.sum(new_lengths)
        new_L = np.zeros((self.L.shape[0]+size, self.L.shape[1]+new_total_length))
        new_L[0:self.L.shape[0], 0:self.L.shape[1]] = self.L

        for l in range(size):
            step = int(np.sum(new_lengths[:l]))
            new_L[l+self.L.shape[0], (self.L.shape[1]+step):(self.L.shape[1]+step+new_lengths[l])] = 1            
        return new_L

    def _make_sigma(self, atoms_list):
        sigma_inv = np.diag([1/(len(atoms)*self.noise**2) for atoms in atoms_list])
        weights = np.ones(len(atoms_list))
        weights[:len(self.transfer_weights)] = self.transfer_weights
        sigma_inv[np.diag_indices_from(sigma_inv)] *= weights
        return sigma_inv
        
    def symmetrize(self, A):
        return (A + A.T)/2
    
    def nearestPD(self, A):
        """Find the nearest positive-definite matrix to input
        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].
        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """
        B = self.symmetrize(A)
        if self.isPD(B):
            return B

        _, s, V = svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        if self.isPD(A3):
            return A3
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self.isPD(A3):
            mineig = np.min(np.real(eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
        return A3    
        
    def isPD(self, A):
        try:
            _ = cholesky(A, lower=True)
            return True
        except LinAlgError:
            return False            

    ####################################################################################################################
    # LOAD/SAVE
    ####################################################################################################################

        
    def get_model_parameters(self):
        parameters = {}
        parameters['Xm'] = self.Xm
        parameters['K_inv'] = self.K_inv
        parameters['Kmm_inv'] = self.Kmm_inv
        parameters['alpha'] = self.alpha
        parameters['single_atom_energies'] = self.single_atom_energies
        parameters['theta'] = self.kernel.theta
        return parameters

    def set_model_parameters(self, parameters):
        self.Xm = parameters['Xm']
        self.K_inv = parameters['K_inv']
        self.Kmm_inv = parameters['Kmm_inv']
        self.alpha = parameters['alpha']
        self.single_atom_energies = parameters['single_atom_energies']
        self.kernel.theta = parameters['theta']
        self.ready_state = True

    
