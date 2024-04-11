import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from scipy.linalg import svd
from timeit import default_timer as dt
from agox.writer import agox_writer

from agox.models.ABC_model import ModelBaseClass

from agox.models.gaussian_process.kernels import clone
from agox.observer import Observer

class ModelGPR(ModelBaseClass):
    """
    Model calculator that trains on structures form ASLA memory.
    """
    implemented_properties = ['energy', 'forces', 'uncertainty', 'force_uncertainty']

    name = 'ModelGPR'

    dynamic_attributes = ['model']
    
    def __init__(self, model=None, max_training_data=10e8, iteration_start_training=7,
                 update_interval=1, max_energy=1000, max_adapt_iters=0, n_adapt=25, 
                 force_kappa=0, extrapolate=False, optimize_loglikelyhood=True, use_saved_features=False,
                 sparsifier=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.iteration_start_training = iteration_start_training
        self.update_interval = update_interval
        self.max_energy = max_energy
        self.max_training_data = max_training_data
        self.optimize_loglikelyhood = optimize_loglikelyhood
        self.max_adapt_iters = max_adapt_iters
        self.n_adapt = n_adapt
        self.use_saved_features = use_saved_features

        self.force_kappa = force_kappa
        self.extrapolate = extrapolate
        self.sparsifier = sparsifier

        # Important for use with DeltaModel
        self.part_of_delta_model = False

        self.model.verbose = self.verbose # This may cause somewhat unexpected print out behaviour, but deal with it.
               
    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        E, E_err = self.predict_energy(self.atoms, return_uncertainty=True)
        self.results['energy'] = E
        self.results['uncertainty'] = E_err
        
        if 'forces' in properties:
            forces,f_err = self.predict_forces(self.atoms, return_uncertainty=True)
            self.results['forces'] = forces
            self.results['force_uncertainty'] = f_err

    def predict_energy(self, atoms, feature=None, return_uncertainty=False, **kwargs):
        if return_uncertainty:
            E, E_err,_ = self.model.predict_energy(atoms, fnew=feature, return_error=True)
            return E, E_err
        else:
            E = self.model.predict_energy(atoms, fnew=feature, return_error=False)
            return E

    def predict_forces(self, atoms, feature=None, return_uncertainty=False, **kwargs):
        if return_uncertainty:
            F, F_err = self.model.predict_force(atoms, fnew=feature, return_error=True)
            return F.reshape(len(atoms), 3), F_err.reshape(len(atoms), 3)
        else:
            F = self.model.predict_force(atoms, fnew=feature).reshape(len(atoms), 3)
            return F
        
    def batch_predict(self, data, over_write=False):        
        E = np.array([self.model.predict_energy(atoms, return_error=False) for atoms in data])
        if over_write:
            self.batch_assign(data, E)

        return E

    ####################################################################################################################
    # Training:
    ####################################################################################################################

    def pretrain(self, structures):
        energies = [atoms.get_potential_energy() for atoms in structures]
        self.model.train(structures)
        self.pretrain_structures = structures

        self.writer('Model pretrained on {} structures'.format(len(structures)))
        self.writer('Lowest energy: {} eV'.format(np.min(energies)))
        self.writer('Median energy {} eV'.format(np.mean(energies)))
        self.writer('Max energy {} eV'.format(np.max(energies)))
        self.ready_state = True

    def train_model(self, all_data, energies):        
        # Train on the best structures:
        args = np.argsort(energies)
        E_best = energies[args[0]]        
        allowed_idx = [arg for arg in args if energies[arg] - E_best < self.max_energy]
        # always keep at least 10 best structures irrespective of their energies
        if len(allowed_idx) < 10:
            allowed_idx = args[:10]

        training_data = [all_data[i] for i in allowed_idx]
        training_energies = [energies[i] for i in allowed_idx]
        remaining = [arg for arg in args if arg not in allowed_idx]
        
        # calculate features
        if self.use_saved_features:
            features = []
            deltas = []
            for candidate in training_data:
                F = candidate.get_meta_information('GPR_feature')
                d = candidate.get_meta_information('GPR_delta')
                if F is None:
                    F = self.model.descriptor.get_global_features(candidate)[0]
                    candidate.add_meta_information('GPR_feature', F)
                    d = self.model.delta_function.energy(candidate)
                    candidate.add_meta_information('GPR_delta', d)
                features.append(F)
                deltas.append(d)
            features = np.array(features)
            deltas = np.array(deltas)
        else:
            features = np.array(self.model.descriptor.get_global_features(training_data))
            deltas = np.array([self.model.delta_function.energy(cand) for cand in training_data]) #np.array([c.get_meta_information('GPR_delta') for c in training_data])

        
        if self.max_training_data is not None and self.max_training_data < len(training_data):
            self.writer('Performing structure sparsification using CUR')
            self.writer(f'Training structures before sparsification: {len(training_data)}')            
            U, _, _ = svd(features)
            score = np.sum(U[:,:self.max_training_data]**2, axis=1)/self.max_training_data
            sorter = np.argsort(score)[::-1]
            training_data = [training_data[i] for i in sorter[:self.max_training_data]]
            features = features[sorter, :][:self.max_training_data, :]
            training_energies = [d.get_potential_energy() for d in training_data]
            self.writer(f'Training structures after sparsification: {len(training_data)}')
            

        self.writer(f'Min GPR training energy: {np.min(training_energies)}')
        self.writer(f'Max GPR training energy: {np.max(training_energies)}')
        
        # Wipe any stored atoms object, since now the model will change and hence give another result
        self.atoms = None

        self.model.train(features=features, data_values=training_energies,
                         delta_values=deltas, add_new_data=False, optimize=self.optimize_loglikelyhood) 


        self.ready_state = True
        if self.max_adapt_iters > 0 and len(remaining) > 0:
            self.writer('Running adaptive training on remaining {}'.format(len(remaining)))
            self.adaptive_training(all_data, energies, remaining)

            
    def adaptive_training(self, all_data, energies, remaining):
        run_adapt = True
        t0 = dt()
        adapt_iters = 0
        E_best = np.min(energies)
        while run_adapt and self.n_adapt > 0:

            if len(remaining) > 0:

                # Cases:
                # 1: Predict outside, True outside
                # 2: Predict inside,  True outside --> Worst case
                # 3: Predict outside, True inside  --> Model predicts a too high energy. 
                # 4: Predict inside,  True inside

                # 1: cond2 = False, cond1 = false
                # 2: cond2 = True,  cond1 = False
                # 3: cond2 = False, cond1 = True
                # 4: cond2 = True,  cond2 = True

                # Get prediction energies:
                true_energies = np.array([energies[idx] for idx in remaining])
                predicted_energies = np.array([self.model.predict_energy(all_data[i], return_error=False) for i in remaining])
                error = true_energies - predicted_energies
             
                # Conditions:
                cond1 = (true_energies - E_best < self.max_energy) # True if true_energy within max_energy of best, False otherwise
                cond2 = (predicted_energies - E_best < self.max_energy) # True if predicted energy within max_energy of best, False otherwise
                cond3 = error > self.max_energy / 4 # True if difference larger than max_energy / 4
                total_cond = (cond1 + cond2) * cond3

                non_zero_prob_idx = np.argwhere(total_cond == True)

                if len(non_zero_prob_idx) == 0:
                    break

                # Uniform probability for all non-discarded structures:
                probs = np.zeros(len(remaining))
                probs[non_zero_prob_idx] = 1
                probs = probs / np.sum(probs)
                
                # Pick the new training data:
                strucs_left = len(np.argwhere(probs != 0))
                if self.n_adapt <= strucs_left:
                    choices = np.random.choice(remaining, size=self.n_adapt, p=probs, replace=False)
                else:
                    choices = np.random.choice(remaining, size=strucs_left, p=probs, replace=False)
                    
                # Train on it
                adapt_batch = [all_data[c] for c in choices]
                data_values = [energies[c] for c in choices]
                self.model.train(adapt_batch, data_values=data_values, add_new_data=True)

                # Remove choices from remaining:
                remaining = [i for i in remaining if i not in choices]

                # When to stop:
                if (total_cond == False).all():
                    break
                elif adapt_iters >= self.max_adapt_iters:
                    break
                
                adapt_iters += 1 

            else:
                break

        t1 = dt()
        print('Model trained on {} structures'.format(len(all_data)-len(remaining)))
        print('Model training took {:4.2f} s'.format(t1-t0))

    @agox_writer
    @Observer.observer_method
    def training_observer(self, database, state):

        iteration = state.get_iteration_counter()
        if iteration < self.iteration_start_training:
            self.writer('Not yet training GPR')
            return
        if (iteration % self.update_interval != 0) * (iteration != self.iteration_start_training):
            self.writer('Iteration matching update_interval')
            return 

        self.writer('Training GPR')

        # Find complete builds:
        all_data = database.get_all_candidates()
        if self.sparsifier != None:
            _, all_data = self.sparsifier(all_data)
        energies = np.array([x.get_potential_energy() for x in all_data])

        self.writer('Training GPR on {} structures'.format(len(all_data)))
        self.train_model(all_data, energies)

        self.writer('GPR model k1:',self.model.kernel_.get_params().get('k1','NA'))
        self.writer('GPR model k2:',self.model.kernel_.get_params().get('k2','NA'))

    ####################################################################################################################
    # For MPI Relax. 
    ####################################################################################################################

    def set_verbosity(self, verbose):
        self.verbose = verbose
        self.model.verbose = verbose

    def get_model_parameters(self):
        DeprecationWarning("'get_model_parameters'-method will be removed in a future release. Use the save/load functions instead.")
        parameters = {}
        parameters['feature_mat'] = self.model.featureMat
        parameters['alpha'] = self.model.alpha
        parameters['bias'] = self.model.bias
        parameters['kernel_hyperparameters'] = self.model.kernel_.get_params()
        parameters['K_inv'] = self.model.K_inv
        parameters['iteration'] = self.get_iteration_counter()
        return parameters
    
    def set_model_parameters(self, parameters):
        DeprecationWarning("'set_model_parameters'-method will be removed in a future release. Use the save/load functions instead.")

        self.model.featureMat = parameters['feature_mat']
        self.model.alpha = parameters['alpha']
        self.model.bias = parameters['bias']
        self.model.K_inv = parameters['K_inv']
        self.set_iteration_counter(parameters['iteration'])

        try:
            self.model.kernel_ = clone(self.model.kernel_)
        except:
            self.model.kernel_ = clone(self.model.kernel)

        self.model.kernel_.set_params(**parameters['kernel_hyperparameters'])
        self.ready_state = True

    ####################################################################################################################
    # Misc. 
    ####################################################################################################################

    @classmethod
    def default(cls, environment=None, database=None, temp_atoms=None, lambda1min=1e-1, lambda1max=1e3, lambda2min=1e-1, lambda2max=1e3, 
                theta0min=1, theta0max=1e5, beta=0.01, use_delta_func=True, sigma_noise = 1e-2,
                descriptor=None, kernel=None, max_iterations=None, max_training_data=1000):

        """
        Creates a GPR model. 

        Parameters
        ------------
        environment: AGOX environment. 
            Used to create an atoms object to initialize e.g. the feature calculator. 
        database: 
            AGOX database that the model will be attached to. 
        lambda1min/lambda1max: float
            Length scale minimum and maximum. 
        lambda2min/lambda2max: float
            Length scale minimum and maximum. 
        theta0min/theta0max: float
            Amplitude minimum and maximum 
        use_delta_func: bool
            Whether to use the repulsive prior function. 
        sigma_noise: float
            Noise amplitude. 
        feature_calculator: object
            A feature calculator object, if None defaults to reasonable 
            fingerprint settings. 
        kernel: str or kernel object or None. 
            If kernel='anisotropic' the anisotropic RBF kernel is used where
            radial and angular componenets are treated at different length scales
            If None the standard RBF is used. 
            If a kernel object then that kernel is used. 
        max_iterations: int or None
            Maximum number of iterations for the hyperparameter optimization during 
            its BFGS optimization through scipy. 
        """
        from ase import Atoms
        from agox.models.gaussian_process.delta_functions_multi.delta import delta as deltaFunc
        from agox.models.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, GeneralAnisotropicRBF
        from agox.models.gaussian_process.GPR import GPR

        assert temp_atoms is not None or environment is not None

        if temp_atoms is None:
            temp_atoms = environment.get_template()
            temp_atoms += Atoms(environment.get_numbers())

        if descriptor is None:
            from agox.models.descriptors import Fingerprint
            descriptor = Fingerprint(temp_atoms)

        lambda1ini = (lambda1max - lambda1min)/2 + lambda1min
        lambda2ini = (lambda2max - lambda2min)/2 + lambda2min
        theta0ini = 5000                         
        
        if kernel is None:
            kernel = C(theta0ini, (theta0min, theta0max)) * \
            ( \
            C((1-beta), ((1-beta), (1-beta))) * RBF(lambda1ini, (lambda1min,lambda1max)) + \
            C(beta, (beta, beta)) * RBF(lambda2ini, (lambda2min,lambda2max)) 
            ) + \
            WhiteKernel(sigma_noise, (sigma_noise,sigma_noise))
        if kernel == 'anisotropic':
            
            lambda1ini_aniso = np.array([lambda1ini, lambda1ini])
            lambda2ini_aniso = np.array([lambda2ini, lambda2ini])

            length_scale_indices = np.array([0 for _ in range(descriptor.cython_module.Nelements_2body)] 
                + [1 for _ in range(descriptor.cython_module.Nelements_3body)])

            kernel = C(theta0ini, (theta0min, theta0max)) * \
            ( \
            C((1-beta), ((1-beta), (1-beta))) * GeneralAnisotropicRBF(lambda1ini_aniso, (lambda1min,lambda1max), length_scale_indices=length_scale_indices) + \
            C(beta, (beta, beta)) * GeneralAnisotropicRBF(lambda2ini_aniso, (lambda2min,lambda2max), length_scale_indices=length_scale_indices) 
            ) + \
            WhiteKernel(sigma_noise, (sigma_noise,sigma_noise))
            

        if use_delta_func:
            delta = deltaFunc(atoms=temp_atoms, rcut=6)
        else:
            delta = None
        
        gpr = GPR(kernel=kernel,
                descriptor=descriptor,
                delta_function=delta,
                bias_func=None,
                optimize=True,
                n_restarts_optimizer=1,
                n_maxiter_optimizer = max_iterations,
                use_delta_in_training=False)

        return cls(gpr, database=database, update_interval=1, optimize_loglikelyhood=True, use_saved_features=True, max_training_data=max_training_data)

    def get_feature_calculator(self):
        print(DeprecationWarning("The 'get_feature_calculator'-method will be deprecated in a future release, please use 'get_descriptor'-method instead."))
        return self.model.descriptor

    def get_descriptor(self):
        return self.model.descriptor
