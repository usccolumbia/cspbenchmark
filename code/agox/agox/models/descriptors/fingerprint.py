from agox.models.descriptors import DescriptorBaseClass
from agox.models.descriptors.fingerprint_cython.angular_fingerprintFeature_cy import Angular_Fingerprint

class Fingerprint(DescriptorBaseClass):

    feature_types = ['global', 'global_gradient']

    name = 'Fingerprint'

    def __init__(self, init_atoms, rc1=6, rc2=4, binwidth=0.2, Nbins=30, sigma1=0.2, sigma2=0.2, gamma=2, 
        eta=20, use_angular=True, **kwargs):
        super().__init__(**kwargs)
        self.cython_module = Angular_Fingerprint(init_atoms, Rc1=rc1, Rc2=rc2, 
            binwidth1=binwidth, Nbins2=Nbins, sigma1=sigma1, sigma2=sigma2, 
            gamma=gamma, eta=eta, use_angular=use_angular)

    def create_global_features(self, atoms):
        return self.cython_module.get_feature(atoms)

    def create_global_feature_gradient(self, atoms):
        return self.cython_module.get_featureGradient(atoms)

        
