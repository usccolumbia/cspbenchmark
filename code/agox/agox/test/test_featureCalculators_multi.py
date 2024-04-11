import unittest

class TestStringMethods(unittest.TestCase):

    def test_feature(self):
        import numpy as np
        from ase import Atoms
        from agox.models.gaussian_process.featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint

        L = 2
        d = 1
        dim = 3
        pbc = [1,0,0]
        x = np.array([0.2*L, 0.7*L, d/2,
                      0.3*L, 0.2*L, d/2,
                      0.7*L, 0.9*L, d/2,
                      0.7*L, 0.5*L, d/2,
                      0.9*L, 0.1*L, d/2,
                      1.9*L, 1.1*L, d/2])
        positions = x.reshape((-1,dim))
        atomtypes = ['H','C','O','C','C','Cu']
        a = Atoms(atomtypes,
                  positions=positions,
                  cell=[L,L,d],
                  pbc=pbc)

        Rc1 = 4
        binwidth1 = 0.1
        sigma1 = 0.2

        Rc2 = 3
        Nbins2 = 50
        sigma2 = 0.2

        eta = 30
        gamma = 2
        use_angular = True


        featureCalculator = Angular_Fingerprint(a, Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2,
                                                eta=eta, gamma=gamma, use_angular=use_angular)
        
        f = featureCalculator.get_feature(a)
        self.assertEqual(len(f), 2400)
        self.assertTrue(np.amax(f)-0.23507344658046264 < 1e-8)


if __name__ == '__main__':
    unittest.main()


