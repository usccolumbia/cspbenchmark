from agox.models.local_GPR.LSGPR_CUR import LSGPRModelCUR
from agox.models.descriptors.soap import SOAP
from agox.models.priors.repulsive import Repulsive
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from agox.models.datasets import datasets

data = datasets['Ag5O3']
training_data = data[:80]

descriptor = SOAP(['O', 'Ag'], r_cut=5., nmax=3, lmax=2, sigma=1., weight=True, periodic=True)
kernel = C(1)*RBF(length_scale=20)
model = LSGPRModelCUR(kernel=kernel, descriptor=descriptor, noise=0.01, prior=Repulsive(ratio=0.7),
                      m_points=1000, verbose=True)

model.train_model(training_data)

model.save('my-model')
