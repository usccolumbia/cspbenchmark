from agox.models.local_GPR.LSGPR_MBKMeans import LSGPRModelMBKMeans


class LSGPRModelMBKMeansNoise(LSGPRModelMBKMeans):
    def __init__(self, noise_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_model = noise_model

    def predict_uncertainty(self, atoms=None, **kwargs):
        _, unc = self.noise_model.predict_energy(atoms, return_uncertainty=True)
        return unc

    def get_model_parameters(self):
        parameters = super().get_model_parameters()
        parameters['noise_model_parameters'] = self.noise_model.get_model_parameters()
        return parameters


    def set_model_parameters(self, parameters):
        self.noise_model.set_model_parameters(parameters['noise_model_parameters'])
        super().set_model_parameters(parameters)

