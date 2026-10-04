import numpy as np
from cobaya.likelihood import Likelihood

class iam_rsd(Likelihood):
    def initialize(self):
        self.data = np.array([
            [0.067, 0.423, 0.055],
            [0.150, 0.530, 0.160],
            [0.380, 0.497, 0.045],
            [0.510, 0.459, 0.038],
            [0.700, 0.473, 0.041],
            [0.850, 0.315, 0.095],
            [1.480, 0.462, 0.045],
        ])
        self.z_data = self.data[:, 0]
        self.fsig8_data = self.data[:, 1]
        self.fsig8_err = self.data[:, 2]

    def get_requirements(self):
        return {"fsigma8": {"z": self.z_data.tolist()}}

    def logp(self, **params_values):
        fsig8_theory = self.provider.get_fsigma8(self.z_data)
        chi2 = np.sum(((fsig8_theory - self.fsig8_data) / self.fsig8_err) ** 2)
        return -0.5 * chi2
