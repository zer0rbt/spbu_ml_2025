from .base_layer import BaseLayer
import numpy as np

class BatchNorm(BaseLayer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        self.gamma = np.ones(num_features)
        self.beta  = np.zeros(num_features)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta  = np.zeros_like(self.beta)

        self.running_mean = np.zeros(num_features)
        self.running_var  = np.ones(num_features)

        # кэш
        self.x = self.mean = self.var = self.x_norm = self.std_inv = None

    def forward(self, x):
        self.x = x
        if self.training:
            self.mean = x.mean(axis=0)
            self.var  = x.var(axis=0)

            self.std_inv = 1.0 / np.sqrt(self.var + self.eps)
            self.x_norm = (x - self.mean) * self.std_inv

            out = self.gamma * self.x_norm + self.beta

            # обновляем running stats
            m = self.momentum
            self.running_mean = m * self.mean + (1 - m) * self.running_mean
            self.running_var  = m * self.var  + (1 - m) * self.running_var
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        return out

    def backward(self, dout):
        N = self.x.shape[0]
        x_centered = self.x - self.mean

        self.dgamma = np.sum(dout * self.x_norm, axis=0)
        self.dbeta  = np.sum(dout, axis=0)

        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * x_centered * (-0.5) * self.std_inv**3, axis=0)
        dmean = np.sum(dx_norm * (-self.std_inv), axis=0) + dvar * np.mean(-2*x_centered, axis=0)

        dx = dx_norm * self.std_inv
        dx += 2.0 * x_centered * dvar / N
        dx += dmean / N
        return dx

    def parameters(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]