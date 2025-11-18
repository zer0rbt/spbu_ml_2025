from .base_layer import BaseLayer
import numpy as np
class Linear(BaseLayer):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros(out_features) if bias else None

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None

        self.x = None                                          # кэш

    def forward(self, x):
        self.x = x
        out = x @ self.W
        if self.use_bias:
            out = out + self.b
        return out

    def backward(self, dout):
        self.dW = self.x.T @ dout
        if self.use_bias:
            self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

    def parameters(self):
        params = [(self.W, self.dW)]
        if self.use_bias:
            params.append((self.b, self.db))
        return params