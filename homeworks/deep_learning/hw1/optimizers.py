"""
Assignment 3: AdamW Optimizer Implementation

General AdamW Optimizer Formulas:


    
Initialization:
    m₀ = 0, v₀ = 0, t = 0

At each step t:
    m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
    v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)
    θ_t = θ_{t-1} * (1 - lr * λ) - lr * m̂_t / (√v̂_t + ε)

Where:
    g_t - gradient at step t
    β₁, β₂ - exponential smoothing coefficients (usually 0.9, 0.999)
    lr - learning rate
    λ - weight decay coefficient
    ε - constant for numerical stability (usually 1e-8)
    θ - model parameters
"""
import torch


class AdamW:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0
        
        for i, param in enumerate(self.params):
            self.m[i] = torch.zeros_like(param)
            self.v[i] = torch.zeros_like(param)
    
    def step(self, grads):
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if grad.shape != param.shape:
                if grad.numel() == 1 and param.numel() == 1:
                    grad = grad.view_as(param)
                elif grad.dim() == 0 and param.numel() == 1:
                    grad = grad.unsqueeze(0)
                else:
                    grad = grad.view_as(param)
            
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            
            # m_hat = m_t / (1 - beta1^t)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # v_hat = v_t / (1 - beta2^t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # theta = theta * (1 - lr * lambda) - lr * m_hat / (sqrt(v_hat) + eps)
            update = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            new_data = param.data * (1 - self.lr * self.weight_decay) - update
            param.data.copy_(new_data)
    
    def zero_grad(self):
        pass
