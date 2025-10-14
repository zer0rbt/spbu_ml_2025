import torch
import numpy as np


def sigmoid(x):
    return 1 / (1 + torch.exp(-torch.clamp(x, min=-500, max=500)))


def nll_loss(predicted, labels):
    # L = -[y*log(p) + (1-y)*log(1-p)]
    epsilon = 1e-8
    predicted = torch.clamp(predicted, min=epsilon, max=1 - epsilon)
    loss = -torch.mean(labels * torch.log(predicted) + (1 - labels) * torch.log(1 - predicted))
    return loss


def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs, 
                 batch_size=None, use_sgd=False):
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    weights = torch.tensor(initial_weights, dtype=torch.float32, requires_grad=False)
    bias = torch.tensor(initial_bias, dtype=torch.float32, requires_grad=False)
    
    n_samples = features.shape[0]
    
    if use_sgd:
        batch_size = 1
    elif batch_size is None:
        batch_size = n_samples
    
    nll_values = []
    
    for epoch in range(epochs):
        if batch_size < n_samples:
            indices = torch.randperm(n_samples)
            features_shuffled = features[indices]
            labels_shuffled = labels[indices]
        else:
            features_shuffled = features
            labels_shuffled = labels
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_features = features_shuffled[i:i+batch_size]
            batch_labels = labels_shuffled[i:i+batch_size]
            
            # z = w^T * x + b
            z = torch.matmul(batch_features, weights) + bias
            # a = sigmoid(z)
            a = sigmoid(z)
            
            batch_loss = nll_loss(a, batch_labels)
            epoch_loss += batch_loss.item()
            n_batches += 1
            
            # dL/dz = a - y
            dL_dz = a - batch_labels
            
            # dL/dw = dL/dz * x
            # dL/db = dL/dz
            dL_dw = torch.mean(dL_dz.unsqueeze(1) * batch_features, dim=0)
            dL_db = torch.mean(dL_dz)
            
            # w = w - lr * dL/dw
            weights = weights - learning_rate * dL_dw
            bias = bias - learning_rate * dL_db
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else epoch_loss
        nll_values.append(round(avg_loss, 4))
    
    return weights.tolist(), bias.item(), nll_values


def train_neuron_with_optimizer(features, labels, initial_weights, initial_bias, optimizer, epochs, 
                                batch_size=None):
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    weights = optimizer.params[0]
    bias = optimizer.params[1]
    
    with torch.no_grad():
        weights.data.copy_(torch.tensor(initial_weights, dtype=torch.float32))
        bias.data.copy_(torch.tensor([initial_bias], dtype=torch.float32))
    
    n_samples = features.shape[0]
    
    if batch_size is None:
        batch_size = n_samples
    
    nll_values = []
    
    for epoch in range(epochs):
        if batch_size < n_samples:
            indices = torch.randperm(n_samples)
            features_shuffled = features[indices]
            labels_shuffled = labels[indices]
        else:
            features_shuffled = features
            labels_shuffled = labels
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_features = features_shuffled[i:i+batch_size]
            batch_labels = labels_shuffled[i:i+batch_size]
            
            if bias.dim() > 0:
                bias_val = bias.squeeze() if bias.numel() == 1 else bias
            else:
                bias_val = bias
            z = torch.matmul(batch_features, weights) + bias_val
            a = sigmoid(z)
            
            batch_loss = nll_loss(a, batch_labels)
            epoch_loss += batch_loss.item()
            n_batches += 1
            
            # dL/dz = a - y
            dL_dz = a - batch_labels
            
            dL_dw = torch.mean(dL_dz.unsqueeze(1) * batch_features, dim=0)
            dL_db = torch.mean(dL_dz)
            if bias.dim() > 0 and bias.shape[0] == 1:
                dL_db = dL_db.unsqueeze(0)
            
            optimizer.step([dL_dw, dL_db])
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else epoch_loss
        nll_values.append(round(avg_loss, 4))
    
    return weights.tolist(), bias.item(), nll_values
