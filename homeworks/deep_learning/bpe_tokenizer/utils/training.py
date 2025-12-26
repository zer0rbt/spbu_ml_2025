"""
Training utilities for language models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(inputs)
        
        # Reshape for loss calculation
        loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Count non-padding tokens
        mask = targets != 0
        total_loss += (loss.item() * mask.sum().item())
        total_tokens += mask.sum().item()
    
    return total_loss / total_tokens if total_tokens > 0 else 0


def evaluate(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> float:
    """
    Evaluate model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: Evaluation data loader
        criterion: Loss function
        device: Device to use
    
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            
            loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
            
            mask = targets != 0
            total_loss += (loss.item() * mask.sum().item())
            total_tokens += mask.sum().item()
    
    return total_loss / total_tokens if total_tokens > 0 else 0


def train_gpt2_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device
) -> float:
    """
    Train GPT-2 model for one epoch.
    
    Args:
        model: GPT-2 model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to use
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    total_steps = 0
    
    for batch in tqdm(dataloader, desc="Training GPT-2", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_steps += 1
    
    return total_loss / total_steps


def evaluate_gpt2(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device
) -> float:
    """
    Evaluate GPT-2 model on a dataset.
    
    Args:
        model: GPT-2 model
        dataloader: Evaluation data loader
        device: Device to use
    
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_steps += 1
    
    return total_loss / total_steps

