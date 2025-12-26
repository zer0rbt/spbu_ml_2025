"""
Dataset implementations for language modeling.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..tokenizer import BPETokenizer


class JokesDataset(Dataset):
    """Dataset for language modeling with custom BPE tokenizer."""
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer: 'BPETokenizer', 
        max_length: int = 128
    ):
        """
        Initialize JokesDataset.
        
        Args:
            texts: List of text samples
            tokenizer: BPETokenizer instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) > 2:  # Need at least BOS + some tokens + EOS
                # Truncate if needed
                if len(tokens) > max_length:
                    tokens = tokens[:max_length-1] + [tokenizer.vocab["<EOS>"]]
                self.data.append(tokens)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        # Input: all tokens except last, Target: all tokens except first
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])


class GPT2Dataset(Dataset):
    """Dataset for GPT-2 training with HuggingFace tokenizer."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        """
        Initialize GPT2Dataset.
        
        Args:
            texts: List of text samples
            tokenizer: HuggingFace tokenizer instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            if encoding["input_ids"].shape[1] > 1:
                self.data.append(encoding["input_ids"].squeeze(0))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    Collate function for padding sequences.
    
    Args:
        batch: List of (input, target) tuples
    
    Returns:
        Tuple of padded inputs and targets
    """
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded


def gpt2_collate_fn(batch):
    """
    Collate function for GPT-2 training.
    
    Args:
        batch: List of input_ids tensors
    
    Returns:
        Dictionary with input_ids, attention_mask, and labels
    """
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    attention_mask = (padded != 0).long()
    return {"input_ids": padded, "attention_mask": attention_mask, "labels": padded}

