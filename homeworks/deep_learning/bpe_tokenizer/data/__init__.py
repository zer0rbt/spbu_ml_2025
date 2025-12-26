"""
Data module for BPE Tokenizer package.
"""

from .datasets import JokesDataset, GPT2Dataset, collate_fn, gpt2_collate_fn

__all__ = ['JokesDataset', 'GPT2Dataset', 'collate_fn', 'gpt2_collate_fn']

