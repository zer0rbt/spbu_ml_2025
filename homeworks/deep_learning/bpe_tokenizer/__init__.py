"""
BPE Tokenizer Package

A Byte Pair Encoding tokenizer implementation with pre-tokenization and special tokens support.
"""

from .tokenizer import BPETokenizer
from .utils.metrics import compute_metrics
from .data.datasets import JokesDataset, GPT2Dataset, collate_fn, gpt2_collate_fn
from .models.rnn import RNNLanguageModel
from .analysis import (
    compute_detailed_metrics,
    get_unused_tokens,
    build_compression_curve,
    plot_compression_curve,
    compare_domains,
    print_domain_comparison
)

__all__ = [
    'BPETokenizer',
    'compute_metrics',
    'compute_detailed_metrics',
    'get_unused_tokens',
    'build_compression_curve',
    'plot_compression_curve',
    'compare_domains',
    'print_domain_comparison',
    'JokesDataset',
    'GPT2Dataset',
    'collate_fn',
    'gpt2_collate_fn',
    'RNNLanguageModel',
]

__version__ = '1.0.0'

