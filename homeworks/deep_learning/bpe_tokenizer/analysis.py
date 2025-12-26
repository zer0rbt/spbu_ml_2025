"""
Analysis utilities for tokenizer evaluation.

Functions for computing metrics, analyzing domains, and comparing tokenizers.
"""

import re
from collections import Counter
from typing import List, Dict, Set, TYPE_CHECKING
import matplotlib.pyplot as plt
from tqdm import tqdm

if TYPE_CHECKING:
    from .tokenizer import BPETokenizer


def compute_detailed_metrics(tokenizer: 'BPETokenizer', texts: List[str], domain_name: str = "default") -> Dict:
    """
    Compute detailed tokenization metrics for a domain.
    
    Args:
        tokenizer: BPETokenizer instance
        texts: List of texts to analyze
        domain_name: Name of the domain for reporting
    
    Returns:
        Dictionary with computed metrics
    """
    total_tokens = 0
    total_bytes = 0
    total_chars = 0
    
    word_token_counts = {}  # word -> list of token counts
    word_frequencies = Counter()
    used_token_ids: Set[int] = set()
    
    word_pattern = re.compile(r'[а-яА-ЯёЁa-zA-Z]+')
    
    for text in tqdm(texts, desc=f"Analyzing {domain_name}", leave=False):
        # Overall metrics
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
        total_bytes += len(text.encode('utf-8'))
        total_chars += len(text)
        used_token_ids.update(tokens)
        
        # Per-word metrics
        words = word_pattern.findall(text)
        for word in words:
            word_lower = word.lower()
            word_frequencies[word_lower] += 1
            word_tokens = tokenizer.encode(word)
            if word_lower not in word_token_counts:
                word_token_counts[word_lower] = []
            word_token_counts[word_lower].append(len(word_tokens))
    
    # Calculate average tokens per word
    total_word_tokens = sum(sum(counts) for counts in word_token_counts.values())
    total_words = sum(len(counts) for counts in word_token_counts.values())
    avg_tokens_per_word = total_word_tokens / total_words if total_words > 0 else 0
    
    # Top 10% frequent words
    sorted_words = word_frequencies.most_common()
    top_10_percent_count = max(1, len(sorted_words) // 10)
    top_words = set(word for word, _ in sorted_words[:top_10_percent_count])
    
    top_word_tokens = sum(
        sum(word_token_counts[word]) 
        for word in top_words if word in word_token_counts
    )
    top_word_count = sum(
        len(word_token_counts[word]) 
        for word in top_words if word in word_token_counts
    )
    avg_tokens_per_top_word = top_word_tokens / top_word_count if top_word_count > 0 else 0
    
    return {
        'domain': domain_name,
        'compression_ratio_bytes': total_tokens / total_bytes if total_bytes > 0 else 0,
        'compression_ratio_chars': total_tokens / total_chars if total_chars > 0 else 0,
        'avg_tokens_per_word': avg_tokens_per_word,
        'avg_tokens_per_top_10_word': avg_tokens_per_top_word,
        'total_tokens': total_tokens,
        'total_bytes': total_bytes,
        'total_chars': total_chars,
        'unique_words': len(word_token_counts),
        'vocab_size': len(tokenizer),
        'used_tokens': len(used_token_ids),
        'unused_tokens': len(tokenizer) - len(used_token_ids),
        'used_token_ids': used_token_ids
    }


def get_unused_tokens(tokenizer: 'BPETokenizer', texts: List[str]) -> Dict:
    """
    Find tokens that were never used in the given texts.
    
    Args:
        tokenizer: BPETokenizer instance
        texts: List of texts to analyze
    
    Returns:
        Dictionary with unused token information
    """
    used_token_ids: Set[int] = set()
    
    for text in tqdm(texts, desc="Finding unused tokens"):
        tokens = tokenizer.encode(text)
        used_token_ids.update(tokens)
    
    all_token_ids = set(tokenizer.vocab.values())
    unused_ids = all_token_ids - used_token_ids
    
    # Get unused token strings
    unused_tokens = []
    for token_id in unused_ids:
        token_str = tokenizer.inverse_vocab.get(token_id, f"<ID:{token_id}>")
        unused_tokens.append((token_id, token_str))
    
    return {
        'total_vocab': len(all_token_ids),
        'used_count': len(used_token_ids),
        'unused_count': len(unused_ids),
        'unused_ratio': len(unused_ids) / len(all_token_ids) if all_token_ids else 0,
        'unused_tokens': sorted(unused_tokens, key=lambda x: x[0])
    }


def build_compression_curve(
    texts: List[str],
    vocab_sizes: List[int],
    special_tokens: List[str] = None,
    min_frequency: int = 2
) -> Dict[int, float]:
    """
    Build vocabulary size vs compression ratio curve.
    
    Args:
        texts: Training texts
        vocab_sizes: List of vocabulary sizes to test
        special_tokens: Special tokens for tokenizer
        min_frequency: Minimum frequency for BPE merges
    
    Returns:
        Dictionary mapping vocab_size to compression_ratio
    """
    from .tokenizer import BPETokenizer
    
    results = {}
    
    for vocab_size in tqdm(vocab_sizes, desc="Building compression curve"):
        tokenizer = BPETokenizer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=min_frequency
        )
        tokenizer.train(texts, verbose=False)
        
        total_tokens = 0
        total_bytes = 0
        
        for text in texts:
            tokens = tokenizer.encode(text)
            total_tokens += len(tokens)
            total_bytes += len(text.encode('utf-8'))
        
        compression_ratio = total_tokens / total_bytes if total_bytes > 0 else 0
        results[vocab_size] = compression_ratio
    
    return results


def plot_compression_curve(results: Dict[int, float], save_path: str = None):
    """
    Plot vocabulary size vs compression ratio curve.
    
    Args:
        results: Dictionary mapping vocab_size to compression_ratio
        save_path: Optional path to save the figure
    """
    vocab_sizes = sorted(results.keys())
    compression_ratios = [results[v] for v in vocab_sizes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(vocab_sizes, compression_ratios, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Vocabulary Size', fontsize=12)
    plt.ylabel('Compression Ratio (tokens/bytes)', fontsize=12)
    plt.title('Vocabulary Size vs Compression Ratio', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def compare_domains(
    tokenizer: 'BPETokenizer',
    domain_texts: Dict[str, List[str]]
) -> Dict[str, Dict]:
    """
    Compare tokenization efficiency across different domains.
    
    Args:
        tokenizer: BPETokenizer instance
        domain_texts: Dictionary mapping domain names to text lists
    
    Returns:
        Dictionary with metrics for each domain
    """
    results = {}
    
    for domain_name, texts in domain_texts.items():
        metrics = compute_detailed_metrics(tokenizer, texts, domain_name)
        results[domain_name] = metrics
    
    return results


def print_domain_comparison(results: Dict[str, Dict]):
    """Print formatted domain comparison table."""
    print("\n" + "=" * 80)
    print("Domain Comparison Results")
    print("=" * 80)
    
    headers = ["Domain", "Comp. Ratio", "Tokens/Word", "Tokens/Top10%", "Used Tokens"]
    widths = [20, 15, 15, 15, 15]
    
    header_line = " | ".join(h.center(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))
    
    for domain, metrics in results.items():
        row = [
            domain[:18],
            f"{metrics['compression_ratio_bytes']:.4f}",
            f"{metrics['avg_tokens_per_word']:.2f}",
            f"{metrics['avg_tokens_per_top_10_word']:.2f}",
            f"{metrics['used_tokens']}"
        ]
        row_line = " | ".join(str(r).center(w) for r, w in zip(row, widths))
        print(row_line)
    
    print("=" * 80)

