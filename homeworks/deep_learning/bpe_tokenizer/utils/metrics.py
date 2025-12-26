"""
Metrics computation for tokenizer evaluation.
"""

import re
from collections import defaultdict, Counter
from typing import List, Dict, TYPE_CHECKING
from tqdm import tqdm

if TYPE_CHECKING:
    from ..tokenizer import BPETokenizer


def compute_metrics(tokenizer: 'BPETokenizer', texts: List[str]) -> Dict:
    """
    Compute tokenization metrics.
    
    Metrics computed:
    - Compression ratio (tokens / bytes)
    - Compression ratio (tokens / chars)
    - Average tokens per word
    - Average tokens per word for top 10% frequent words
    
    Args:
        tokenizer: BPETokenizer instance
        texts: List of texts to analyze
    
    Returns:
        Dictionary with computed metrics
    """
    total_tokens = 0
    total_bytes = 0
    total_chars = 0
    
    word_token_counts = defaultdict(list)  # word -> list of token counts
    word_frequencies = Counter()
    
    word_pattern = re.compile(r'[а-яА-ЯёЁa-zA-Z]+')
    
    for text in tqdm(texts, desc="Computing metrics"):
        # Overall metrics
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
        total_bytes += len(text.encode('utf-8'))
        total_chars += len(text)
        
        # Per-word metrics
        words = word_pattern.findall(text)
        for word in words:
            word_lower = word.lower()
            word_frequencies[word_lower] += 1
            word_tokens = tokenizer.encode(word)
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
        'compression_ratio_bytes': total_tokens / total_bytes if total_bytes > 0 else 0,
        'compression_ratio_chars': total_tokens / total_chars if total_chars > 0 else 0,
        'avg_tokens_per_word': avg_tokens_per_word,
        'avg_tokens_per_top_10_word': avg_tokens_per_top_word,
        'total_tokens': total_tokens,
        'total_bytes': total_bytes,
        'total_chars': total_chars,
        'unique_words': len(word_token_counts),
        'vocab_size': len(tokenizer)
    }

