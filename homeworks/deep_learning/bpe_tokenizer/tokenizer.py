"""
BPE Tokenizer Implementation

Byte Pair Encoding tokenizer with pre-tokenization and special tokens support.
Pre-tokenization pattern based on GPT-4's cl100k_base pattern.

Source: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
"""

import re
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer with pre-tokenization and special tokens.
    
    Pre-tokenization pattern based on GPT-4's cl100k_base pattern.
    Source: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    """
    
    # Simple pattern that works well with regex module for Russian + English
    SIMPLE_PATTERN = re.compile(
        r"""[а-яА-ЯёЁa-zA-Z]+|"""  # Words (Russian and English)
        r"""[0-9]+|"""              # Numbers
        r"""[^\s]+|"""              # Non-whitespace sequences
        r"""\s+""",                 # Whitespace
        re.UNICODE
    )
    
    def __init__(
        self,
        vocab_size: int = 1000,
        special_tokens: Optional[List[str]] = None,
        min_frequency: int = 2
    ):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            special_tokens: List of special tokens (e.g., <PAD>, <UNK>, <BOS>, <EOS>)
            min_frequency: Minimum frequency for a pair to be merged
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        
        # Vocabulary: token -> id
        self.vocab: Dict[str, int] = {}
        # Inverse vocabulary: id -> token
        self.inverse_vocab: Dict[int, str] = {}
        # Merge rules: (token1, token2) -> merged_token
        self.merges: List[Tuple[str, str]] = []
        
        self._initialize_vocab()
    
    def _initialize_vocab(self):
        """Initialize vocabulary with special tokens and byte-level tokens."""
        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
        
        # Add all possible bytes (0-255) as base vocabulary
        offset = len(self.special_tokens)
        for byte_val in range(256):
            token = self._byte_to_token(byte_val)
            self.vocab[token] = offset + byte_val
            self.inverse_vocab[offset + byte_val] = token
    
    def _byte_to_token(self, byte_val: int) -> str:
        """Convert byte value to token representation."""
        # Use printable characters directly, encode others
        if 32 <= byte_val < 127 and byte_val != ord(' '):
            return chr(byte_val)
        return f"<0x{byte_val:02X}>"
    
    def _token_to_byte(self, token: str) -> Optional[int]:
        """Convert token back to byte value."""
        if token.startswith("<0x") and token.endswith(">"):
            return int(token[3:-1], 16)
        elif len(token) == 1 and 32 <= ord(token) < 127:
            return ord(token)
        return None
    
    def pretokenize(self, text: str) -> List[str]:
        """Split text into pre-tokens using regex pattern."""
        return self.SIMPLE_PATTERN.findall(text)
    
    def _text_to_tokens(self, text: str) -> List[str]:
        """Convert text to initial byte-level tokens."""
        tokens = []
        for byte_val in text.encode('utf-8'):
            tokens.append(self._byte_to_token(byte_val))
        return tokens
    
    def _get_pair_frequencies(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """Count frequency of adjacent token pairs."""
        pair_freqs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def _merge_pair(self, word_freqs: Dict[Tuple[str, ...], int], 
                    pair: Tuple[str, str]) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of a pair in the vocabulary."""
        new_word_freqs = {}
        merged = pair[0] + pair[1]
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def train(self, texts: List[str], verbose: bool = True):
        """
        Train BPE tokenizer on a corpus of texts.
        
        Args:
            texts: List of training texts
            verbose: Whether to show progress bar
        """
        # Step 1: Pre-tokenize and count word frequencies
        word_freqs: Dict[Tuple[str, ...], int] = defaultdict(int)
        
        for text in tqdm(texts, desc="Pre-tokenizing", disable=not verbose):
            pre_tokens = self.pretokenize(text)
            for pre_token in pre_tokens:
                tokens = tuple(self._text_to_tokens(pre_token))
                word_freqs[tokens] += 1
        
        # Step 2: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.vocab)
        
        iterator = tqdm(range(num_merges), desc="Learning merges", disable=not verbose)
        for _ in iterator:
            pair_freqs = self._get_pair_frequencies(word_freqs)
            
            if not pair_freqs:
                break
            
            best_pair = pair_freqs.most_common(1)[0]
            if best_pair[1] < self.min_frequency:
                break
            
            pair = best_pair[0]
            merged = pair[0] + pair[1]
            
            # Add to merges and vocabulary
            self.merges.append(pair)
            token_id = len(self.vocab)
            self.vocab[merged] = token_id
            self.inverse_vocab[token_id] = merged
            
            # Apply merge to word frequencies
            word_freqs = self._merge_pair(word_freqs, pair)
        
        if verbose:
            print(f"Trained tokenizer with {len(self.vocab)} tokens and {len(self.merges)} merges")
    
    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """Apply learned merges to a list of tokens."""
        for pair in self.merges:
            merged = pair[0] + pair[1]
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens = tokens[:i] + [merged] + tokens[i + 2:]
                else:
                    i += 1
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns:
            List of token IDs
        """
        ids = []
        
        if add_special_tokens and "<BOS>" in self.vocab:
            ids.append(self.vocab["<BOS>"])
        
        # Pre-tokenize
        pre_tokens = self.pretokenize(text)
        
        for pre_token in pre_tokens:
            # Convert to byte-level tokens
            tokens = self._text_to_tokens(pre_token)
            # Apply merges
            tokens = self._apply_merges(tokens)
            # Convert to IDs
            for token in tokens:
                if token in self.vocab:
                    ids.append(self.vocab[token])
                else:
                    ids.append(self.vocab.get("<UNK>", 0))
        
        if add_special_tokens and "<EOS>" in self.vocab:
            ids.append(self.vocab["<EOS>"])
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text
        """
        tokens = [self.inverse_vocab.get(id_, "<UNK>") for id_ in ids]
        
        # Remove special tokens
        tokens = [t for t in tokens if t not in self.special_tokens]
        
        # Convert tokens back to bytes
        byte_list = []
        for token in tokens:
            self._token_to_bytes(token, byte_list)
        
        # Decode bytes to string
        try:
            return bytes(byte_list).decode('utf-8', errors='replace')
        except:
            return ''.join(tokens)
    
    def _token_to_bytes(self, token: str, byte_list: list):
        """Convert a token to bytes and append to byte_list."""
        i = 0
        while i < len(token):
            if token[i:].startswith("<0x") and len(token) >= i + 6 and token[i + 5] == ">":
                # Byte token like <0x00>
                byte_val = int(token[i + 3:i + 5], 16)
                byte_list.append(byte_val)
                i += 6
            elif 32 <= ord(token[i]) < 127:
                byte_list.append(ord(token[i]))
                i += 1
            else:
                # Fallback: encode character as UTF-8
                byte_list.extend(token[i].encode('utf-8'))
                i += 1
    
    def get_tokens(self, text: str) -> List[str]:
        """Get token strings for text (for analysis)."""
        tokens = []
        pre_tokens = self.pretokenize(text)
        
        for pre_token in pre_tokens:
            byte_tokens = self._text_to_tokens(pre_token)
            merged_tokens = self._apply_merges(byte_tokens)
            tokens.extend(merged_tokens)
        
        return tokens
    
    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'min_frequency': self.min_frequency,
            'vocab': self.vocab,
            'merges': self.merges
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(
            vocab_size=data['vocab_size'],
            special_tokens=data['special_tokens'],
            min_frequency=data['min_frequency']
        )
        tokenizer.vocab = data['vocab']
        tokenizer.inverse_vocab = {v: k for k, v in data['vocab'].items()}
        tokenizer.merges = data['merges']
        
        return tokenizer
    
    def __len__(self):
        return len(self.vocab)

