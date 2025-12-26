"""
RNN Language Model Implementation

A simple RNN-based language model for text generation.
"""

import torch
import torch.nn as nn
from typing import Optional


class RNNLanguageModel(nn.Module):
    """Simple RNN-based language model."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = "LSTM"
    ):
        """
        Initialize RNN language model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of RNN hidden state
            num_layers: Number of RNN layers
            dropout: Dropout probability
            rnn_type: Type of RNN ('LSTM', 'GRU', or 'RNN')
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                embedding_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                embedding_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.RNN(
                embedding_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len)
            hidden: Optional hidden state
        
        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            hidden: Updated hidden state
        """
        embedded = self.dropout(self.embedding(x))  # (batch, seq_len, embed_dim)
        output, hidden = self.rnn(embedded, hidden)  # (batch, seq_len, hidden_dim)
        output = self.dropout(output)
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        return logits, hidden
    
    def generate(
        self, 
        tokenizer, 
        start_text: str = "", 
        max_length: int = 100, 
        temperature: float = 1.0
    ) -> str:
        """
        Generate text using the model.
        
        Args:
            tokenizer: BPETokenizer instance
            start_text: Optional starting text
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Generated text string
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Start with BOS token or encoded start_text
        if start_text:
            tokens = [tokenizer.vocab["<BOS>"]] + tokenizer.encode(start_text)
        else:
            tokens = [tokenizer.vocab["<BOS>"]]
        
        generated = tokens.copy()
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                x = torch.tensor([tokens[-1:]]).to(device)
                logits, hidden = self.forward(x, hidden)
                
                # Apply temperature
                probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                tokens = [next_token]
                
                if next_token == tokenizer.vocab.get("<EOS>", -1):
                    break
        
        return tokenizer.decode(generated)

