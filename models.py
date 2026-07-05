"""
Definicije PyTorch modela:
  - MLP (Viseslojna neuronska mreza)
  - TextCNN (1D konvolutivna mreza za tekst)
"""

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Viseslojna neuronska mreza za klasifikaciju teksta.
    Ulaz: TF-IDF vektor (10 000 dimenzija).
    Izlaz: 2 klase (covjek / AI).
    """
    def __init__(self, input_size: int, hidden1: int = 512, hidden2: int = 128,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextCNN(nn.Module):
    """
    1D konvolutivna mreza za klasifikaciju teksta.
    Analogna CNN za slike (vjezba 8), ali sa 1D konvolucijama
    nad sekvencama rijeci umjesto 2D konvolucija nad pikselima.

    Ulaz: sekvenca indeksa rijeci (batch, seq_len).
    Izlaz: 2 klase (covjek / AI).
    """
    def __init__(self, vocab_size: int, embed_dim: int,
                 embedding_matrix: np.ndarray,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        # Embedding sloj inicijalizovan pretreniranim Word2Vec tezinama
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.FloatTensor(embedding_matrix), requires_grad=True
        )

        # Ekstrakcija obelezja — 1D konvolucije
        self.feature_extraction = nn.Sequential(
            # ulaz: (batch, embed_dim, seq_len)
            nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),      # seq_len -> seq_len/2
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Global max pooling — uzima najvazniji signal iz cijele sekvence
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)            # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)          # (batch, embed_dim, seq_len)
        x = self.feature_extraction(x)  # (batch, 64, seq_len/2)
        x = self.global_pool(x)         # (batch, 64, 1)
        x = x.squeeze(-1)               # (batch, 64)
        x = self.dropout(x)
        return self.classifier(x)       # (batch, 2)
