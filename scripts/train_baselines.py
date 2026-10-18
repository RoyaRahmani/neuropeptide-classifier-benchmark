"""Model architectures for Neuropeptide Classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Calculates position-specific attention weights and pools token vectors."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class PurePLMNet(nn.Module):
    """Model B: BiLSTM + Attention Pooling on ESM-2 650M representations."""

    def __init__(
        self, esm_dim: int = 1280, lstm_hidden: int = 128, dropout: float = 0.3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=esm_dim,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True,
        )
        self.pool = AttentionPooling(hidden_dim=lstm_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        lstm_out, _ = self.lstm(tokens)
        pooled, attn_weights = self.pool(lstm_out, mask)
        logits = self.classifier(pooled)
        return logits, attn_weights
