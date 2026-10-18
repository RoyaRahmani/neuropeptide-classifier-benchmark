"""Dataset loaders and batch collation routines for peptide sequences."""

import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiModalPeptideDataset(Dataset):
    """Pairs residue token embeddings with dense tabular biophysical descriptors."""

    def __init__(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor,
        handcrafted_df: pd.DataFrame,
        labels: torch.Tensor,
    ):
        self.tokens = tokens.float()
        self.masks = masks.bool()

        # Isolate numerical features, dropping non-feature metadata if present
        meta_cols = ["id", "sequence", "label"]
        feature_cols = [
            c for c in handcrafted_df.columns if c not in meta_cols
        ]
        self.handcrafted = torch.tensor(
            handcrafted_df[feature_cols].values, dtype=torch.float32
        )
        self.labels = labels.float().unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "tokens": self.tokens[idx],
            "mask": self.masks[idx],
            "handcrafted": self.handcrafted[idx],
            "label": self.labels[idx],
        }


def collate_multimodal_fn(batch: list) -> dict:
    """Pads variable-length residue token sequences dynamically to the batch maximum."""
    batch_tokens = [item["tokens"] for item in batch]
    batch_masks = [item["mask"] for item in batch]
    batch_handcrafted = torch.stack([item["handcrafted"] for item in batch])
    batch_labels = torch.stack([item["label"] for item in batch])

    max_len = max(t.size(0) for t in batch_tokens)
    hidden_dim = batch_tokens[0].size(1)

    padded_tokens = []
    padded_masks = []

    for t, m in zip(batch_tokens, batch_masks):
        pad_len = max_len - t.size(0)
        if pad_len > 0:
            pad_t = torch.zeros(
                (pad_len, hidden_dim), dtype=t.dtype, device=t.device
            )
            pad_m = torch.zeros(pad_len, dtype=m.dtype, device=m.device)
            padded_tokens.append(torch.cat([t, pad_t], dim=0))
            padded_masks.append(torch.cat([m, pad_m], dim=0))
        else:
            padded_tokens.append(t)
            padded_masks.append(m)

    return {
        "tokens": torch.stack(padded_tokens),
        "mask": torch.stack(padded_masks),
        "handcrafted": batch_handcrafted,
        "label": batch_labels,
    }
