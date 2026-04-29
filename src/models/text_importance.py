import torch
import torch.nn as nn
import torch.nn.functional as F


class TextImportanceMLP(nn.Module):
    """
    beta = softmax(MLP_text(T))
    T: [B, L, D]
    beta: [B, L]
    """
    def __init__(self, dim=256, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, text_tokens: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        text_tokens: [B, L, D]
        attention_mask: [B, L] with 1 for valid, 0 for pad
        return beta: [B, L]
        """
        logits = self.mlp(text_tokens).squeeze(-1)  # [B, L]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

        if attention_mask is not None:
            # Use large negative values instead of -inf to avoid all--inf rows
            # producing NaN after softmax.
            logits = logits.masked_fill(attention_mask == 0, -1e4)

        beta = F.softmax(logits, dim=-1)
        beta = torch.nan_to_num(beta, nan=0.0, posinf=1.0, neginf=0.0)

        if attention_mask is not None:
            # If a row has no valid text tokens, fall back to token-0 weight = 1.
            valid_counts = attention_mask.sum(dim=-1)  # [B]
            no_valid = valid_counts == 0
            if no_valid.any():
                beta = beta.clone()
                beta[no_valid] = 0.0
                beta[no_valid, 0] = 1.0

        return beta
