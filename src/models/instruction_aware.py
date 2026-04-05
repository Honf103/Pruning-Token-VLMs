import torch
import torch.nn as nn
import torch.nn.functional as F


class InstructionAwareScorer(nn.Module):
    """
    Compute instruction-aware visual token scores using multi-head cross-attention.

    Pipeline:
        text_tokens  → Q  (after linear + LayerNorm)
        visual_tokens → K, V  (after linear + LayerNorm)
        A = MHA(Q, K, V)          [B, L, N]
        S_ia = beta^T A           [B, N]

    num_heads=4 captures richer text-visual alignment patterns than a single
    dot-product head without a large parameter overhead.

    Shapes
    -------
    text_tokens   : [B, L, D_text]
    visual_tokens : [B, N, D_vis]
    beta          : [B, L]

    Outputs
    -------
    A    : [B, L, N]  – averaged attention map
    S_ia : [B, N]
    """

    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()

        # ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            hidden_dim = (hidden_dim // num_heads) * num_heads

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vis_proj = nn.Linear(vision_dim, hidden_dim)

        self.norm_text = nn.LayerNorm(hidden_dim)
        self.norm_vis = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        beta: torch.Tensor,
        text_attention_mask: torch.Tensor = None,
        return_logits: bool = False,
    ):
        """
        text_tokens        : [B, L, D_text]
        visual_tokens      : [B, N, D_vis]
        beta               : [B, L]
        text_attention_mask: [B, L]  1=valid, 0=pad
        """
        B, L, _ = text_tokens.shape
        N = visual_tokens.size(1)

        q = self.norm_text(self.text_proj(text_tokens))    # [B, L, H]
        k = self.norm_vis(self.vis_proj(visual_tokens))    # [B, N, H]

        # Manual multi-head cross-attention (only attention weights needed)
        q_h = q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, L, d]
        k_h = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N, d]

        scale = (self.head_dim ** -0.5) / max(float(self.temperature), 1e-6)
        attn_logits = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale  # [B, h, L, N]
        A = F.softmax(attn_logits, dim=-1).mean(dim=1)  # [B, L, N]

        # Aggregate attention over text positions.
        # When beta is provided use it as text-token weights; otherwise fall back
        # to a simple masked mean (avoids double-weighting on already-normalized
        # attention and removes the TextImportanceMLP dependency).
        if beta is not None:
            if text_attention_mask is not None:
                A = A * text_attention_mask.unsqueeze(-1).to(A.dtype)
                beta = beta * text_attention_mask.to(beta.dtype)
            beta = beta / (beta.sum(dim=-1, keepdim=True) + 1e-8)
            s_ia = torch.einsum("bl,bln->bn", beta, A)   # [B, N]
        else:
            if text_attention_mask is not None:
                mask = text_attention_mask.unsqueeze(-1).to(A.dtype)    # [B, L, 1]
                s_ia = (A * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-8)  # [B, N]
            else:
                s_ia = A.mean(dim=1)   # [B, N]

        if return_logits:
            return s_ia, A, A

        return s_ia, A