import torch
import torch.nn as nn


class Projector(nn.Module):
    """
    Visual projector: vision encoder space → LLM hidden space.

    Architecture mirrors LLaVA-1.5's multi_modal_projector exactly:
        Linear(in_dim → out_dim) → GELU → Linear(out_dim → out_dim)

    Using the same 2-layer architecture allows loading pretrained weights
    directly from llava-hf/llava-1.5-7b-hf, skipping Stage 1 cold-start.
    LLaVA-1.5's projector was trained on 558k align samples + 665k instruct
    samples — much better than anything a short Stage 1 can produce.

    Input : (B, N, in_dim)
    Output: (B, N, out_dim)
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 512,  # kept for API compat, unused in proj
        out_dim: int = 512,
        text_dim: int = None,
        budget_hidden_dim: int = 128,
    ):
        super().__init__()

        # Matches LLaVA-1.5 multi_modal_projector: linear_1 + act + linear_2
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

        # Predict per-sample keep ratio from visual features and, when available,
        # the question representation so the budget can react to the query.
        self.budget_visual_proj = nn.Linear(out_dim, budget_hidden_dim)
        self.budget_text_proj = (
            nn.Linear(text_dim, budget_hidden_dim)
            if text_dim is not None
            else None
        )
        budget_in_dim = budget_hidden_dim * (2 if self.budget_text_proj is not None else 1)
        self.budget_head = nn.Sequential(
            nn.Linear(budget_in_dim, budget_hidden_dim),
            nn.GELU(),
            nn.Linear(budget_hidden_dim, budget_hidden_dim),
            nn.GELU(),
            nn.Linear(budget_hidden_dim, 1),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.proj(v)

    def load_from_llava(self, llava_multi_modal_projector: nn.Module):
        """
        Load pretrained weights from LLaVA-1.5's multi_modal_projector.
        LLaVA-1.5 uses: linear_1 (in→out, with bias) + act + linear_2 (out→out, with bias).
        Our proj[0] = Linear(in→out), proj[2] = Linear(out→out).
        """
        src = llava_multi_modal_projector
        # linear_1 → proj[0], linear_2 → proj[2]
        src_l1 = getattr(src, "linear_1", None) or getattr(src, "0", None)
        src_l2 = getattr(src, "linear_2", None) or getattr(src, "2", None)
        if src_l1 is None or src_l2 is None:
            # fallback: iterate children
            children = list(src.children())
            linears = [c for c in children if isinstance(c, nn.Linear)]
            if len(linears) >= 2:
                src_l1, src_l2 = linears[0], linears[1]
        if src_l1 is not None:
            self.proj[0].weight.data.copy_(src_l1.weight.data)
            if src_l1.bias is not None and self.proj[0].bias is not None:
                self.proj[0].bias.data.copy_(src_l1.bias.data)
        if src_l2 is not None:
            self.proj[2].weight.data.copy_(src_l2.weight.data)
            if src_l2.bias is not None and self.proj[2].bias is not None:
                self.proj[2].bias.data.copy_(src_l2.bias.data)
        print("[Projector] Loaded pretrained weights from LLaVA-1.5 multi_modal_projector")

    def predict_keep_ratio(
        self,
        projected_tokens: torch.Tensor,
        min_keep_ratio: float,
        max_keep_ratio: float,
        text_tokens: torch.Tensor = None,
        text_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        projected_tokens: [B, N, D]
        return keep ratio per sample: [B], bounded in [min_keep_ratio, max_keep_ratio]
        """
        pooled_visual = projected_tokens.mean(dim=1)
        budget_features = [self.budget_visual_proj(pooled_visual)]

        if self.budget_text_proj is not None and text_tokens is not None:
            if text_attention_mask is not None:
                text_mask = text_attention_mask.unsqueeze(-1).to(text_tokens.dtype)
                pooled_text = (text_tokens * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1e-8)
            else:
                pooled_text = text_tokens.mean(dim=1)
            budget_features.append(self.budget_text_proj(pooled_text))

        fused_budget_features = torch.cat(budget_features, dim=-1)
        raw = self.budget_head(fused_budget_features).squeeze(-1)
        gate = torch.sigmoid(raw)

        min_r = float(min_keep_ratio)
        max_r = float(max_keep_ratio)
        if not (0.0 < min_r <= max_r <= 1.0):
            raise ValueError(
                f"Expected 0 < min_keep_ratio <= max_keep_ratio <= 1, got "
                f"{min_r}, {max_r}"
            )

        return min_r + (max_r - min_r) * gate