import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenPruner(nn.Module):
    """
    Keep top-k visual tokens according to fused score S.

    sorted_output=True: after top-k selection, tokens are re-ordered by their
    original spatial index, preserving the 2-D grid layout that helps the LLM.

    use_merging=True: instead of discarding dropped tokens, each one is merged
    (aggregated by cosine-similarity nearest-neighbour) into its closest kept
    token.  This is a pure inference-time trick — no retraining required.
    """
    def __init__(
        self,
        keep_ratio: float = 0.25,
        sorted_output: bool = True,
        use_merging: bool = False,
    ):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.sorted_output = sorted_output
        self.use_merging = use_merging

    # ------------------------------------------------------------------
    # Token-merging helper
    # ------------------------------------------------------------------
    def _merge_into_kept(
        self,
        kept_tokens: torch.Tensor,    # [B, K, D]
        dropped_tokens: torch.Tensor, # [B, M, D]
        kept_scores: torch.Tensor,    # [B, K]   fused score of kept tokens
        dropped_scores: torch.Tensor, # [B, M]   fused score of dropped tokens
    ) -> torch.Tensor:
        """
        For each dropped token find its nearest kept token (cosine similarity)
        and accumulate it there via score-weighted pooling.

        Merge formula:
            t̃_j = (s_j·t_j + Σ_{i→j} s_i·t_i) / (s_j + Σ_{i→j} s_i)

        Using fused scores as weights ensures high-score dropped tokens
        contribute more than near-zero-score ones.

        Returns merged kept tokens, same shape [B, K, D].
        """
        B, K, D = kept_tokens.shape
        M = dropped_tokens.size(1)
        if M == 0:
            return kept_tokens

        orig_dtype = kept_tokens.dtype

        # Cosine similarity [B, M, K].
        # Detach here: assign comes from argmax (no gradient), so building a
        # backward graph through normalize+bmm wastes ~88 MB of activation
        # memory per batch with D=4096 for zero gradient gain.
        k_norm = F.normalize(kept_tokens.detach().float(), dim=-1)    # [B, K, D]
        d_norm = F.normalize(dropped_tokens.detach().float(), dim=-1) # [B, M, D]
        sim = torch.bmm(d_norm, k_norm.transpose(1, 2))               # [B, M, K]

        # Each dropped token → index of nearest kept token
        assign = sim.argmax(dim=-1)  # [B, M]

        # Score-weighted scatter-accumulate
        # weight each dropped token by its fused score (clamped to ≥ 0)
        d_scores = dropped_scores.float().clamp(min=0.0)   # [B, M]
        k_scores = kept_scores.float().clamp(min=0.0)      # [B, K]

        expand_assign = assign.unsqueeze(-1).expand(B, M, D)
        weighted_dropped = dropped_tokens.float() * d_scores.unsqueeze(-1)  # [B, M, D]

        accum       = torch.zeros(B, K, D, device=kept_tokens.device, dtype=torch.float32)
        score_accum = torch.zeros(B, K,    device=kept_tokens.device, dtype=torch.float32)

        accum.scatter_add_(1, expand_assign, weighted_dropped)
        score_accum.scatter_add_(1, assign, d_scores)

        # Score-weighted merge: (s_j·t_j + Σ s_i·t_i) / (s_j + Σ s_i)
        denom  = (k_scores + score_accum).clamp(min=1e-6).unsqueeze(-1)  # [B, K, 1]
        merged = (kept_tokens.float() * k_scores.unsqueeze(-1) + accum) / denom

        return merged.to(orig_dtype)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        scores: torch.Tensor,
        keep_ratio: float = None,
    ):
        """
        visual_tokens : [B, N, D]
        scores        : [B, N]
        keep_ratio    : override self.keep_ratio at call time (curriculum)
                        - float in (0,1]
                        - Tensor [B] with per-sample keep ratios
        returns:
            kept_tokens  : [B, K, D]
            kept_indices : [B, K]   – in spatial order when sorted_output=True
            kept_scores  : [B, K]
        """
        B, N, D = visual_tokens.shape
        ratio = keep_ratio if keep_ratio is not None else self.keep_ratio

        if isinstance(ratio, torch.Tensor):
            if ratio.dim() != 1 or ratio.size(0) != B:
                raise ValueError(
                    f"Per-sample keep_ratio must have shape [B], got {tuple(ratio.shape)}"
                )

            ratio = ratio.to(device=scores.device, dtype=scores.dtype).clamp(1e-4, 1.0)
            k_each = torch.clamp((ratio * N).round().long(), min=1, max=N)
            k_max = int(k_each.max().item())

            topk_scores, topk_idx = torch.topk(scores, k=k_max, dim=-1)
            rank = torch.arange(k_max, device=scores.device).unsqueeze(0).expand(B, -1)
            valid_mask = rank < k_each.unsqueeze(1)

            # Move invalid positions to tail while preserving spatial order for valid ones.
            if self.sorted_output:
                invalid_fill = N + rank
                sortable = torch.where(valid_mask, topk_idx, invalid_fill)
                sort_order = sortable.argsort(dim=-1)
                topk_idx = topk_idx.gather(1, sort_order)
                topk_scores = topk_scores.gather(1, sort_order)
                valid_mask = valid_mask.gather(1, sort_order)

            safe_idx = torch.where(valid_mask, topk_idx, torch.zeros_like(topk_idx))
            gather_idx = safe_idx.unsqueeze(-1).expand(B, k_max, D)
            kept_tokens = torch.gather(visual_tokens, dim=1, index=gather_idx)
            kept_tokens = kept_tokens * valid_mask.unsqueeze(-1).to(kept_tokens.dtype)
            kept_scores = torch.where(valid_mask, topk_scores, torch.zeros_like(topk_scores))
            kept_indices = torch.where(valid_mask, topk_idx, torch.full_like(topk_idx, -1))
            return kept_tokens, kept_indices, kept_scores

        K = max(1, int(N * float(ratio)))
        kept_scores, kept_indices = torch.topk(scores, k=K, dim=-1)  # [B, K]

        if self.sorted_output:
            # restore original spatial order so the LLM sees a consistent layout
            sort_order = kept_indices.argsort(dim=-1)              # [B, K]
            kept_indices = kept_indices.gather(1, sort_order)      # [B, K]
            kept_scores = kept_scores.gather(1, sort_order)        # [B, K]

        gather_idx = kept_indices.unsqueeze(-1).expand(B, K, D)
        kept_tokens = torch.gather(visual_tokens, dim=1, index=gather_idx)

        # Token Merging: instead of discarding bottom-K tokens, merge their
        # features into their nearest kept token (cosine similarity).
        if self.use_merging and K < N:
            # Mark which positions are kept → derive dropped positions
            kept_flag = torch.zeros(B, N, dtype=torch.bool, device=scores.device)
            kept_flag.scatter_(1, kept_indices, True)              # [B, N]
            M_drop = N - K
            # argsort: False (0) < True (1), so dropped positions come first
            sort_for_drop = kept_flag.long().argsort(dim=1)        # [B, N]
            dropped_idx = sort_for_drop[:, :M_drop]                # [B, M]
            dropped_gather = dropped_idx.unsqueeze(-1).expand(B, M_drop, D)
            dropped_tokens = torch.gather(visual_tokens, 1, dropped_gather)
            dropped_scores_gather = dropped_idx
            dropped_sc = scores.gather(1, dropped_scores_gather)   # [B, M]
            kept_tokens = self._merge_into_kept(
                kept_tokens, dropped_tokens, kept_scores, dropped_sc
            )

        return kept_tokens, kept_indices, kept_scores
