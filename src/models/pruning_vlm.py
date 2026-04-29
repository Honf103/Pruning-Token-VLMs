import torch
import torch.nn as nn
import torch.nn.functional as F
from .cls_scorer import CLSScorer
from .text_importance import TextImportanceMLP
from .instruction_aware import InstructionAwareScorer
from .score_fusion import ScoreFusion
from .token_pruner import TokenPruner
from .projector import Projector

try:
    from .hf_backbones import CLIPVisionEncoder, CLIPTextEncoder, LLaVALM
except Exception as e:
    raise ImportError(
        "HF backbones (CLIP + LLaVA-style wrapper) are required. "
        "Install transformers and provide hf_backbones.py"
    ) from e


class PruningVLM(nn.Module):
    """
    Full instruction-aware token pruning VLM.

    Train:
        staged pruning over projected tokens z:
            - soft gating for early dense gradient flow
            - STE gating for sharper selection while keeping differentiability
            - structural top-k pruning near the end to match deployment

    Inference:
        hard top-k pruning on z
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        llm_model_name: str = "lmsys/vicuna-7b-v1.5",
        keep_ratio: float = 0.25,
        alpha: float = 0.5,
        learnable_alpha: bool = False,
        remove_cls_token: bool = False,
        llm_torch_dtype=None,
        ia_hidden_dim: int = 512,
        projector_hidden_dim: int = 2048,
        gate_temperature: float = 1.0,
        gate_score_scale: float = 5.0,
        use_ste: bool = False,
        gate_threshold_mode: str = "topk",   # "topk" | "zero" | "mean"
        llm_use_grad: bool = False,
        llm_scale_visual_prefix: bool = True,
        llm_visual_prefix_scale: float = 1.0,
        dynamic_budget_enabled: bool = False,
        dynamic_budget_min_keep_ratio: float = 0.35,
        dynamic_budget_max_keep_ratio: float = 0.85,
        use_merging: bool = False,
        question_conditioned_alpha: bool = False,
        text_dim: int = 768,
        attn_distill_layers: list = None,
    ):
        super().__init__()

        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

        if gate_threshold_mode not in {"topk", "zero", "mean"}:
            raise ValueError(
                f"gate_threshold_mode must be one of ['topk', 'zero', 'mean'], "
                f"got {gate_threshold_mode}"
            )

        self.keep_ratio = float(keep_ratio)
        self.gate_temperature = float(gate_temperature)
        self.gate_score_scale = float(gate_score_scale)
        self.use_ste = bool(use_ste)
        self.gate_threshold_mode = gate_threshold_mode
        self.dynamic_budget_enabled = bool(dynamic_budget_enabled)
        self.dynamic_budget_min_keep_ratio = float(dynamic_budget_min_keep_ratio)
        self.dynamic_budget_max_keep_ratio = float(dynamic_budget_max_keep_ratio)
        self.use_merging = bool(use_merging)
        self.question_conditioned_alpha = bool(question_conditioned_alpha)
        self.attn_distill_layers = attn_distill_layers or [8, 16, 23]

        # ---------------------------
        # Encoders
        # ---------------------------
        self.vision_encoder = CLIPVisionEncoder(
            model_name=clip_model_name,
            remove_cls_token=remove_cls_token,
        )

        self.text_encoder = CLIPTextEncoder(model_name=clip_model_name)

        vision_dim = self.vision_encoder.config.hidden_size
        clip_text_dim = self.text_encoder.config.hidden_size

        # ---------------------------
        # LLM
        # ---------------------------
        self.llm = LLaVALM(
            model_name=llm_model_name,
            torch_dtype=llm_torch_dtype,
            use_grad=llm_use_grad,
            scale_visual_prefix=llm_scale_visual_prefix,
            visual_prefix_scale=llm_visual_prefix_scale,
        )

        llm_dim = self.llm.hidden_size

        # ---------------------------
        # Projector
        # ---------------------------
        self.projector = Projector(
            in_dim=vision_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=llm_dim,
            text_dim=clip_text_dim,
        )

        # If LLaVA-1.5 was used as backbone, load its pretrained projector weights.
        # LLaVA's projector was trained on 558k+665k samples — much better than
        # cold-starting Stage 1 from random init.
        if hasattr(self.llm, "_llava_projector"):
            self.projector.load_from_llava(self.llm._llava_projector)
            del self.llm._llava_projector  # free memory

        # ---------------------------
        # Scorers
        # ---------------------------
        # Use the frozen CLIP projection dimension for scoring
        clip_joint_dim = self.vision_encoder.projection_dim
        self.cls_scorer = CLSScorer(vision_dim=clip_joint_dim, llm_dim=clip_joint_dim)
        self.text_importance = TextImportanceMLP(dim=clip_text_dim)

        self.instruction_aware = InstructionAwareScorer(
            text_dim=clip_text_dim,
            vision_dim=clip_joint_dim,
            hidden_dim=ia_hidden_dim,
        )

        self.score_fusion = ScoreFusion(
            alpha=alpha,
            learnable=learnable_alpha,
            question_conditioned=question_conditioned_alpha,
            text_dim=text_dim,
            hidden_dim=64,
        )

        # ---------------------------
        # Hard pruner for inference
        # ---------------------------
        self.token_pruner = TokenPruner(keep_ratio=self.keep_ratio, use_merging=self.use_merging)

    def set_gate_temperature(self, temperature: float):
        self.gate_temperature = float(temperature)

    def set_gate_score_scale(self, score_scale: float):
        self.gate_score_scale = float(score_scale)

    def set_keep_ratio(self, keep_ratio: float):
        """Update keep_ratio at runtime (e.g. for curriculum pruning schedule)."""
        self.keep_ratio = float(keep_ratio)
        self.token_pruner.keep_ratio = float(keep_ratio)

    def set_merging(self, enabled: bool):
        """Enable/disable token merging at inference time (no retraining needed)."""
        self.use_merging = bool(enabled)
        self.token_pruner.use_merging = bool(enabled)

    def set_dynamic_budget(self, enabled: bool):
        self.dynamic_budget_enabled = bool(enabled)

    def set_dynamic_budget_range(self, min_keep_ratio: float, max_keep_ratio: float):
        min_r = float(min_keep_ratio)
        max_r = float(max_keep_ratio)
        if not (0.0 < min_r <= max_r <= 1.0):
            raise ValueError(
                f"Expected 0 < min_keep_ratio <= max_keep_ratio <= 1, got {min_r}, {max_r}"
            )
        self.dynamic_budget_min_keep_ratio = min_r
        self.dynamic_budget_max_keep_ratio = max_r

    def get_dynamic_keep_ratio(
        self,
        projected_tokens: torch.Tensor,
        text_tokens: torch.Tensor = None,
        text_attention_mask: torch.Tensor = None,
    ):
        if not self.dynamic_budget_enabled:
            return None
        return self.projector.predict_keep_ratio(
            projected_tokens=projected_tokens,
            min_keep_ratio=self.dynamic_budget_min_keep_ratio,
            max_keep_ratio=self.dynamic_budget_max_keep_ratio,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
        )

    def _compute_tau(self, scores: torch.Tensor, keep_ratios: torch.Tensor = None) -> torch.Tensor:
        """
        scores: [B, N]
        return tau: [B, 1]
        """
        if self.gate_threshold_mode == "zero":
            return torch.zeros(
                scores.size(0), 1, device=scores.device, dtype=scores.dtype
            )

        if self.gate_threshold_mode == "mean":
            return scores.mean(dim=-1, keepdim=True)

        # default: top-k threshold
        bsz, num_tokens = scores.shape
        if keep_ratios is None:
            k = max(1, int(round(self.keep_ratio * num_tokens)))
            topk_vals, _ = torch.topk(scores, k=k, dim=-1)
            tau = topk_vals[:, -1:].detach()  # [B, 1]
            return tau

        k_each = torch.clamp(
            (keep_ratios.to(device=scores.device, dtype=scores.dtype) * num_tokens).round().long(),
            min=1,
            max=num_tokens,
        )
        topk_vals, _ = torch.topk(scores, k=int(k_each.max().item()), dim=-1)
        tau = topk_vals.gather(1, (k_each - 1).unsqueeze(1)).detach()
        return tau

    def _build_soft_gates(self, scores: torch.Tensor, keep_ratios: torch.Tensor = None):
        """
        Build differentiable soft gates centered around tau.
        This aligns soft gating with the intended pruning threshold.
        """
        tau = self._compute_tau(scores, keep_ratios=keep_ratios)
        temperature = max(float(self.gate_temperature), 1e-6)
        score_scale = max(float(self.gate_score_scale), 1e-6)

        scores_centered = scores - tau
        soft_gates = torch.sigmoid(score_scale * scores_centered / temperature)
        return soft_gates, tau

    def _build_hard_mask(self, scores: torch.Tensor, keep_ratios: torch.Tensor = None):
        """
        scores: [B, N]
        """
        bsz, num_tokens = scores.shape
        if keep_ratios is None:
            k = max(1, int(round(self.keep_ratio * num_tokens)))
            topk_scores, topk_idx = torch.topk(scores, k=k, dim=-1)
            hard_mask = torch.zeros_like(scores)
            hard_mask.scatter_(1, topk_idx, 1.0)
            return hard_mask, topk_idx, topk_scores

        keep_ratios = keep_ratios.to(device=scores.device, dtype=scores.dtype).clamp(1e-4, 1.0)
        k_each = torch.clamp((keep_ratios * num_tokens).round().long(), min=1, max=num_tokens)
        k_max = int(k_each.max().item())
        topk_scores, topk_idx = torch.topk(scores, k=k_max, dim=-1)

        rank = torch.arange(k_max, device=scores.device).unsqueeze(0).expand(bsz, -1)
        valid_mask = rank < k_each.unsqueeze(1)

        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, topk_idx, valid_mask.to(scores.dtype))

        topk_scores = torch.where(valid_mask, topk_scores, torch.zeros_like(topk_scores))
        topk_idx = torch.where(valid_mask, topk_idx, torch.full_like(topk_idx, -1))
        return hard_mask, topk_idx, topk_scores

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        llm_input_ids: torch.Tensor = None,
        llm_attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        return_intermediates: bool = False,
        use_hard_pruning: bool = None,
        train_pruning_mode: str = "structural",
        compute_align_loss: bool = False,
        align_layer: int = 2,
        compute_distill_loss: bool = False,
        distill_temperature: float = 2.0,
    ):
        if use_hard_pruning is None:
            use_hard_pruning = not self.training

        if train_pruning_mode not in {"soft", "ste", "structural"}:
            raise ValueError(
                "train_pruning_mode must be one of ['soft', 'ste', 'structural'], "
                f"got {train_pruning_mode}"
            )

        # --------------------------------------------------
        # 1) Encode
        # --------------------------------------------------
        cls_emb, visual_tokens = self.vision_encoder(images)  # cls_emb:[B,Dv]  patches:[B,576,Dv]

        text_tokens = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # [B, L, Dt]

        # --------------------------------------------------
        # 2) Build scoring features in frozen CLIP space, then project once
        #    to LLM space after token selection.
        # --------------------------------------------------
        cls_score_tokens = self.vision_encoder.project_features(cls_emb)
        visual_score_tokens = self.vision_encoder.project_features(visual_tokens)

        z = self.projector(visual_tokens)  # [B, N, D_llm]
        dynamic_keep_ratio = self.get_dynamic_keep_ratio(
            z,
            text_tokens=text_tokens,
            text_attention_mask=attention_mask,
        )

        # --------------------------------------------------
        # 3) Text token importance weights for S_ia aggregation
        # --------------------------------------------------
        beta = self.text_importance(text_tokens, attention_mask)  # [B, L]

        # --------------------------------------------------
        # 4) Visual saliency scoring: cosine(CLS_emb, patch_token_i)
        # --------------------------------------------------
        s_cls = self.cls_scorer(
            cls_embedding=cls_score_tokens,
            patch_tokens=visual_score_tokens,
        )  # [B, N]

        # --------------------------------------------------
        # 5) Instruction-aware scoring: parameter-free text→visual attention
        # --------------------------------------------------
        s_ia, A = self.instruction_aware(
            text_tokens=text_tokens,
            visual_tokens=visual_score_tokens,
            beta=beta,
            text_attention_mask=attention_mask,
        )  # [B, N]

        # --------------------------------------------------
        # 6) Score fusion: S = alpha * norm(S_ia) + (1-alpha) * norm(S_cls)
        # --------------------------------------------------
        # For question-conditioned alpha, pass text CLS token (index 0)
        text_cls = text_tokens[:, 0, :] if self.question_conditioned_alpha else None
        fused_scores, predicted_alpha = self.score_fusion(
            s_ia,
            s_cls,
            text_cls_token=text_cls,
        )  # [B, N], [B,1] or scalar
        fused_scores = torch.nan_to_num(fused_scores, nan=0.0, posinf=20.0, neginf=-20.0)
        fused_scores = fused_scores.clamp(-20.0, 20.0)

        # --------------------------------------------------
        # 7) Soft gate or hard pruning
        # --------------------------------------------------
        soft_gates, tau = self._build_soft_gates(fused_scores, keep_ratios=dynamic_keep_ratio)
        hard_mask, topk_idx, topk_scores = self._build_hard_mask(
            fused_scores,
            keep_ratios=dynamic_keep_ratio,
        )

        if use_hard_pruning:
            # inference: real structural pruning
            visual_for_llm, kept_indices, kept_scores = self.token_pruner(
                z,
                fused_scores,
                keep_ratio=dynamic_keep_ratio,
            )
            gate_mask_used = hard_mask
        else:
            if train_pruning_mode == "soft":
                # Early training: dense supervision over all tokens.
                visual_for_llm = z * soft_gates.unsqueeze(-1)
                kept_indices = topk_idx
                kept_scores = topk_scores
                gate_mask_used = soft_gates
            elif train_pruning_mode == "ste":
                # Mid training: same sequence length as soft mode, but hard-ish mask.
                gate_mask_used = hard_mask.detach() - soft_gates.detach() + soft_gates
                visual_for_llm = z * gate_mask_used.unsqueeze(-1)
                kept_indices = topk_idx
                kept_scores = topk_scores
            else:
                # Late training: structural pruning to match inference.
                visual_for_llm, kept_indices, kept_scores = self.token_pruner(
                    z,
                    fused_scores,
                    keep_ratio=dynamic_keep_ratio,
                )
                gate_mask_used = soft_gates  # used for budget_loss

        visual_for_llm = torch.nan_to_num(
            visual_for_llm,
            nan=0.0,
            posinf=100.0,
            neginf=-100.0,
        )

        # --------------------------------------------------
        # 8) LLM forward
        # --------------------------------------------------
        llm_input_ids = llm_input_ids if llm_input_ids is not None else input_ids
        llm_attention_mask = (
            llm_attention_mask if llm_attention_mask is not None else attention_mask
        )
        visual_attention_mask = None
        if kept_indices is not None and kept_indices.size(1) == visual_for_llm.size(1):
            visual_attention_mask = kept_indices.ge(0).to(
                device=llm_input_ids.device,
                dtype=llm_attention_mask.dtype if llm_attention_mask is not None else torch.long,
            )

        llm_outputs = self.llm(
            projected_visual_tokens=visual_for_llm,
            input_ids=llm_input_ids,
            attention_mask=llm_attention_mask,
            visual_attention_mask=visual_attention_mask,
            labels=labels,
        )

        if isinstance(llm_outputs, dict):
            logits = llm_outputs.get("logits", None)
            loss = llm_outputs.get("loss", None)
        else:
            logits = getattr(llm_outputs, "logits", llm_outputs)
            loss = getattr(llm_outputs, "loss", None)

        # --------------------------------------------------
        # 9) [Optional] Alignment + Distillation losses
        #    One teacher forward (no_grad, full N tokens) feeds both:
        #    • L_align  : KL(S_ia || LLM visual hidden-state importance)
        #                 Trains the scorer to predict which tokens the LLM
        #                 actually uses at an early layer.
        #    • L_distill: KL(student text logits || teacher text logits)
        #                 at answer positions, scaled by temperature².
        # --------------------------------------------------
        align_loss   = None
        distill_loss = None

        if (compute_align_loss or compute_distill_loss) and not use_hard_pruning:
            _llm_in  = llm_input_ids  if llm_input_ids  is not None else input_ids
            _llm_attn = llm_attention_mask if llm_attention_mask is not None else attention_mask
            N_vis = z.size(1)            # full visual tokens (no pruning)
            K_vis = visual_for_llm.size(1)  # student visual tokens (pruned / gated)

            with torch.no_grad():
                teacher_out = self.llm(
                    projected_visual_tokens=z,   # all N tokens, no pruning
                    input_ids=_llm_in,
                    attention_mask=_llm_attn,
                    visual_attention_mask=None,
                    labels=None,
                    output_hidden_states=False,
                    output_attentions=compute_align_loss,
                )

            # ── Distillation: KL(student || teacher) at answer positions ──────
            if compute_distill_loss and labels is not None and logits is not None:
                t_logits = (
                    teacher_out.get("logits") if isinstance(teacher_out, dict)
                    else getattr(teacher_out, "logits", None)
                )
                if t_logits is not None:
                    # Text segment starts after visual prefix
                    s_text = logits[:, K_vis:, :]    # [B, L, vocab]
                    t_text = t_logits[:, N_vis:, :]  # [B, L, vocab]
                    T = distill_temperature
                    log_p_s = F.log_softmax(s_text / T, dim=-1)
                    p_t     = F.softmax(t_text.detach() / T, dim=-1)
                    kl = F.kl_div(log_p_s, p_t, reduction="none").sum(-1)  # [B, L]
                    # Mask to answer-token positions only
                    ans_mask = (labels != -100).float()
                    L_min = min(kl.size(1), ans_mask.size(1))
                    kl = kl[:, :L_min]
                    ans_mask = ans_mask[:, :L_min]
                    n_ans = ans_mask.sum().clamp_min(1.0)
                    distill_loss = (kl * ans_mask).sum() / n_ans * (T ** 2)

            # ── Align: KL(S_ia || LLM cross-attention to visual tokens) ────────
            # Uses actual LLM attention weights (text->visual) as teacher signal.
            # This is strictly better than L2-norm of hidden states because it
            # directly measures which visual tokens the LLM attends to.
            if compute_align_loss and s_ia is not None:
                t_attentions = (
                    teacher_out.get("attentions") if isinstance(teacher_out, dict)
                    else getattr(teacher_out, "attentions", None)
                )
                if t_attentions is not None and len(t_attentions) > 0:
                    # Collect attention to visual tokens from target layers
                    # visual tokens are the first N_vis tokens in the LLM sequence
                    layer_importances = []
                    for layer_idx in self.attn_distill_layers:
                        if layer_idx >= len(t_attentions):
                            continue
                        attn = t_attentions[layer_idx]  # [B, num_heads, seq_len, seq_len]
                        # text tokens attend to visual prefix: attn[:, :, N_vis:, :N_vis]
                        seq_len = attn.size(2)
                        if seq_len > N_vis:
                            # [B, num_heads, L_text, N_vis] -> mean over heads and text positions
                            attn_to_vis = attn[:, :, N_vis:, :N_vis]        # [B, h, L, N]
                            importance = attn_to_vis.mean(dim=(1, 2))        # [B, N]
                            layer_importances.append(importance)

                    if layer_importances:
                        # Average importance across selected layers
                        teacher_importance = torch.stack(layer_importances, dim=0).mean(0)  # [B, N]
                        # Clamp to avoid degenerate softmax
                        teacher_importance = teacher_importance.clamp(min=0.0)

                        align_loss = F.kl_div(
                            F.log_softmax(s_ia, dim=-1),
                            F.softmax(teacher_importance.detach() * 10.0, dim=-1),
                            reduction="batchmean",
                        )

        out = {
            "logits": logits,
            "loss": loss,
            "fused_scores": fused_scores,
            "predicted_alpha": predicted_alpha,
            "s_cls": s_cls,
            "s_ia": s_ia,
            "beta": beta,
            "A": A,
            "kept_indices": kept_indices,
            "kept_scores": kept_scores,
            "projected_tokens": z,
            "visual_for_llm": visual_for_llm,
            "soft_gates": soft_gates,
            "gate_mask_used": gate_mask_used,
            "hard_mask": hard_mask,
            "tau": tau,
            "dynamic_keep_ratio": dynamic_keep_ratio,
            "align_loss": align_loss,
            "distill_loss": distill_loss,
            "alpha_kd_loss": None,
        }

        if return_intermediates:
            return out

        return {
            "logits": logits,
            "loss": loss,
        }
