# Listen to the Prompt: Instruction-Aware Token Pruning for VLMs

# Listen to the Prompt: Instruction-Aware Token Pruning for VLMs

Instruction-aware visual token pruning for Vision-Language Models (VLMs), built on
LLaVA-1.5 (CLIP ViT-L/14@336 + Vicuna-7B).

**Core idea:** score each visual patch token by its relevance to the current question, keep
only the top-k most relevant tokens, and recover accuracy via self-distillation from the
same model run with all tokens (teacher = `keep_ratio=1.0` under `torch.no_grad`).

---

## 1. Architecture

```
                                   Image
                                        │
          CLIP ViT-L/14@336  (frozen)
                                        │
                patch tokens  [B, 576, D_clip]        question text
                                        │                                  │
                     ┌──────┴──────┐            CLIP text encoder (frozen)
                     │             │                  │
           CLSScorer    InstructionAware     β weights
      (param-free)      Scorer            (TextImportanceMLP)
      cosine(CLS,pᵢ)  (param-free)
                     │       β-weighted cross-attn
                     │             │
                     └──ScoreFusion MLP──┘
                          α(q)·S_ia + (1-α(q))·S_cls
                                   α predicted per sample
                                        │
                          Top-k pruning
                          keep_ratio × 576 tokens kept
                                        │
                         Projector MLP  (LLaVA-1.5 init)
                         CLIP space → LLM embedding space
                                        │
                         Vicuna-7B  (frozen)
                                        │
                          answer tokens → loss
```

### Module summary

| Module | File | Params | Trainable | Role |
|---|---|---|---|---|
| `CLSScorer` | `models/cls_scorer.py` | **0** | No | Cosine sim: CLIP CLS vs each patch |
| `InstructionAwareScorer` | `models/instruction_aware.py` | **0** | No | β-weighted text→visual cross-attention |
| `TextImportanceMLP` | `models/text_importance.py` | ~0.13 M | Yes | Learns β weights over question tokens |
| `ScoreFusion` | `models/score_fusion.py` | ~0.15 M | Yes | α-MLP; fuses S_cls and S_ia per sample |
| `TokenPruner` | `models/token_pruner.py` | **0** | No | Hard top-k; optional token merging at inference |
| `Projector` | `models/projector.py` | ~16 M | Yes (one_stage) / No (two_stage) | 2-layer MLP, LLaVA-1.5 weights |
| `CLIP ViT-L/14` | `models/hf_backbones.py` | ~307 M | **Always frozen** | Visual + text feature extraction |
| `Vicuna-7B` | `models/hf_backbones.py` | ~7 B | **Always frozen** | Language model |

**Total trainable params:** ~16.3 M (one_stage) or ~0.28 M (two_stage, projector frozen).

---

## 2. Training Objective

$$
\mathcal{L} = \mathcal{L}_{LM} + \lambda_b \cdot \mathcal{L}_{budget} + \lambda_{KD}(t) \cdot \mathcal{L}_{logitKD}
$$

| Term | Formula | Default weight |
|---|---|---|
| $\mathcal{L}_{LM}$ | Cross-entropy on answer tokens | — |
| $\mathcal{L}_{budget}$ | $\text{MSE}(\bar{g}, r_{target})$, $\bar{g}$ = mean soft gate per sample | $\lambda_b = 0.03$ |
| $\mathcal{L}_{logitKD}$ | $T^2 \cdot \text{KL}(\text{log\_softmax}(z_s/T) \| \text{softmax}(z_t/T))$ on answer positions | $\lambda_{KD} = 0.5$, $T = 1.0$ |

**Teacher:** the same `PruningVLM` model forwarded with `keep_ratio=1.0` (all 576 tokens)
inside `pruning_vlm.forward()` under `torch.no_grad`. No separate checkpoint needed.

**λ_KD warmup (one_stage only):** λ_KD ramps from 0 → 0.5 over the first 20% of steps
(`--kd_warmup_ratio 0.2`), letting the LM loss stabilise before distillation pressure begins.

**Ratio-adaptive scaling:** when `keep_ratio < 0.3`, λ_b is scaled ×1.5 and λ_KD ×2.0
automatically to compensate for extreme pruning.

---

## 3. Pruning Curriculum

The full curriculum runs **across all training steps** (not per epoch), split into three
equal-length phases:

| Phase | Steps fraction | Gate type | Gradient |
|---|---|---|---|
| **Soft** | 0 – 33 % | $\sigma\!\left(\frac{s_i - \tau}{T_{gate}}\right)$ continuous in (0,1) | Dense; every token contributes |
| **STE** | 33 – 66 % | Hard 0/1 mask + straight-through estimator | Sharp selection, still differentiable |
| **Structural** | 66 – 100 % | Exact hard top-k (= inference behaviour) | True pruning |

Additionally:
- **Gate temperature** $T_{gate}$: anneals 1.5 → 0.5 across epochs (gate sharpens over time).
- **keep_ratio curriculum**: linearly anneals `keep_ratio_start=1.0 → keep_ratio=0.5`
     across all steps so the model starts with almost all tokens and gradually tightens.

---

## 4. Training Recipes

### `one_stage` (default, recommended)

Single training loop. Both the **projector** and all **scorer modules** are trainable.
Projector is initialised from LLaVA-1.5 pretrained weights so it starts aligned.

```
Trainable : projector (~16 M)  +  text_importance + score_fusion (~0.28 M)
Frozen    : vision_encoder, text_encoder, llm
Dataset   : LLaVA-665K 10% (train) + LLaVA-665K 10% (val, seed=999)
```

### `two_stage`

Projector is immediately frozen (loaded from LLaVA-1.5). Only scorer heads train.
Use this when you want to isolate scorer learning from projector adaptation.

```
Trainable : text_importance + score_fusion (~0.28 M)
Frozen    : vision_encoder, text_encoder, projector, llm
Dataset   : same as one_stage
```

> Stage 1 projector warmup has been removed because `llava15_init=True` already provides
> a well-aligned projector. Running Stage 1 from scratch is no longer necessary.

---

## 5. Project Structure

```
Pruning-Token-VLMs/
├── src/
│   ├── train.py              # Main training script — all recipes, argparse CLI
│   ├── eval.py               # VQAv2 evaluation + generate_answers helper
│   ├── infer.py              # Single-image / single-question inference
│   ├── test_mme.py           # MME benchmark (sweep keep_ratio, optional merging)
│   ├── models/
│   │   ├── pruning_vlm.py    # PruningVLM — full forward pass, KD, pruning modes
│   │   ├── cls_scorer.py     # CLSScorer (param-free cosine similarity)
│   │   ├── instruction_aware.py  # InstructionAwareScorer (param-free cross-attn)
│   │   ├── text_importance.py    # TextImportanceMLP → β weights
│   │   ├── score_fusion.py       # ScoreFusion → α per sample
│   │   ├── token_pruner.py       # Hard top-k + optional score-weighted merging
│   │   ├── projector.py          # 2-layer MLP; load_from_llava() copies LLaVA-1.5 weights
│   │   └── hf_backbones.py       # CLIPVisionEncoder, CLIPTextEncoder, LLaVALM wrappers
│   ├── datasets/
│   │   ├── llava_dataset.py          # LLaVAInstructDataset (standalone)
│   │   ├── prepare_llava_subset.py   # Download + sample 10% of LLaVA-665K
│   │   └── data/
│   │       ├── llava_665k_10p/       # Default training + val data (git-ignored)
│   │       │   ├── llava_subset.json
│   │       │   └── images/{coco,gqa,textvqa,vg}/
│   │       └── vqa_v2/               # Optional fallback (git-ignored)
│   └── checkpoints/                  # Saved during training (git-ignored)
│       ├── best_model_{recipe}/
│       └── latest_{recipe}/
├── src/requirements.txt
└── README.md
```

Key internal functions in `train.py`:

| Function | Purpose |
|---|---|
| `compute_losses()` | Shared 3-loss forward for all recipes |
| `train_one_stage()` | One-stage training loop with λ_KD warmup |
| `train_stage2()` | Scorer-only training loop (two_stage recipe) |
| `validate_vqa_generation()` | Eval loop; uses LLaVA-665K held-out subset by default |
| `step_pruning_mode()` | Returns `soft / ste / structural` given global step |
| `step_keep_ratio()` | Linear curriculum: `keep_ratio_start → keep_ratio` |
| `load_llava15_pretrained_weights()` | Copies projector + LLM weights from `llava-hf/llava-1.5-7b-hf` |

---

## 6. Environment Setup

```bash
cd /workspace/Pruning-Token-VLMs
python -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

---

## 7. Dataset Layout

### LLaVA-665K 10% (default — train + val)

```
src/datasets/data/llava_665k_10p/
├── llava_subset.json          # ~66.5K samples (10% of LLaVA-665K)
└── images/
          ├── coco/train2017/
          ├── gqa/images/
          ├── textvqa/train_images/
          └── vg/VG_100K/ and VG_100K_2/
```

Training uses `seed=42` to shuffle and pick `max_samples` from the JSON.  
Validation uses **the same JSON with `seed=999`** to draw a non-overlapping held-out subset.

To prepare:
```bash
cd src/datasets
python prepare_llava_subset.py --download-images
```

### VQAv2 (optional — only needed for `eval.py` or `--no_use_llava_data`)

```
src/datasets/data/vqa_v2/
├── v2_OpenEnded_mscoco_train2014_questions.json
├── v2_mscoco_train2014_annotations.json
├── v2_OpenEnded_mscoco_val2014_questions.json
├── v2_mscoco_val2014_annotations.json
├── train2014/
└── val2014/
```

---

## 8. Training Strategy & Commands

### Recommended: `one_stage` at keep_ratio=0.5

This is the standard setup. Projector + scorers train jointly on LLaVA-665K 10%.

```bash
cd /workspace/Pruning-Token-VLMs
python src/train.py \
     --training_recipe one_stage \
     --keep_ratio 0.5 \
     --num_epochs 3 \
     --batch_size 2 \
     --grad_accum_steps 8 \
     --adapter_lr 2e-5 \
     --lambda_budget 0.03 \
     --lambda_kd 0.5 \
     --kd_warmup_ratio 0.2
```

**Effective batch size:** 2 × 8 = 16 samples per update.

---

### Alternative: `two_stage` (scorer-only, lighter)

Projector is frozen at LLaVA-1.5 weights. Only ~0.28 M scorer params train.
Faster per step, but projector cannot adapt to the pruned token distribution.

```bash
python src/train.py \
     --training_recipe two_stage \
     --keep_ratio 0.5 \
     --num_epochs 3 \
     --adapter_lr 2e-5
```

---

### Sweep across keep_ratio values (for methodology comparison)

Train separate checkpoints at different pruning budgets to produce a
accuracy-vs-efficiency curve for the paper:

```bash
for KR in 0.7 0.5 0.3; do
     python src/train.py \
          --training_recipe one_stage \
          --keep_ratio $KR \
          --num_epochs 3 \
          --best_checkpoint_name best_kr${KR} \
          --latest_checkpoint_name latest_kr${KR}
done
```

---

### Resume a crashed run

Auto-resume detects the latest checkpoint automatically:

```bash
python src/train.py --training_recipe one_stage  # resumes if checkpoints/latest_one_stage/ exists
# or explicitly:
python src/train.py --training_recipe one_stage --resume_from src/checkpoints/latest_one_stage
```

---

### What to watch in the logs

```
Epoch 1 | Step 40/500 | Loss 2.12 | LM 1.98 | Budget 0.004 | KD 0.12 |
Keep 0.823 | λ(b/kd) 0.0300/0.1200 | Temp 1.35 | Mode soft | LR 1.23e-05
```

| Signal | Healthy | Warning |
|---|---|---|
| **LM** | Decreasing steadily | Flat or increasing after epoch 1 |
| **Budget** | Converges to ~0 | Stays large → scorer not learning the budget |
| **KD** | Decreasing from ~epoch 1 | Spike at start is OK (warmup), but should fall |
| **SoftKeep ≈ HardKeep** | Values close together | Large gap → gate not sharp enough |
| **Mode** | `soft → ste → structural` | Stuck in soft → check `soft_stage_ratio` |
| **LM spike when Mode → structural** | Small spike normal | Large spike → lower `ste_stage_ratio` |

---

## 9. Evaluation & Inference

```bash
# LLaVA-665K held-out accuracy (generation-based)
cd src
python eval.py

# Single image / question
python infer.py

# MME benchmark at fixed keep_ratio
python test_mme.py --keep_ratio 0.5

# MME sweep across multiple ratios (with and without token merging)
python test_mme.py --sweep_keep_ratio --sweep_ratios 1.0 0.7 0.5 0.3 0.1
python test_mme.py --sweep_keep_ratio --sweep_ratios 1.0 0.7 0.5 0.3 0.1 --use_merging
```

---

## 10. CLI Reference

```bash
python src/train.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--training_recipe` | `one_stage` | `one_stage` or `two_stage` |
| `--keep_ratio` | `0.5` | Target fraction of 576 tokens to keep |
| `--keep_ratio_start` | `1.0` | Curriculum start (anneals to `--keep_ratio`) |
| `--lambda_budget` | `0.03` | λ_b — budget regularisation weight |
| `--lambda_kd` | `0.5` | λ_KD — logit distillation weight |
| `--kd_temperature` | `1.0` | Temperature T for KD softmax |
| `--kd_warmup_ratio` | `0.2` | Fraction of steps to ramp λ_KD from 0 (one_stage) |
| `--num_epochs` | `2` | Training epochs |
| `--batch_size` | `3` | Per-GPU batch size |
| `--grad_accum_steps` | `8` | Gradient accumulation (effective batch = batch × accum) |
| `--adapter_lr` | `2e-5` | LR for scorer modules |
| `--multi_ratio_enabled` | off | Sample a different keep_ratio per step from a pool |
| `--use_llava_data` / `--no_use_llava_data` | `True` | LLaVA-665K 10% (default) vs VQAv2 |
| `--resume_from` | auto | Explicit checkpoint dir to resume from |
| `--stage1_checkpoint` | `None` | (two_stage) Load pre-trained scorer weights before Stage 2 |
| `--best_checkpoint_name` | `best_model_{recipe}` | Subdirectory name under `checkpoints/` |
| `--latest_checkpoint_name` | `latest_{recipe}` | Subdirectory name under `checkpoints/` |
| `--lambda_distill` | *deprecated* | Alias for `--lambda_kd`; will be removed |
| `--lambda_align` | *deprecated* | Align loss removed — silently ignored |
| `--lambda_alpha_kd` | *deprecated* | Alpha KD removed — silently ignored |

---

## 11. Design Notes (for Methodology)

**Why parameter-free scorers?**  
`CLSScorer` and `InstructionAwareScorer` operate entirely in frozen CLIP space and have zero
learned weights. This means the scoring signal is never contaminated by gradient noise from
the LM task. Only the *weighting* of those signals (`TextImportanceMLP` β, `ScoreFusion` α)
is learned — a total of ~0.28 M parameters.

**Why initialise the projector from LLaVA-1.5?**  
The projector architecture exactly mirrors LLaVA-1.5's `multi_modal_projector`
(`Linear(1024→4096) → GELU → Linear(4096→4096)`), so pretrained weights can be copied
directly. This eliminates the need for a projector warmup stage and provides a strong
prior that already maps CLIP features into the Vicuna embedding space.

**Why self-distillation?**  
The teacher and student share the same weights — the teacher is just a second forward pass
of the same model with `keep_ratio=1.0`. This means: (1) no separate teacher training or
storage, (2) the teacher distribution is always up-to-date with the student, and (3) the KD
signal captures exactly what information the pruned model is losing, not some external proxy.

**Why a three-phase pruning curriculum?**  
Starting directly with hard top-k causes gradient collapse because the gate becomes
non-differentiable before the scorer has learned anything. The soft→STE→structural
progression gradually tightens the gradient path so the scorer converges stably.

**Token merging (inference only).**  
Pruned tokens can be merged (score-weighted nearest-neighbour in CLIP space) into kept
tokens at inference time. No retraining — the merge is applied after the trained scorer
selects which tokens to keep.

---

## License

Add your preferred license here.
