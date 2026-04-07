# Listen to the Prompt: Instruction-Aware Token Pruning for Vision-Language Models

This repository implements an instruction-aware token pruning VLM pipeline with
optional token merging for higher efficiency at low visual-token budgets.

Core idea: keep visual tokens that matter for the current question, not just
globally salient tokens.

## Goals

- Instruction-aware token selection conditioned on the text query.
- Robust performance across multiple keep ratios (single model, multi-budget use).
- Efficient inference via hard top-k pruning, with optional token merging.

## Project Structure

```
VLM_Project/
|-- src/
|   |-- train.py
|   |-- eval.py
|   |-- infer.py
|   |-- models/
|   |-- datasets/
|   |   |-- data/                 # local datasets (ignored by git)
|   |-- checkpoints/              # local checkpoints (ignored by git)
|   |-- eval_outputs_paper/       # eval artifacts (ignored by git)
|   |-- tmp_eval/                 # temp outputs (ignored by git)
|-- README.md
|-- .gitignore
```

## ASCII Diagram (Model Architecture: Instruction-Aware + Merge)

```
IMAGE BRANCH
------------
[Image]
  -> [CLIP Vision Encoder (ViT-L/14-336)]
       outputs: cls_emb [B,1024], patch_tokens [B,576,1024]
  -> [Projector: Linear(1024->4096) -> GELU -> Linear(4096->4096)]
       z [B,576,4096]
  -> [CLSScorer]
       cls_q = normalize(Linear(cls_emb))
       S_cls[i] = cosine(cls_q, z_i)                         # [B,576]

TEXT BRANCH
-----------
[Question / Instruction]
  -> [CLIP Text Encoder]
       text_tokens [B,L,768]
  -> [TextImportanceMLP]
       beta = softmax(MLP(text_tokens))                      # [B,L]

CROSS-MODAL INSTRUCTION-AWARE SCORING
-------------------------------------
[InstructionAwareScorer: multi-head text->visual attention]
  Q = LN(W_q * text_tokens), K = LN(W_k * z)
  A = softmax(QK^T / sqrt(d))                                # [B,L,576]
  S_ia = beta^T * A                                          # [B,576]

SCORE FUSION + PRUNING
----------------------
[ScoreFusion]
  S = alpha * zscore(S_ia) + (1 - alpha) * zscore(S_cls)    # [B,576]
  alpha in [0,1] (optionally learnable)

[TokenPruner: top-k by S]
  keep K = round(keep_ratio * 576)
  kept tokens: z_keep [B,K,4096]

  if use_merging=True:
    each dropped token -> nearest kept token (cosine)
    score-weighted merge:
      t_tilde_j = (s_j*t_j + sum_{i->j} s_i*t_i) / (s_j + sum_{i->j} s_i)

LLM DECODING
------------
[Pruned/Merged visual tokens + text tokens]
  -> [LLM Decoder (Vicuna/LLaVA backbone)]
  -> [Response]
```

## Method Overview

- Visual branch: CLIP ViT-L/14-336 image tokens.
- Text branch: CLIP text tokens from the question/prompt.
- Scoring modules:
  - CLS-based saliency scorer.
  - Instruction-aware cross-modal scorer.
  - Learnable score fusion for final token importance.
- Pruning curriculum during Stage 2:
  - Early training: soft gates for dense gradient flow.
  - Middle: STE gating for sharper discrete behavior.
  - Late: structural top-k behavior aligned with deployment.
- Multi-ratio training:
  - Sample keep ratios during training to improve robustness at different budgets.
- Dynamic budget (optional):
  - Predict/adjust keep ratio per sample within configured min-max bounds.
- Token merging:
  - Optional merge at inference (and enabled in current Stage 2 construction) to
    reduce effective token load further.

## Environment Setup

From the project root:

```bash
cd /workspace/VLM_Project
python -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

## Dataset

Expected VQAv2 root used by training script:

```text
src/datasets/data/vqa_v2/
```

Expected files/folders include:

- `v2_OpenEnded_mscoco_train2014_questions.json`
- `v2_mscoco_train2014_annotations.json`
- `v2_OpenEnded_mscoco_val2014_questions.json`
- `v2_mscoco_val2014_annotations.json`
- `train2014/`
- `val2014/`

## Training

Run full pipeline:

```bash
cd /workspace/VLM_Project
python src/train.py
```

Current script behavior in src/train.py:

- Stage 1: projector alignment (auto-skip if Stage 1 checkpoint exists).
- Stage 2: visual instruction tuning + instruction-aware pruning.
- Step-level pruning curriculum: soft -> ste -> structural.
- Multi-ratio sampling support and ratio-adaptive loss scaling.
- Dynamic budget hooks (configurable; off in current default recipe).
- Token merging is enabled in the current model construction for Stage 2.
- Auto-resume from latest Stage 2 checkpoint when available.

## Evaluation / Inference

```bash
cd /workspace/VLM_Project
python src/eval.py
python src/infer.py
```

MME evaluation examples:

```bash
cd /workspace/VLM_Project/src

# single keep ratio
python test_mme.py --keep_ratio 0.7

# keep-ratio sweep (baseline)
python test_mme.py --sweep_keep_ratio

# keep-ratio sweep + token merging
python test_mme.py --sweep_keep_ratio --sweep_ratios 1.0 0.7 0.4 0.2 0.1 --use_merging
```

Useful flags in src/test_mme.py:

- --task_type all|perception|cognition
- --tasks <task1 task2 ...>
- --use_merging
- --batch_size 0 (auto batch size by free VRAM + keep ratio)

## Git Notes (Code Only, No Data)

The repository is configured to avoid pushing local heavy artifacts:

- `src/datasets/data/`
- `src/checkpoints/`
- `src/eval_outputs_paper/`
- `src/tmp_eval/`

If a folder was tracked before, untrack it without deleting local files:

```bash
git rm -r --cached --ignore-unmatch \
  src/datasets/data src/checkpoints src/eval_outputs_paper src/tmp_eval
git add .gitignore
git commit -m "Stop tracking local data/artifacts"
```

## Reproducibility Tips

- Keep `seed` fixed across runs.
- Log keep-ratio schedule/sample, soft/hard keep stats, and validation metrics.
- Compare both merge OFF and merge ON in MME sweeps.
- Save both `latest` and `best` checkpoints.

## License

Add your preferred license here (MIT/Apache-2.0/etc.).
