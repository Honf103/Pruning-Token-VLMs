# Listen to the Prompt: Instruction-Aware Token Pruning for Vision-Language Models

# Listen to the Prompt: Instruction-Aware Token Pruning for Vision-Language Models

This repository implements instruction-aware token pruning for VLMs, with an
updated Knowledge Distillation (KD) training recipe focused on fair comparison.

Core idea: keep visual tokens relevant to the current question, then distill
from the frozen teacher backbone into lightweight scorer modules.

## What Is Updated

The latest Stage 2 training logic has these key changes.

1. Backbone frozen for fair comparison.
2. Proper attention-based alignment KD replaces hidden-state proxy alignment.
3. Question-conditioned score fusion predicts alpha per sample from text CLS.
4. Added alpha KD supervision to teach when instruction-aware vs CLS scoring
   should dominate.

## Project Structure

```
Pruning-Token-VLMs/
|-- src/
|   |-- train.py
|   |-- eval.py
|   |-- infer.py
|   |-- models/
|   |-- datasets/
|   |   |-- data/                 # local datasets (ignored by git)
|   |-- checkpoints/              # local checkpoints (ignored by git)
|-- README.md
```

## Stage-Wise Trainable Plan

### Stage 1

Trainable by default.

1. projector

Frozen.

1. vision_encoder
2. text_encoder
3. cls_scorer
4. text_importance
5. instruction_aware
6. score_fusion
7. token_pruner
8. llm

Optional mode.

1. If train_budget_head_only=True, projector main projection is frozen and only
   projector budget head branches are trained.

### Stage 2 (Current KD Recipe)

Trainable.

1. cls_scorer
2. text_importance
3. instruction_aware
4. score_fusion

Frozen.

1. vision_encoder
2. text_encoder
3. projector
4. llm

This is the intended fair-comparison setup in current code.

## KD Losses in Stage 2

Total loss is:

$$
\mathcal{L}=\mathcal{L}_{lm}+\lambda_{budget}\mathcal{L}_{budget}+\lambda_{distill}\mathcal{L}_{distill}+\lambda_{align}\mathcal{L}_{align}+\lambda_{\alpha}\mathcal{L}_{\alpha\_kd}
$$

Terms.

1. LM loss: standard causal LM loss on answer tokens.
2. Distill loss: KL between student and teacher text logits.
3. Align loss: KL between scorer importance and teacher cross-attention
   importance from selected LLM layers.
4. Alpha KD loss: MSE between predicted alpha and alpha target computed from
   LM-loss gap between alpha=1 and alpha=0 pruning passes.
5. Budget loss: keep-ratio regularization.

## Environment Setup

From project root.

```bash
cd /workspace/Pruning-Token-VLMs
python -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

## Dataset Layout

Expected root used by training script:

```text
src/datasets/data/vqa_v2/
```

Required files/folders.

1. v2_OpenEnded_mscoco_train2014_questions.json
2. v2_mscoco_train2014_annotations.json
3. v2_OpenEnded_mscoco_val2014_questions.json
4. v2_mscoco_val2014_annotations.json
5. train2014/
6. val2014/

## How To Train the New Code

### Quick Run (full pipeline)

```bash
cd /workspace/Pruning-Token-VLMs
python src/train.py
```

Behavior.

1. Stage 1 runs first and auto-skips if Stage 1 checkpoint already exists.
2. Stage 2 then runs with KD-focused frozen-backbone setup.
3. Auto-resume uses latest Stage 2 checkpoint if present.

### Current Default Stage 2 Recipe in Code

The bottom call in src/train.py currently uses.

1. keep_ratio=0.333
2. lambda_distill=1.0
3. lambda_align=0.5
4. lambda_alpha_kd=0.2
5. alpha_kd_every_n_steps=20
6. question_conditioned_alpha=True
7. attn_distill_layers=[8, 16, 23]
8. dynamic_budget_enabled=False
9. projector_unfreeze_threshold=0.0

### What To Watch While Training

Stage 2 logs include.

1. LM
2. Budget
3. Distill
4. AlignLoss
5. AlphaKD
6. Alpha(mean)
7. SoftKeep / HardKeep
8. Keep(schedule/sample)

These are the most useful signals to verify KD is behaving as expected.

## Evaluation / Inference

```bash
cd /workspace/Pruning-Token-VLMs
python src/eval.py
python src/infer.py
```

MME examples.

```bash
cd /workspace/Pruning-Token-VLMs/src

python test_mme.py --keep_ratio 0.7
python test_mme.py --sweep_keep_ratio
python test_mme.py --sweep_keep_ratio --sweep_ratios 1.0 0.7 0.4 0.2 0.1 --use_merging
```

## Reproducibility Tips

1. Keep seed fixed.
2. Track best and latest checkpoints.
3. Compare both merge OFF and merge ON for MME sweeps.
4. Monitor AlignLoss and AlphaKD together, not only LM loss.

## License

Add your preferred license here.
