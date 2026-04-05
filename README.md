# Pruning-Token-VLMs

Pruning-Token-VLMs is a Vision-Language Model (VLM) training project focused on
token pruning for better inference efficiency while preserving answer quality.

## Goals

- Reduce visual token computation with learned pruning.
- Keep VQA-style accuracy competitive under lower keep-ratio budgets.
- Support a practical training pipeline (Stage 1 -> Stage 2) with resume.

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

## ASCII Diagram (Training Flow)

```
+-------------------------+
|   VQAv2 Images + Q/A    |
+------------+------------+
             |
             v
+-------------------------+
| Vision Encoder + Text   |
| Encoder + Projector     |
+------------+------------+
             |
             v
+-------------------------+        +---------------------------+
| Token Scorer / Pruning  |------->| Keep-Ratio Schedule       |
| (soft -> ste -> hard)   |        | (1.0 -> target ratio)     |
+------------+------------+        +---------------------------+
             |
             v
+-------------------------+
| LLM Decoder (Vicuna /   |
| LoRA in Stage 2)        |
+------------+------------+
             |
             v
+-------------------------+
| Answer Generation +     |
| VQA Soft Accuracy Eval  |
+-------------------------+
```

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

Current script behavior in `src/train.py`:

- Stage 1: projector alignment (auto-skip if Stage 1 checkpoint exists).
- Stage 2: pruning-aware tuning with multi-ratio curriculum.
- Auto-resume from latest Stage 2 checkpoint when available.

## Evaluation / Inference

```bash
cd /workspace/VLM_Project
python src/eval.py
python src/infer.py
```

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
- Log keep-ratio, soft/hard keep stats, and validation metrics every epoch.
- Save both `latest` and `best` checkpoints.

## License

Add your preferred license here (MIT/Apache-2.0/etc.).
