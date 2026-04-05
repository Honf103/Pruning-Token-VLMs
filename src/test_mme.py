"""
test_mme.py  —  MME evaluation for PruningVLM

MME (MultiModal Evaluation Benchmark) covers 14 subtasks across perception
and cognition capabilities.

Perception tasks (10):
  existence, count, position, color, poster, celebrity,
  scene, landmark, artwork, OCR

Cognition tasks (4):
  commonsense_reasoning, numerical_calculation,
  text_translation, code_reasoning

Scoring:
  Score   — total correct answers per task (max = 2 × num_images)
  Score+  — images where BOTH questions are answered correctly (per task)
  Total Perception Score  = sum of Score across 10 perception tasks
  Total Cognition  Score  = sum of Score across 4 cognition tasks
  Total MME Score         = Perception + Cognition

Expected data layout (--mme_data_dir):
  <mme_data_dir>/
    existence/
        COCO_val2014_*.jpg    (or symlink to COCO val2014 dir)
        existence.txt         # image_name TAB question TAB answer
    count/
        ...
    ...

Annotation files are tab-separated, one line per question:
  COCO_val2014_000000000042.jpg\tIs there a car in the image?\tYes

Official data: https://huggingface.co/datasets/BradyFU/MME
              https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models

For COCO-based tasks (existence, count, position, color) you can pass
--coco_image_dir to reuse the already-downloaded COCO val2014 images instead
of duplicating them inside each task folder.

Usage:
  # Print data setup instructions
  python test_mme.py --setup_instructions

  # Evaluate at kr=0.7 (auto-detects available tasks)
  python test_mme.py --keep_ratio 0.7

  # Baseline (no pruning)
  python test_mme.py --keep_ratio 1.0

  # Only perception or cognition tasks
  python test_mme.py --task_type perception --keep_ratio 0.7

  # Specific tasks only
  python test_mme.py --tasks existence count color --keep_ratio 0.7

  # Sweep across keep ratios
  python test_mme.py --sweep_keep_ratio

  # Custom sweep with token merging
  python test_mme.py --sweep_keep_ratio --sweep_ratios 1.0 0.7 0.4 0.1 --use_merging
"""

import io
import os
import sys
import glob
import json
import math
import time
import argparse
import collections
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPTokenizer
from peft import LoraConfig, TaskType, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.pruning_vlm import PruningVLM
from eval import (
    load_lightweight_checkpoint,
    prepare_images_for_model,
    generate_answers,
    normalize_vqa_answer,
)
from utils.misc import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CKPT_ROOT  = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "eval_outputs_paper")

MME_DATA_DIR    = os.path.join(BASE_DIR, "datasets", "data", "mme")
MME_PARQUET_DIR = os.path.join(MME_DATA_DIR, "data")
COCO_IMAGE_DIR  = os.path.join(BASE_DIR, "datasets", "data", "vqa_v2", "val2014")

# Vicuna system prompt
_VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# ─────────────────────────────────────────────────────────────────────────────
# MME task definitions
# ─────────────────────────────────────────────────────────────────────────────
MME_PERCEPTION_TASKS = [
    "existence",
    "count",
    "position",
    "color",
    "posters",       # dataset uses 'posters' (not 'poster')
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
]

MME_COGNITION_TASKS = [
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
]

MME_ALL_TASKS = MME_PERCEPTION_TASKS + MME_COGNITION_TASKS

# All MME questions are binary yes/no (confirmed from lmms-lab/MME parquet)
_OPEN_ENDED_TASKS: set = set()


# ─────────────────────────────────────────────────────────────────────────────
# Setup instructions
# ─────────────────────────────────────────────────────────────────────────────
_SETUP_TEXT = """
MME Dataset Setup Instructions
================================

1. Download the dataset (images + annotations bundled as Parquet):
     huggingface-cli download lmms-lab/MME --repo-type dataset \\
         --local-dir {mme_dir}

   This downloads ~865 MB and requires no extra image downloads.

2. After download the layout should be:
     {mme_dir}/
       data/
         test-00000-of-00004-*.parquet
         test-00001-of-00004-*.parquet
         test-00002-of-00004-*.parquet
         test-00003-of-00004-*.parquet

3. Run evaluation:
     python test_mme.py --keep_ratio 0.7
""".strip()


def print_setup_instructions(mme_dir: str, coco_dir: str):
    print(_SETUP_TEXT.format(mme_dir=mme_dir, coco_dir=coco_dir))


# ─────────────────────────────────────────────────────────────────────────────
# Parquet loader
# ─────────────────────────────────────────────────────────────────────────────
def load_mme_parquet(parquet_dir: str) -> "pandas.DataFrame":
    """
    Load all Parquet shards under parquet_dir into a single DataFrame.
    Columns: question_id, image (dict with 'bytes'/'path'), question, answer, category.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas pyarrow")

    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {parquet_dir}\n"
            "Run: huggingface-cli download lmms-lab/MME --repo-type dataset "
            f"--local-dir {os.path.dirname(parquet_dir)}"
        )
    dfs = [pd.read_parquet(f) for f in files]
    import pandas as pd
    df = pd.concat(dfs, ignore_index=True)
    print(f"  [MME] Loaded {len(df)} samples from {len(files)} Parquet shards")
    print(f"  [MME] Categories: {sorted(df['category'].unique().tolist())}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────
def build_mme_prompt(question: str) -> str:
    """Vicuna-style prompt for MME binary yes/no questions."""
    # <image> token marks where visual embeddings are inserted inline,
    # matching the LLaVA-1.5 pretraining format: USER: <image>\n{question}
    return (
        f"{_VICUNA_SYSTEM} USER: <image>\n{question}\n"
        "Answer with yes or no. ASSISTANT:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MME Dataset
# ─────────────────────────────────────────────────────────────────────────────
class MMEDataset(Dataset):
    """
    One MME task loaded from the lmms-lab/MME Parquet DataFrame.

    df_task: pre-filtered DataFrame (rows where category == task_name).
    Images are stored as bytes in df['image']['bytes'] — no disk I/O needed.
    """

    def __init__(
        self,
        task_name: str,
        df_task,          # pandas.DataFrame filtered to this task
        max_samples: Optional[int] = None,
    ):
        self.task_name = task_name
        self.samples: List[Dict] = []

        for idx, row in enumerate(df_task.itertuples(index=False)):
            img_info   = row.image                       # dict: {'bytes': ..., 'path': ...}
            image_name = row.question_id                 # e.g. 'existence/COCO_val2014_*.jpg'
            question   = str(row.question).strip()
            answer     = str(row.answer).strip().lower()

            self.samples.append({
                "image_bytes": img_info["bytes"],
                "image_name":  image_name,
                "question":    question,
                "prompt_text": build_mme_prompt(question),
                "answer":      answer,
                "question_id": idx,
            })

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"  [MME/{task_name}] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(io.BytesIO(s["image_bytes"])).convert("RGB")
        return {
            "image":       img,
            "image_name":  s["image_name"],
            "question":    s["question"],
            "prompt_text": s["prompt_text"],
            "answer":      s["answer"],
            "question_id": s["question_id"],
        }


def mme_collate_fn(batch: List[Dict]) -> Dict:
    return {
        "image":       [x["image"]       for x in batch],
        "image_name":  [x["image_name"]  for x in batch],
        "question":    [x["question"]    for x in batch],
        "prompt_text": [x["prompt_text"] for x in batch],
        "answer":      [x["answer"]      for x in batch],
        "question_id": [x["question_id"] for x in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_yes_no(raw_pred: str) -> str:
    """Returns 'yes', 'no', or 'other'."""
    text = normalize_vqa_answer(raw_pred).lower()
    if text in ("yes", "no"):
        return text
    for w in text.split():
        if w == "yes":
            return "yes"
        if w == "no":
            return "no"
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# MME scoring
# ─────────────────────────────────────────────────────────────────────────────
def compute_mme_task_score(records: List[Dict]) -> Dict:
    """
    Compute MME Score and Score+ for a single task.

    records: list of dicts with keys:
        image_name, answer (gt), pred, correct (0/1)

    Score  = sum(correct)  (max = len(records))
    Score+ = number of images where ALL questions are correct
             (each image typically has 2 questions in MME)
    """
    total   = len(records)
    correct = sum(r["correct"] for r in records)

    # Group by image to compute Score+
    by_image: Dict[str, List[int]] = collections.defaultdict(list)
    for r in records:
        by_image[r["image_name"]].append(r["correct"])

    score_plus = sum(1 for corrects in by_image.values() if all(c == 1 for c in corrects))

    accuracy = correct / max(1, total) * 100

    return {
        "Score":       correct,
        "Score_Plus":  score_plus,
        "Accuracy":    round(accuracy, 2),
        "num_samples": total,
        "num_images":  len(by_image),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch-size auto-tuning (same calibration as test_pope.py)
# ─────────────────────────────────────────────────────────────────────────────
_MB_PER_SAMPLE_BY_KR = {
    1.0: 800,
    0.7: 600,
    0.4: 450,   # measured 346 MB/sample (diag); 390 was too low → caused bs to jump to 32
    0.2: 255,
    0.1: 190,
}
_BS_CLAMP_MIN = 4
_BS_CLAMP_MAX = 32


def estimate_optimal_batch_size(keep_ratio: float, vram_headroom: float = 0.80) -> int:
    if not torch.cuda.is_available():
        return 16
    kr_keys = sorted(_MB_PER_SAMPLE_BY_KR.keys())
    nearest_kr = min(kr_keys, key=lambda k: abs(k - keep_ratio))
    mb_per_sample = _MB_PER_SAMPLE_BY_KR[nearest_kr]
    free_mb = (
        torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    ) / 1024 ** 2
    bs = int(free_mb * vram_headroom / mb_per_sample)
    bs = max(_BS_CLAMP_MIN, min(_BS_CLAMP_MAX, bs))
    bs = 2 ** int(math.log2(bs))
    return bs


# ─────────────────────────────────────────────────────────────────────────────
# Build model
# ─────────────────────────────────────────────────────────────────────────────
def build_and_load_model(
    checkpoint_dir: str,
    keep_ratio: float,
    use_merging: bool,
    projector_hidden_dim: int,
    ia_hidden_dim: int,
    llava15_model_name: str,
    device: str,
    use_bf16: bool,
) -> PruningVLM:
    model = PruningVLM(
        clip_model_name="openai/clip-vit-large-patch14-336",
        llm_model_name=llava15_model_name,
        keep_ratio=keep_ratio,
        alpha=0.6,
        learnable_alpha=True,
        llm_torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        gate_temperature=0.2,
        gate_score_scale=5.0,
        gate_threshold_mode="topk",
        llm_use_grad=False,
        llm_scale_visual_prefix=False,
        projector_hidden_dim=projector_hidden_dim,
        ia_hidden_dim=ia_hidden_dim,
        use_merging=use_merging,
    )

    for p in model.vision_encoder.parameters():
        p.requires_grad = False
    for p in model.text_encoder.parameters():
        p.requires_grad = False

    lora_adapter_path = os.path.join(checkpoint_dir, "llm_lora_adapter")
    if os.path.exists(lora_adapter_path):
        # LoRA checkpoint (freeze_llm=False training path)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model.llm.model = get_peft_model(model.llm.model, lora_config)
    # else: freeze_llm=True path — load_lightweight_checkpoint will handle it

    model, _ = load_lightweight_checkpoint(model, checkpoint_dir, device)
    model = model.to(device)
    model.eval()
    model.set_keep_ratio(keep_ratio)
    if use_merging:
        print("  [Token Merging] enabled")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Single-task eval loop
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_mme_task(
    model: PruningVLM,
    task_name: str,
    df_task,               # pandas.DataFrame filtered to this task
    keep_ratio: float,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int],
    clip_tokenizer: CLIPTokenizer,
    clip_max_length: int,
    llm_max_length: int,
    max_new_tokens: int,
    num_beams: int,
    device: str,
    use_bf16: bool,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate one MME task. Returns (task_metrics, per_sample_records)."""

    if batch_size == 0:
        batch_size = estimate_optimal_batch_size(keep_ratio)
        print(f"  [AutoBS] batch_size={batch_size} (kr={keep_ratio:.2f})")

    dataset = MMEDataset(
        task_name=task_name,
        df_task=df_task,
        max_samples=max_samples,
    )
    if len(dataset) == 0:
        print(f"  [MME/{task_name}] No samples found — skipping.")
        return None, []

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=mme_collate_fn,
        drop_last=False,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    llm_tokenizer = model.llm.tokenizer
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    records: List[Dict] = []
    kept_tokens_sum  = 0.0
    total_tokens_sum = 0.0
    latency_sum      = 0.0
    other_count      = 0
    current_bs       = batch_size

    for batch_idx, batch in enumerate(loader):
        images       = prepare_images_for_model(batch["image"], device=device)
        questions    = batch["question"]
        prompt_texts = batch["prompt_text"]
        gt_answers   = batch["answer"]
        image_names  = batch["image_name"]
        qids         = batch["question_id"]

        clip_enc = clip_tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=clip_max_length,
        )
        clip_ids  = clip_enc.input_ids.to(device, non_blocking=True)
        clip_mask = clip_enc.attention_mask.to(device, non_blocking=True)

        llm_enc = llm_tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=llm_max_length,
        )
        p_ids  = llm_enc.input_ids.to(device, non_blocking=True)
        p_mask = llm_enc.attention_mask.to(device, non_blocking=True)

        # OOM-safe inference
        t0 = time.perf_counter()
        sub_bs = current_bs
        preds, info = None, None
        merged_hard_mask, merged_proj = None, None

        while True:
            try:
                if sub_bs >= len(questions):
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16 if use_bf16 else torch.float16,
                        enabled=torch.cuda.is_available(),
                    ):
                        preds, info = generate_answers(
                            model=model,
                            images=images,
                            clip_input_ids=clip_ids,
                            clip_attention_mask=clip_mask,
                            prompt_input_ids=p_ids,
                            prompt_attention_mask=p_mask,
                            max_new_tokens=max_new_tokens,
                            num_beams=num_beams,
                            use_hard_pruning=True,
                        )
                    break
                else:
                    preds = []
                    for start in range(0, len(questions), sub_bs):
                        sl = slice(start, start + sub_bs)
                        with torch.autocast(
                            device_type="cuda",
                            dtype=torch.bfloat16 if use_bf16 else torch.float16,
                            enabled=torch.cuda.is_available(),
                        ):
                            sp, si = generate_answers(
                                model=model,
                                images=images[sl],
                                clip_input_ids=clip_ids[sl],
                                clip_attention_mask=clip_mask[sl],
                                prompt_input_ids=p_ids[sl],
                                prompt_attention_mask=p_mask[sl],
                                max_new_tokens=max_new_tokens,
                                num_beams=num_beams,
                                use_hard_pruning=True,
                            )
                        preds.extend(sp)
                        merged_hard_mask = si["hard_mask"] if merged_hard_mask is None \
                            else torch.cat([merged_hard_mask, si["hard_mask"]], dim=0)
                        merged_proj = si["projected_tokens"] if merged_proj is None \
                            else torch.cat([merged_proj, si["projected_tokens"]], dim=0)
                    info = {"hard_mask": merged_hard_mask, "projected_tokens": merged_proj}
                    break
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                new_sub = max(1, sub_bs // 2)
                if new_sub == sub_bs:
                    raise
                print(f"  [OOM] sub_bs {sub_bs} → {new_sub}, retrying")
                sub_bs = new_sub
                current_bs = sub_bs

        latency_sum += time.perf_counter() - t0

        N_total = info["projected_tokens"].shape[1]
        kept    = info["hard_mask"].sum(dim=1).float().mean().item()
        kept_tokens_sum  += kept * len(preds)
        total_tokens_sum += N_total * len(preds)

        for i, raw_pred in enumerate(preds):
            gt = gt_answers[i]

            # All MME tasks are binary yes/no
            pred_norm = extract_yes_no(raw_pred)
            if pred_norm == "other":
                other_count += 1
            gt_norm = gt.strip().lower()
            correct = int(pred_norm == gt_norm)

            records.append({
                "question_id": qids[i],
                "image_name":  image_names[i],
                "question":    questions[i],
                "gt":          gt,
                "raw_pred":    raw_pred,
                "pred":        pred_norm,
                "correct":     correct,
            })

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
            n_done = len(records)
            running_acc = sum(r["correct"] for r in records) / max(1, n_done) * 100
            print(f"    [{task_name}] [{batch_idx+1:>3}/{len(loader)}]  "
                  f"acc={running_acc:.1f}%  n={n_done}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    task_score = compute_mme_task_score(records)
    task_score["other_predictions"] = other_count

    total_n      = len(records)
    throughput   = total_n / max(1e-6, latency_sum)
    ms_per_image = latency_sum / max(1, total_n) * 1000
    kept_ratio_act = kept_tokens_sum / max(1, total_tokens_sum)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
    )

    result = {
        "task":                    task_name,
        "task_type":               "perception" if task_name in MME_PERCEPTION_TASKS else "cognition",
        "keep_ratio":              keep_ratio,
        **task_score,
        "Token_Keep_Ratio_Actual": round(kept_ratio_act, 4),
        "Tokens_Kept_Avg":         round(kept_tokens_sum / max(1, total_n), 1),
        "Tokens_Total":            int(total_tokens_sum / max(1, total_n)),
        "Latency_ms_per_image":    round(ms_per_image, 1),
        "Throughput_img_per_sec":  round(throughput, 2),
        "Peak_VRAM_MB":            round(peak_vram_mb, 1),
    }
    return result, records


# ─────────────────────────────────────────────────────────────────────────────
# Discover available tasks from loaded DataFrame
# ─────────────────────────────────────────────────────────────────────────────
def discover_tasks(df, requested_tasks: List[str]) -> List[str]:
    """
    Returns list of task names that are present in the DataFrame.
    df must have a 'category' column.
    """
    available_cats = set(df["category"].unique().tolist())
    available = [t for t in requested_tasks if t in available_cats]
    missing   = [t for t in requested_tasks if t not in available_cats]
    if missing:
        print(f"  [MME] Tasks not in dataset: {missing}")
    return available


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate scores
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_mme_scores(results_by_task: Dict[str, Dict]) -> Dict:
    perception_score = sum(
        r["Score"] for name, r in results_by_task.items()
        if name in MME_PERCEPTION_TASKS
    )
    cognition_score = sum(
        r["Score"] for name, r in results_by_task.items()
        if name in MME_COGNITION_TASKS
    )
    perception_plus = sum(
        r["Score_Plus"] for name, r in results_by_task.items()
        if name in MME_PERCEPTION_TASKS
    )
    cognition_plus = sum(
        r["Score_Plus"] for name, r in results_by_task.items()
        if name in MME_COGNITION_TASKS
    )
    total_score = perception_score + cognition_score
    total_plus  = perception_plus + cognition_plus

    return {
        "Total_Score":         total_score,
        "Total_Score_Plus":    total_plus,
        "Perception_Score":    perception_score,
        "Perception_Score_Plus": perception_plus,
        "Cognition_Score":     cognition_score,
        "Cognition_Score_Plus": cognition_plus,
        "num_tasks_evaluated": len(results_by_task),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────────────────────
def print_mme_table(results_by_task: Dict[str, Dict], keep_ratio: float, use_merging: bool):
    merge_tag = " + merge" if use_merging else ""
    print(f"\n{'='*70}")
    print(f"  MME RESULTS  —  keep_ratio={keep_ratio:.2f}{merge_tag}")
    print(f"{'='*70}")

    for type_label, task_list in [("PERCEPTION", MME_PERCEPTION_TASKS),
                                   ("COGNITION",  MME_COGNITION_TASKS)]:
        tasks_here = [t for t in task_list if t in results_by_task]
        if not tasks_here:
            continue
        print(f"\n  {type_label}")
        print(f"  {'Task':<28}  {'Score':>6}  {'Score+':>7}  {'Acc':>7}  {'N':>5}")
        print("  " + "─" * 56)
        for task_name in tasks_here:
            r = results_by_task[task_name]
            print(f"  {task_name:<28}  {r['Score']:>6}  {r['Score_Plus']:>7}  "
                  f"{r['Accuracy']:>7.2f}  {r['num_samples']:>5}")

    agg = aggregate_mme_scores(results_by_task)
    print(f"\n  {'─'*56}")
    print(f"  Total MME Score    : {agg['Total_Score']:>6}   (Score+: {agg['Total_Score_Plus']})")
    print(f"  Perception Score   : {agg['Perception_Score']:>6}   (Score+: {agg['Perception_Score_Plus']})")
    print(f"  Cognition  Score   : {agg['Cognition_Score']:>6}   (Score+: {agg['Cognition_Score_Plus']})")

    # Efficiency stats from first available task
    if results_by_task:
        first = next(iter(results_by_task.values()))
        print(f"\n  Tokens: {first['Tokens_Kept_Avg']:.0f}/{first['Tokens_Total']} "
              f"(actual ratio={first['Token_Keep_Ratio_Actual']:.4f})")
        print(f"  Latency: {first['Latency_ms_per_image']:.1f} ms/img  "
              f"Throughput: {first['Throughput_img_per_sec']:.2f} img/s  "
              f"Peak VRAM: {first['Peak_VRAM_MB']:.0f} MB")
    print(f"{'='*70}\n")


def print_sweep_table(sweep_results: List[Dict]):
    print(f"\n{'='*75}")
    print("  MME SWEEP TABLE")
    print(f"{'='*75}")
    header = (f"  {'kr':>5}  {'Total':>7}  {'Perc.':>7}  {'Cogn.':>7}  "
              f"{'T+':>6}  {'Tokens':>8}  {'ms/img':>7}")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for row in sweep_results:
        kr  = row["keep_ratio"]
        agg = row.get("aggregate", {})
        eff = row.get("efficiency", {})
        print(
            f"  {kr:>5.2f}  "
            f"{agg.get('Total_Score', 0):>7}  "
            f"{agg.get('Perception_Score', 0):>7}  "
            f"{agg.get('Cognition_Score', 0):>7}  "
            f"{agg.get('Total_Score_Plus', 0):>6}  "
            f"{eff.get('Tokens_Kept_Avg', 0):>8.0f}  "
            f"{eff.get('Latency_ms_per_image', 0):>7.1f}"
        )
    print(f"{'='*75}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main eval for one keep_ratio
# ─────────────────────────────────────────────────────────────────────────────
def run_mme_eval(
    args,
    keep_ratio: float,
    model: Optional[PruningVLM] = None,
) -> Dict:
    """
    Evaluate all available MME tasks for a given keep_ratio.
    Returns summary dict with 'keep_ratio', 'by_task', 'aggregate'.
    """
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    own_model = model is None
    if own_model:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        print(f"\n{'='*70}")
        print(f"  PruningVLM  —  MME Evaluation")
        print(f"  keep_ratio={keep_ratio:.2f}  ckpt={args.checkpoint}")
        print(f"{'='*70}\n")
        model = build_and_load_model(
            checkpoint_dir=args.checkpoint,
            keep_ratio=keep_ratio,
            use_merging=args.use_merging,
            projector_hidden_dim=args.projector_hidden_dim,
            ia_hidden_dim=args.ia_hidden_dim,
            llava15_model_name=args.llava15_model_name,
            device=device,
            use_bf16=use_bf16,
        )
    else:
        model.set_keep_ratio(keep_ratio)

    # Determine which tasks to run
    if args.tasks:
        requested = args.tasks
    elif args.task_type == "perception":
        requested = list(MME_PERCEPTION_TASKS)
    elif args.task_type == "cognition":
        requested = list(MME_COGNITION_TASKS)
    else:
        requested = list(MME_ALL_TASKS)

    # Load parquet data (cheap — done once per run_mme_eval call,
    # but during sweep the df is reloaded per kr; could be passed in as arg
    # but keeping it simple — it's only ~865MB total and loads in <1s)
    try:
        df_all = load_mme_parquet(MME_PARQUET_DIR)
    except FileNotFoundError as e:
        print(f"  [MME] {e}")
        print("  Run: python test_mme.py --setup_instructions")
        if own_model:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {}

    available = discover_tasks(df_all, requested)
    if not available:
        print("  [MME] No tasks available in the loaded data.")
        if own_model:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {}

    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")
    results_by_task: Dict[str, Dict] = {}
    records_by_task: Dict[str, List] = {}

    for task_name in available:
        print(f"\n  Evaluating MME task: {task_name}")
        df_task = df_all[df_all["category"] == task_name].reset_index(drop=True)

        task_result, task_records = run_mme_task(
            model=model,
            task_name=task_name,
            df_task=df_task,
            keep_ratio=keep_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            clip_tokenizer=clip_tokenizer,
            clip_max_length=args.clip_max_length,
            llm_max_length=args.llm_max_length,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            device=device,
            use_bf16=use_bf16,
        )
        if task_result is not None:
            results_by_task[task_name] = task_result
            records_by_task[task_name] = task_records

    print_mme_table(results_by_task, keep_ratio=keep_ratio, use_merging=args.use_merging)

    agg = aggregate_mme_scores(results_by_task)

    # Efficiency snapshot from first available task
    efficiency = {}
    if results_by_task:
        first = next(iter(results_by_task.values()))
        efficiency = {
            "Tokens_Kept_Avg":        first["Tokens_Kept_Avg"],
            "Tokens_Total":           first["Tokens_Total"],
            "Token_Keep_Ratio_Actual": first["Token_Keep_Ratio_Actual"],
            "Latency_ms_per_image":   first["Latency_ms_per_image"],
            "Throughput_img_per_sec": first["Throughput_img_per_sec"],
            "Peak_VRAM_MB":           first["Peak_VRAM_MB"],
        }

    summary = {
        "keep_ratio":  keep_ratio,
        "use_merging": args.use_merging,
        "checkpoint":  args.checkpoint,
        "by_task":     results_by_task,
        "aggregate":   agg,
        "efficiency":  efficiency,
    }

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    merge_tag = "_merge" if args.use_merging else ""
    kr_tag    = f"kr{keep_ratio:.2f}"

    summary_path = os.path.join(args.output_dir, f"mme_summary_{kr_tag}{merge_tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved → {summary_path}")

    for task_name, records in records_by_task.items():
        rec_path = os.path.join(
            args.output_dir, f"mme_records_{task_name}_{kr_tag}{merge_tag}.jsonl"
        )
        with open(rec_path, "w", encoding="utf-8") as fout:
            for r in records:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Per-sample records saved → {args.output_dir}/mme_records_*_{kr_tag}{merge_tag}.jsonl")

    if own_model:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────
def run_sweep(args):
    keep_ratios = args.sweep_ratios if args.sweep_ratios else [1.0, 0.7, 0.4, 0.2, 0.1]
    sweep_results = []

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"\n{'='*70}")
    print(f"  PruningVLM  —  MME Sweep  ratios={keep_ratios}")
    print(f"  ckpt={args.checkpoint}")
    print(f"{'='*70}\n")

    model = build_and_load_model(
        checkpoint_dir=args.checkpoint,
        keep_ratio=keep_ratios[0],
        use_merging=args.use_merging,
        projector_hidden_dim=args.projector_hidden_dim,
        ia_hidden_dim=args.ia_hidden_dim,
        llava15_model_name=args.llava15_model_name,
        device=device,
        use_bf16=use_bf16,
    )

    for kr in keep_ratios:
        print(f"\n{'─'*60}")
        print(f"  Sweep  keep_ratio = {kr:.2f}")
        print(f"{'─'*60}")
        summary = run_mme_eval(args, keep_ratio=kr, model=model)
        if summary:
            sweep_results.append({
                "keep_ratio": kr,
                "by_task":    summary.get("by_task", {}),
                "aggregate":  summary.get("aggregate", {}),
                "efficiency": summary.get("efficiency", {}),
            })

    print_sweep_table(sweep_results)

    merge_tag  = "_merge" if args.use_merging else ""
    sweep_path = os.path.join(args.output_dir, f"mme_sweep{merge_tag}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Sweep saved → {sweep_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="MME evaluation for PruningVLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=os.path.join(CKPT_ROOT, "best_model_accuracy"),
                   help="Checkpoint directory")
    p.add_argument("--mme_data_dir", default=MME_DATA_DIR,
                   help="Root MME directory (should contain a 'data/' subdir with Parquet files)")
    p.add_argument("--coco_image_dir", default=COCO_IMAGE_DIR,
                   help="COCO val2014 image directory (unused; images embedded in Parquet)")
    p.add_argument("--keep_ratio",    type=float, default=0.7)
    p.add_argument("--use_merging",   action="store_true",
                   help="Enable inference-time token merging (no retraining needed)")
    p.add_argument("--sweep_keep_ratio", action="store_true",
                   help="Sweep over keep_ratios [1.0, 0.7, 0.4, 0.2, 0.1]")
    p.add_argument("--sweep_ratios", type=float, nargs="+", default=None,
                   help="Custom list for sweep, e.g. --sweep_ratios 1.0 0.7 0.4 0.1")
    p.add_argument("--task_type", choices=["all", "perception", "cognition"], default="all",
                   help="Which task group to evaluate")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Specific tasks to evaluate (overrides --task_type), "
                        "e.g. --tasks existence count color")
    p.add_argument("--batch_size",  type=int, default=0,
                   help="Batch size. 0 = auto-detect based on free VRAM and keep_ratio")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap samples per task (default: all)")
    p.add_argument("--max_new_tokens",  type=int, default=16,
                   help="Max tokens to generate (16 sufficient for most MME tasks)")
    p.add_argument("--num_beams",       type=int, default=1)
    p.add_argument("--clip_max_length", type=int, default=64)
    p.add_argument("--llm_max_length",  type=int, default=256)
    p.add_argument("--output_dir",      default=OUTPUT_DIR)
    p.add_argument("--projector_hidden_dim", type=int, default=2048)
    p.add_argument("--ia_hidden_dim",        type=int, default=512)
    p.add_argument("--llava15_model_name",   default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--setup_instructions", action="store_true",
                   help="Print data setup instructions and exit")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.setup_instructions:
        print_setup_instructions(args.mme_data_dir, args.coco_image_dir)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep_keep_ratio:
        run_sweep(args)
    else:
        run_mme_eval(args, keep_ratio=args.keep_ratio)


if __name__ == "__main__":
    main()
