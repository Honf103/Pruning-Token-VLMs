"""
test_pope.py  —  POPE evaluation for PruningVLM

POPE (Polling-based Object Probing Evaluation) tests object hallucination via
binary yes/no questions: "Is there a <object> in the image?"

Three adversarial splits (COCO val2014 images):
  random      — objects sampled randomly from COCO categories
  popular     — frequently-appearing objects (harder: model may hallucinate)
  adversarial — objects that co-occur often with ground-truth objects (hardest)

Metrics per split:
  Accuracy   — (TP + TN) / total
  Precision  — TP / (TP + FP)
  Recall     — TP / (TP + FN)
  F1         — 2 * P * R / (P + R)
  Yes ratio  — fraction of "yes" predictions (bias indicator; ideal ≈ 0.5)

Usage:
  # Download POPE annotations first (run once)
  python test_pope.py --download_annotations

    # Evaluate using best accuracy checkpoint (default)
    python test_pope.py --checkpoint checkpoints/best_model_accuracy --keep_ratio 0.7

  # Evaluate all 3 splits at kr=0.7
  python test_pope.py --keep_ratio 0.7

  # Baseline (no pruning)
  python test_pope.py --keep_ratio 1.0

  # Token merging
  python test_pope.py --keep_ratio 0.4 --use_merging

  # Sweep across keep ratios
  python test_pope.py --sweep_keep_ratio

  # Custom sweep
  python test_pope.py --sweep_keep_ratio --sweep_ratios 1.0 0.7 0.4 0.2 0.1 --use_merging
"""

import os
import sys
import json
import math
import time
import argparse
import collections
import urllib.request
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

POPE_DATA_DIR  = os.path.join(BASE_DIR, "datasets", "data", "pope")
COCO_IMAGE_DIR = os.path.join(BASE_DIR, "datasets", "data", "vqa_v2", "val2014")

POPE_SPLITS = ["random", "popular", "adversarial"]

# Raw POPE annotation files (COCO flavour) — downloaded once via --download_annotations
POPE_ANNOTATION_URLS = {
    "random":      "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_random.json",
    "popular":     "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_popular.json",
    "adversarial": "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_adversarial.json",
}

POPE_ANNOTATION_PATHS = {
    split: os.path.join(POPE_DATA_DIR, f"coco_pope_{split}.json")
    for split in POPE_SPLITS
}

# Vicuna system prompt (same as test_paper.py)
_VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


# ─────────────────────────────────────────────────────────────────────────────
# Annotation download
# ─────────────────────────────────────────────────────────────────────────────
def download_annotations():
    os.makedirs(POPE_DATA_DIR, exist_ok=True)
    for split, url in POPE_ANNOTATION_URLS.items():
        dest = POPE_ANNOTATION_PATHS[split]
        if os.path.exists(dest):
            print(f"  [skip] {split} already exists: {dest}")
            continue
        print(f"  Downloading {split} → {dest} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  Done: {dest}")
        except Exception as e:
            print(f"  [ERROR] Failed to download {split}: {e}")
            print(f"  Please download manually from: {url}")


def check_annotations_exist() -> bool:
    missing = [s for s in POPE_SPLITS if not os.path.exists(POPE_ANNOTATION_PATHS[s])]
    if missing:
        print(f"\n[ERROR] Missing POPE annotation files for splits: {missing}")
        print(f"  Expected location: {POPE_DATA_DIR}/")
        print("  Run with --download_annotations to fetch them automatically.\n")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────
def build_pope_prompt(question: str) -> str:
    """
    Vicuna-style prompt for binary yes/no questions.
    Appending "Answer with yes or no." stabilises output for pruned models.
    """
    # <image> token marks where visual embeddings are inserted inline,
    # matching the LLaVA-1.5 pretraining format: USER: <image>\n{question}
    return f"{_VICUNA_SYSTEM} USER: <image>\n{question}\nAnswer with yes or no. ASSISTANT:"


# ─────────────────────────────────────────────────────────────────────────────
# POPE dataset
# ─────────────────────────────────────────────────────────────────────────────
class POPEDataset(Dataset):
    """
    Loads a single POPE split (random / popular / adversarial).

    Annotation file format (one JSON object per line OR a JSON array):
      {"image": "COCO_val2014_000000000042.jpg",
       "text":  "Is there a bench in the image?",
       "label": "yes"}

    image_dir must contain COCO val2014 images named COCO_val2014_*.jpg.
    """

    def __init__(
        self,
        annotation_path: str,
        image_dir: str,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = image_dir
        self.samples: List[Dict] = []

        with open(annotation_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        # Support both JSON array and JSONL formats
        if raw.startswith("["):
            entries = json.loads(raw)
        else:
            entries = [json.loads(line) for line in raw.splitlines() if line.strip()]

        for idx, entry in enumerate(entries):
            image_name = entry.get("image", entry.get("image_id", ""))
            # Normalise: some versions store just the integer image_id
            if isinstance(image_name, int):
                image_name = f"COCO_val2014_{image_name:012d}.jpg"
            if not image_name.endswith(".jpg"):
                image_name = image_name + ".jpg"

            image_path = os.path.join(image_dir, image_name)
            if not os.path.exists(image_path):
                continue

            question = str(entry.get("text", entry.get("question", ""))).strip()
            label    = str(entry.get("label", entry.get("answer", ""))).strip().lower()
            if label not in ("yes", "no"):
                continue

            question_id = entry.get("question_id", idx)

            self.samples.append({
                "image_path":  image_path,
                "question":    question,
                "prompt_text": build_pope_prompt(question),
                "label":       label,          # "yes" | "no"
                "question_id": question_id,
            })

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"[POPEDataset] Loaded {len(self.samples)} samples from {os.path.basename(annotation_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "image":       Image.open(s["image_path"]).convert("RGB"),
            "question":    s["question"],
            "prompt_text": s["prompt_text"],
            "label":       s["label"],
            "question_id": s["question_id"],
        }


def pope_collate_fn(batch: List[Dict]) -> Dict:
    return {
        "image":       [x["image"]       for x in batch],
        "question":    [x["question"]    for x in batch],
        "prompt_text": [x["prompt_text"] for x in batch],
        "label":       [x["label"]       for x in batch],
        "question_id": [x["question_id"] for x in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_yes_no(raw_pred: str) -> str:
    """
    Extract 'yes' or 'no' from raw model output.
    Returns 'yes', 'no', or 'other' (counted as wrong for both classes).
    """
    text = normalize_vqa_answer(raw_pred).lower()
    # Check for explicit yes/no tokens
    if text in ("yes", "no"):
        return text
    # Partial match — take the first occurrence
    words = text.split()
    for w in words:
        if w == "yes":
            return "yes"
        if w == "no":
            return "no"
    # Substring fallback
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_pope_metrics(preds: List[str], labels: List[str]) -> Dict:
    """
    preds / labels are lists of 'yes' | 'no' (| 'other').
    Returns accuracy, precision, recall, f1, yes_ratio.
    """
    tp = tn = fp = fn = 0
    yes_pred_count = 0

    for p, l in zip(preds, labels):
        p_bin = p if p in ("yes", "no") else "no"   # 'other' treated as 'no'
        if p == "yes":
            yes_pred_count += 1
        if p_bin == "yes" and l == "yes":
            tp += 1
        elif p_bin == "no" and l == "no":
            tn += 1
        elif p_bin == "yes" and l == "no":
            fp += 1
        else:
            fn += 1

    total = len(preds)
    accuracy  = (tp + tn) / max(1, total) * 100
    precision = tp / max(1, tp + fp) * 100
    recall    = tp / max(1, tp + fn) * 100
    f1        = 2 * precision * recall / max(1e-8, precision + recall)
    yes_ratio = yes_pred_count / max(1, total) * 100

    return {
        "Accuracy":  round(accuracy, 2),
        "Precision": round(precision, 2),
        "Recall":    round(recall, 2),
        "F1":        round(f1, 2),
        "Yes_Ratio": round(yes_ratio, 2),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "total": total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch-size auto-tuning
# ─────────────────────────────────────────────────────────────────────────────
# Empirical VRAM per sample (MB, bfloat16), derived from sweep_val_merge.json measurements
# on RTX 5090 with batch_size=16. Incremental = (peak_VRAM - model_base) / 16.
# model_base ≈ 15,000 MB (LLaVA-1.5-7B + LoRA in bfloat16 + CLIP).
_MB_PER_SAMPLE_BY_KR = {
    1.0: 800,   # (27705 - 15000) / 16 ≈ 794 MB
    0.7: 600,   # (24463 - 15000) / 16 ≈ 591 MB
    0.4: 450,   # measured 346 MB/sample (diag); 390 was too low → caused bs to jump to 32
    0.2: 255,   # (19011 - 15000) / 16 ≈ 250 MB
    0.1: 190,   # (17876 - 15000) / 16 ≈ 180 MB
}
_BS_CLAMP_MIN = 4
_BS_CLAMP_MAX = 32


def estimate_optimal_batch_size(keep_ratio: float, vram_headroom: float = 0.80) -> int:
    """
    Estimate the largest safe batch size given current free VRAM and keep_ratio.
    vram_headroom: fraction of free VRAM to use (leave rest for model weights already
    loaded and PyTorch allocator fragmentation).
    """
    if not torch.cuda.is_available():
        return 16

    # Pick the nearest calibrated keep_ratio
    kr_keys = sorted(_MB_PER_SAMPLE_BY_KR.keys())
    nearest_kr = min(kr_keys, key=lambda k: abs(k - keep_ratio))
    mb_per_sample = _MB_PER_SAMPLE_BY_KR[nearest_kr]

    free_mb = (torch.cuda.get_device_properties(0).total_memory
               - torch.cuda.memory_allocated()) / 1024 ** 2
    usable_mb = free_mb * vram_headroom

    bs = int(usable_mb / mb_per_sample)
    bs = max(_BS_CLAMP_MIN, min(_BS_CLAMP_MAX, bs))
    # Round down to nearest power-of-2 for DataLoader efficiency
    bs = 2 ** int(math.log2(bs))
    return bs


# ─────────────────────────────────────────────────────────────────────────────
# Build model (same as test_paper.py)
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

    model, _ = load_lightweight_checkpoint(model, checkpoint_dir, device)
    model = model.to(device)
    model.eval()
    model.set_keep_ratio(keep_ratio)

    if use_merging:
        print(f"  [Token Merging] enabled")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Eval loop for one POPE split
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_pope_split(
    model: PruningVLM,
    pope_split: str,
    annotation_path: str,
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

    # Resolve auto batch size HERE — model is already loaded so free VRAM is accurate
    if batch_size == 0:
        batch_size = estimate_optimal_batch_size(keep_ratio)
        print(f"  [AutoBS] Post-model-load probe → batch_size={batch_size} (kr={keep_ratio:.2f})")

    dataset = POPEDataset(
        annotation_path=annotation_path,
        image_dir=COCO_IMAGE_DIR,
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=pope_collate_fn,
        drop_last=False,
    )
    print(f"  [POPE/{pope_split}] batch_size={batch_size}  workers={num_workers}  "
          f"n_batches={len(loader)}")

    # Reset peak VRAM counter so each split gets its own accurate measurement
    # (max_memory_allocated is a running high-watermark that never drops unless reset)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    llm_tokenizer = model.llm.tokenizer
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    all_preds:  List[str] = []
    all_labels: List[str] = []
    records:    List[Dict] = []

    kept_tokens_sum  = 0.0
    total_tokens_sum = 0.0
    latency_sum      = 0.0
    other_count      = 0

    # Track current batch size so OOM handler can shrink it mid-run
    current_bs = batch_size
    pending_batches = list(loader)

    for batch_idx, batch in enumerate(pending_batches):
        images       = prepare_images_for_model(batch["image"], device=device)
        questions    = batch["question"]
        prompt_texts = batch["prompt_text"]
        labels       = batch["label"]
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

        # OOM-safe generation: keep halving sub-batch size until it fits
        t0 = time.perf_counter()
        sub_bs = current_bs
        preds, merged_info_mask, merged_info_proj = None, None, None
        while True:
            try:
                if sub_bs >= len(questions):  # whole batch fits
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
                else:  # run in sub-batches of sub_bs
                    preds, merged_info_mask, merged_info_proj = [], None, None
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
                        merged_info_mask = si["hard_mask"] if merged_info_mask is None \
                            else torch.cat([merged_info_mask, si["hard_mask"]], dim=0)
                        merged_info_proj = si["projected_tokens"] if merged_info_proj is None \
                            else torch.cat([merged_info_proj, si["projected_tokens"]], dim=0)
                    info = {"hard_mask": merged_info_mask, "projected_tokens": merged_info_proj}
                    break
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                new_sub_bs = max(1, sub_bs // 2)
                print(f"  [OOM] sub_bs {sub_bs} → {new_sub_bs}, retrying")
                if new_sub_bs == sub_bs:
                    raise  # already at 1, can't shrink further
                sub_bs = new_sub_bs
                current_bs = sub_bs
        latency_sum += time.perf_counter() - t0

        N_total = info["projected_tokens"].shape[1]
        kept    = info["hard_mask"].sum(dim=1).float().mean().item()
        kept_tokens_sum  += kept * len(preds)
        total_tokens_sum += N_total * len(preds)

        for i, (raw_pred, label) in enumerate(zip(preds, labels)):
            pred_yn = extract_yes_no(raw_pred)
            if pred_yn == "other":
                other_count += 1
            all_preds.append(pred_yn)
            all_labels.append(label)
            correct = (pred_yn == label) or (pred_yn == "other" and label == "no")
            records.append({
                "question_id":  qids[i],
                "question":     questions[i],
                "label":        label,
                "raw_pred":     raw_pred,
                "pred_yn":      pred_yn,
                "correct":      int(pred_yn == label),
            })

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(pending_batches):
            n_done = len(all_preds)
            running_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(1, n_done) * 100
            print(f"    [{pope_split}] [{batch_idx+1:>3}/{len(pending_batches)}]  "
                  f"acc={running_acc:.1f}%  n={n_done}  bs={current_bs}")

    metrics = compute_pope_metrics(all_preds, all_labels)
    metrics["other_predictions"] = other_count

    total_n        = len(all_preds)
    total_time_s   = latency_sum
    throughput     = total_n / max(1e-6, total_time_s)
    ms_per_image   = total_time_s / max(1, total_n) * 1000
    kept_ratio_act = kept_tokens_sum / max(1, total_tokens_sum)

    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        peak_vram_mb = 0.0

    result = {
        "pope_split":             pope_split,
        "keep_ratio":             keep_ratio,
        "num_samples":            total_n,
        **metrics,
        "Token_Keep_Ratio_Actual": round(kept_ratio_act, 4),
        "Tokens_Kept_Avg":         round(kept_tokens_sum / max(1, total_n), 1),
        "Tokens_Total":            int(total_tokens_sum / max(1, total_n)),
        "Latency_ms_per_image":    round(ms_per_image, 1),
        "Throughput_img_per_sec":  round(throughput, 2),
        "Peak_VRAM_MB":            round(peak_vram_mb, 1),
    }

    return result, records


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────────────────────
def print_pope_table(results_by_split: Dict[str, Dict], keep_ratio: float, use_merging: bool):
    merge_tag = " + merge" if use_merging else ""
    print(f"\n{'='*65}")
    print(f"  POPE RESULTS  —  keep_ratio={keep_ratio:.2f}{merge_tag}")
    print(f"{'='*65}")
    header = f"  {'Split':<12}  {'Acc':>7}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'YesRatio':>9}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    accs = []
    f1s  = []
    for split in POPE_SPLITS:
        if split not in results_by_split:
            continue
        r = results_by_split[split]
        print(f"  {split:<12}  {r['Accuracy']:>7.2f}  {r['Precision']:>7.2f}  "
              f"{r['Recall']:>7.2f}  {r['F1']:>7.2f}  {r['Yes_Ratio']:>9.2f}")
        accs.append(r["Accuracy"])
        f1s.append(r["F1"])

    if len(accs) == 3:
        print("  " + "─" * (len(header) - 2))
        avg_acc = sum(accs) / 3
        avg_f1  = sum(f1s)  / 3
        print(f"  {'Average':<12}  {avg_acc:>7.2f}  {'':>7}  {'':>7}  {avg_f1:>7.2f}")

    # Efficiency from first available split
    first = next(iter(results_by_split.values()))
    print(f"\n  Tokens: {first['Tokens_Kept_Avg']:.0f}/{first['Tokens_Total']} "
          f"(actual ratio={first['Token_Keep_Ratio_Actual']:.4f})")
    print(f"  Latency: {first['Latency_ms_per_image']:.1f} ms/img  "
          f"Throughput: {first['Throughput_img_per_sec']:.2f} img/s  "
          f"Peak VRAM: {first['Peak_VRAM_MB']:.0f} MB")
    print(f"{'='*65}\n")


def print_sweep_table(sweep_results: List[Dict]):
    """Print a compact sweep table across keep_ratios for all 3 POPE splits."""
    print(f"\n{'='*75}")
    print("  POPE SWEEP TABLE")
    print(f"{'='*75}")
    header = (f"  {'kr':>5}  {'Rand.Acc':>9}  {'Pop.Acc':>8}  {'Adv.Acc':>8}  "
              f"{'Rand.F1':>8}  {'Pop.F1':>7}  {'Adv.F1':>7}  {'ms/img':>7}")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for row in sweep_results:
        kr     = row["keep_ratio"]
        splits = row["by_split"]
        r_acc  = splits.get("random",      {}).get("Accuracy",  float("nan"))
        p_acc  = splits.get("popular",     {}).get("Accuracy",  float("nan"))
        a_acc  = splits.get("adversarial", {}).get("Accuracy",  float("nan"))
        r_f1   = splits.get("random",      {}).get("F1",        float("nan"))
        p_f1   = splits.get("popular",     {}).get("F1",        float("nan"))
        a_f1   = splits.get("adversarial", {}).get("F1",        float("nan"))
        ms     = splits.get("random",      {}).get("Latency_ms_per_image", float("nan"))
        print(f"  {kr:>5.2f}  {r_acc:>9.2f}  {p_acc:>8.2f}  {a_acc:>8.2f}  "
              f"{r_f1:>8.2f}  {p_f1:>7.2f}  {a_f1:>7.2f}  {ms:>7.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation for one keep_ratio
# ─────────────────────────────────────────────────────────────────────────────
def run_pope_eval(args, keep_ratio: float, model: Optional[PruningVLM] = None) -> Dict:
    """
    Evaluate all 3 POPE splits for a given keep_ratio.
    If model is None, it will be built and loaded from checkpoint.
    Returns dict with {'keep_ratio': ..., 'by_split': {'random': {...}, ...}}.
    """
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if model is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print(f"\n{'='*65}")
        print(f"  PruningVLM  —  POPE Evaluation")
        print(f"  keep_ratio={keep_ratio:.2f}  batch_size={args.batch_size}  ckpt={args.checkpoint}")
        print(f"{'='*65}\n")

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
        own_model = True
    else:
        model.set_keep_ratio(keep_ratio)
        own_model = False

    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")

    results_by_split: Dict[str, Dict] = {}
    records_by_split: Dict[str, List] = {}

    for pope_split in POPE_SPLITS:
        ann_path = POPE_ANNOTATION_PATHS[pope_split]
        if not os.path.exists(ann_path):
            print(f"  [skip] {pope_split}: annotation not found at {ann_path}")
            continue

        print(f"\n  Evaluating POPE split: {pope_split}")
        result, records = run_pope_split(
            model=model,
            pope_split=pope_split,
            annotation_path=ann_path,
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
        results_by_split[pope_split] = result
        records_by_split[pope_split] = records

    print_pope_table(results_by_split, keep_ratio=keep_ratio, use_merging=args.use_merging)

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    merge_tag = "_merge" if args.use_merging else ""
    kr_tag    = f"kr{keep_ratio:.2f}"

    summary = {
        "keep_ratio":  keep_ratio,
        "use_merging": args.use_merging,
        "checkpoint":  args.checkpoint,
        "by_split":    results_by_split,
    }
    # Append averaged metrics if all 3 splits ran
    if len(results_by_split) == 3:
        summary["average"] = {
            "Accuracy":  round(sum(r["Accuracy"]  for r in results_by_split.values()) / 3, 2),
            "Precision": round(sum(r["Precision"] for r in results_by_split.values()) / 3, 2),
            "Recall":    round(sum(r["Recall"]    for r in results_by_split.values()) / 3, 2),
            "F1":        round(sum(r["F1"]        for r in results_by_split.values()) / 3, 2),
        }

    summary_path = os.path.join(args.output_dir, f"pope_summary_{kr_tag}{merge_tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved → {summary_path}")

    for pope_split, records in records_by_split.items():
        rec_path = os.path.join(
            args.output_dir, f"pope_records_{pope_split}_{kr_tag}{merge_tag}.jsonl"
        )
        with open(rec_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Per-sample records saved → {args.output_dir}/pope_records_*_{kr_tag}{merge_tag}.jsonl")

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

    # Build model once, update keep_ratio for each iteration
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"\n{'='*65}")
    print(f"  PruningVLM  —  POPE Sweep")
    print(f"  ratios={keep_ratios}  ckpt={args.checkpoint}")
    print(f"{'='*65}\n")

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
        print(f"\n{'─'*55}")
        print(f"  Sweep  keep_ratio = {kr:.2f}")
        print(f"{'─'*55}")
        # batch_size=0 sentinel flows to run_pope_split where it's resolved post-model-load
        summary = run_pope_eval(args, keep_ratio=kr, model=model)
        sweep_results.append({"keep_ratio": kr, "by_split": summary["by_split"]})

    print_sweep_table(sweep_results)

    merge_tag  = "_merge" if args.use_merging else ""
    sweep_path = os.path.join(args.output_dir, f"pope_sweep{merge_tag}.json")
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
        description="POPE evaluation for PruningVLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=os.path.join(CKPT_ROOT, "best_model_accuracy"),
                   help="Path to checkpoint directory")
    p.add_argument("--keep_ratio", type=float, default=0.7)
    p.add_argument("--use_merging", action="store_true",
                   help="Merge dropped tokens into nearest kept token (no retraining)")
    p.add_argument("--sweep_keep_ratio", action="store_true",
                   help="Sweep over keep_ratios [1.0, 0.7, 0.4, 0.2, 0.1]")
    p.add_argument("--sweep_ratios", type=float, nargs="+", default=None,
                   help="Custom keep_ratio list for sweep, e.g. --sweep_ratios 1.0 0.7 0.4 0.1")
    p.add_argument("--batch_size",  type=int, default=0,
                   help="Batch size. 0 = auto-detect based on free VRAM and keep_ratio")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap samples per split (default: all ~500 per split)")
    p.add_argument("--max_new_tokens",  type=int, default=8,
                   help="POPE answers are 'yes'/'no' — 8 tokens is sufficient")
    p.add_argument("--num_beams",       type=int, default=1)
    p.add_argument("--clip_max_length", type=int, default=64)
    p.add_argument("--llm_max_length",  type=int, default=256)
    p.add_argument("--output_dir",      default=OUTPUT_DIR)
    p.add_argument("--projector_hidden_dim", type=int, default=2048)
    p.add_argument("--ia_hidden_dim",        type=int, default=512)
    p.add_argument("--llava15_model_name", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--download_annotations", action="store_true",
                   help="Download POPE annotation files from GitHub and exit")
    p.add_argument("--auto_batch_size", action="store_true",
                   help="Alias for --batch_size 0: probe VRAM to pick the largest safe batch")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.download_annotations:
        download_annotations()
        return

    if not check_annotations_exist():
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # batch_size=0 is the sentinel for auto-detection.
    # The actual estimation happens AFTER model is loaded inside run_pope_split,
    # so free VRAM reflects reality (model weights already allocated).
    if args.auto_batch_size:
        args.batch_size = 0  # ensure sentinel is set

    if args.sweep_keep_ratio:
        run_sweep(args)
    else:
        run_pope_eval(args, keep_ratio=args.keep_ratio)


if __name__ == "__main__":
    main()
