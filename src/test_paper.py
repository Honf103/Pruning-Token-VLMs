"""
test_paper.py  —  Paper-ready evaluation for PruningVLM

Default use:
    Evaluate the current static multi-ratio model at fixed keep ratios or fixed
    token budgets (for example 192 / 128 / 64 tokens).

Optional use:
    Enable adaptive budget and calibrate it to a target average token budget for
    side-by-side comparison against static baselines.

Metrics reported:
    VQA Accuracy (soft)      — official VQAv2 metric (min(#matches/3, 1))
    Exact Match (raw)        — case-insensitive exact string match
    Exact Match (norm)       — after VQA answer normalisation
    Token F1                 — token-level precision/recall harmonic mean
    By answer type           — yes/no / number / other breakdown
    Token compression ratios — #kept / #total visual tokens
    Latency                  — wall-clock per-image inference time (ms)
    Throughput               — images/sec
    Peak GPU VRAM (MB)

Usage:
    # Evaluate best_model_accuracy checkpoint on val2014
    python test_paper.py \
            --checkpoint checkpoints/best_model_accuracy \
            --split val \
            --max_samples 5000

    # Full val set
    python test_paper.py --checkpoint checkpoints/best_model_accuracy --split val

    # Static sweep across keep ratios
    python test_paper.py --checkpoint checkpoints/best_model_accuracy --split val --sweep_keep_ratio

    # No-pruning baseline (keep_ratio=1.0 = LLaVA-1.5 baseline)
    python test_paper.py --checkpoint checkpoints/best_model_accuracy --split val --keep_ratio 1.0

    # Adaptive calibration to fixed target budgets
    python test_paper.py --checkpoint checkpoints/best_model_accuracy --split val \
            --adaptive_budget --target_avg_tokens 192 128 64 --use_merging
"""

import os
import sys
import json
import time
import argparse
import collections
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

# ── Local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.pruning_vlm import PruningVLM
from eval import (
    VQAv2EvalDataset,
    vqa_eval_collate_fn,
    build_model,
    load_lightweight_checkpoint,
    prepare_images_for_model,
    generate_answers,
    normalize_vqa_answer,
    compute_vqa_soft_accuracy,
    compute_normalized_exact_match,
    compute_token_f1,
)
from utils.misc import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(BASE_DIR, "datasets", "data", "vqa_v2")
CKPT_ROOT  = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "eval_outputs_paper")

SPLITS = {
    "val": {
        "questions":    os.path.join(DATA_ROOT, "v2_OpenEnded_mscoco_val2014_questions.json"),
        "annotations":  os.path.join(DATA_ROOT, "v2_mscoco_val2014_annotations.json"),
        "image_root":   os.path.join(DATA_ROOT, "val2014"),
    },
    "train": {
        "questions":    os.path.join(DATA_ROOT, "v2_OpenEnded_mscoco_train2014_questions.json"),
        "annotations":  os.path.join(DATA_ROOT, "v2_mscoco_train2014_annotations.json"),
        "image_root":   os.path.join(DATA_ROOT, "train2014"),
    },
    "test": {
        # test-dev: no ground-truth annotations (submit to EvalAI for official score)
        "questions":    os.path.join(DATA_ROOT, "v2_OpenEnded_mscoco_test-dev2015_questions.json"),
        "annotations":  None,
        "image_root":   os.path.join(DATA_ROOT, "test2015"),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_eval(
    checkpoint_dir: str,
    split: str = "val",
    keep_ratio: float = 0.7,
    batch_size: int = 16,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 16,
    num_beams: int = 3,
    clip_max_length: int = 64,
    llm_max_length: int = 256,
    seed: int = 42,
    output_dir: Optional[str] = None,
    projector_hidden_dim: int = 2048,
    ia_hidden_dim: int = 512,
    llava15_model_name: str = "llava-hf/llava-1.5-7b-hf",
    use_merging: bool = False,
    dynamic_budget_enabled: bool = False,
    dynamic_budget_min_keep_ratio: float = 0.2,
    dynamic_budget_max_keep_ratio: float = 0.7,
) -> Dict:
    set_seed(seed)

    split_cfg = SPLITS[split]
    has_gt    = split_cfg["annotations"] is not None

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"\n{'='*65}")
    print(f"  PruningVLM  —  Paper Evaluation")
    print(f"  split={split}  keep_ratio={keep_ratio:.2f}  ckpt={checkpoint_dir}")
    print(f"{'='*65}\n")

    # ── Build model ──────────────────────────────────────────────────────────
    model = PruningVLM(
        clip_model_name="openai/clip-vit-large-patch14-336",
        llm_model_name=llava15_model_name,   # extract LLM from LLaVA-1.5 cache
        keep_ratio=keep_ratio,
        alpha=0.6,
        learnable_alpha=True,
        remove_cls_token=True,
        llm_torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        gate_temperature=0.2,
        gate_score_scale=5.0,
        gate_threshold_mode="topk",
        llm_use_grad=False,
        projector_hidden_dim=projector_hidden_dim,
        ia_hidden_dim=ia_hidden_dim,
        use_merging=use_merging,
        dynamic_budget_enabled=dynamic_budget_enabled,
        dynamic_budget_min_keep_ratio=dynamic_budget_min_keep_ratio,
        dynamic_budget_max_keep_ratio=dynamic_budget_max_keep_ratio,
    )

    # Freeze encoders (same as training)
    for p in model.vision_encoder.parameters():
        p.requires_grad = False
    for p in model.text_encoder.parameters():
        p.requires_grad = False

    # Attach LoRA skeleton before loading adapter
    from peft import LoraConfig, TaskType, get_peft_model
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

    model, metadata = load_lightweight_checkpoint(model, checkpoint_dir, device)
    model = model.to(device)
    model.eval()

    # Override keep_ratio from checkpoint metadata if available
    if metadata and "keep_ratio" in metadata:
        ckpt_kr = float(metadata["keep_ratio"])
        if keep_ratio != ckpt_kr:
            print(f"[Info] keep_ratio overridden by arg: {ckpt_kr:.2f} → {keep_ratio:.2f}")
    model.set_keep_ratio(keep_ratio)
    if use_merging:
        print(f"  [Token Merging] enabled — dropped tokens merged into nearest kept token")

    # ── Tokenizers ───────────────────────────────────────────────────────────
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")
    llm_tokenizer  = model.llm.tokenizer
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = VQAv2EvalDataset(
        questions_path=split_cfg["questions"],
        annotations_path=split_cfg["annotations"],
        image_root=split_cfg["image_root"],
        max_samples=max_samples,
        shuffle_before_select=False,
        seed=seed,
        require_image=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=vqa_eval_collate_fn,
        drop_last=False,
    )

    # ── Accumulators ─────────────────────────────────────────────────────────
    vqa_acc_sum      = 0.0
    nem_sum          = 0.0    # normalised exact match
    f1_sum           = 0.0
    total_n          = 0

    kept_tokens_sum  = 0.0
    total_tokens_sum = 0.0
    latency_sum      = 0.0   # seconds

    by_answer_type: Dict[str, Dict] = collections.defaultdict(
        lambda: {"vqa_acc": 0.0, "n": 0}
    )
    by_question_type: Dict[str, Dict] = collections.defaultdict(
        lambda: {"vqa_acc": 0.0, "n": 0}
    )

    records        = []   # per-sample results (for JSONL output)
    empty_preds    = 0

    for batch_idx, batch in enumerate(loader):
        images       = prepare_images_for_model(batch["image"], device=device)
        questions    = batch["question"]
        prompt_texts = batch["prompt_text"]
        qids         = batch["question_id"]
        gt_answers   = batch["answers"]          # List[List[str] | None]
        gt_mc        = batch["multiple_choice_answer"]
        ans_types    = batch["answer_type"]
        q_types      = batch["question_type"]

        clip_enc = clip_tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=clip_max_length,
        )
        clip_ids  = clip_enc.input_ids.to(device, non_blocking=True)
        clip_mask = clip_enc.attention_mask.to(device, non_blocking=True)

        prompt_enc = llm_tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=llm_max_length,
        )
        p_ids  = prompt_enc.input_ids.to(device, non_blocking=True)
        p_mask = prompt_enc.attention_mask.to(device, non_blocking=True)

        t0 = time.perf_counter()
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
        latency_sum += time.perf_counter() - t0

        # Release KV cache from generate() immediately — it doesn't free automatically.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Token stats
        N_total = info["projected_tokens"].shape[1]   # total visual tokens
        kept    = info["hard_mask"].sum(dim=1).float().mean().item()
        kept_tokens_sum  += kept * len(preds)
        total_tokens_sum += N_total * len(preds)

        for i, pred in enumerate(preds):
            pred_norm = normalize_vqa_answer(pred)
            if len(pred_norm) == 0:
                empty_preds += 1

            gt_list  = gt_answers[i] if has_gt else None
            gt_mc_i  = gt_mc[i]
            at       = ans_types[i]
            qt       = q_types[i]

            vqa_acc = compute_vqa_soft_accuracy(pred, gt_list) if has_gt else None
            nem     = compute_normalized_exact_match(pred, gt_mc_i) if has_gt else None
            f1      = compute_token_f1(pred, gt_mc_i) if has_gt else None

            if has_gt:
                vqa_acc_sum += vqa_acc
                nem_sum     += nem
                f1_sum      += f1
                if at:
                    by_answer_type[at]["vqa_acc"] += vqa_acc
                    by_answer_type[at]["n"]       += 1
                if qt:
                    key = qt.split(", ", 1)[0]   # coarsen e.g. "what is" → "what"
                    by_question_type[key]["vqa_acc"] += vqa_acc
                    by_question_type[key]["n"]       += 1

            total_n += 1
            records.append({
                "question_id": qids[i],
                "question":    questions[i],
                "prediction":  pred,
                "gt_mc":       gt_mc_i,
                "vqa_acc":     round(vqa_acc, 4) if vqa_acc is not None else None,
                "nem":         round(nem, 4) if nem is not None else None,
                "token_f1":    round(f1, 4) if f1 is not None else None,
                "answer_type": at,
            })

        if (batch_idx + 1) % 50 == 0:
            running_acc = vqa_acc_sum / max(1, total_n)
            print(f"  [{batch_idx+1:>4}/{len(loader)}]  "
                  f"VQA acc={running_acc:.4f}  "
                  f"total={total_n}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = max(1, total_n)

    results = {
        "split":            split,
        "checkpoint":       checkpoint_dir,
        "keep_ratio":       keep_ratio,
        "num_samples":      total_n,
        "empty_predictions": empty_preds,
        "dynamic_budget_enabled": dynamic_budget_enabled,
        "dynamic_budget_min_keep_ratio": round(dynamic_budget_min_keep_ratio, 4),
        "dynamic_budget_max_keep_ratio": round(dynamic_budget_max_keep_ratio, 4),
    }

    if has_gt:
        results.update({
            # ── Primary metric (use this in paper Table 1) ──────────────────
            "VQA_Accuracy":          round(vqa_acc_sum / n * 100, 2),   # %
            # ── Secondary metrics ───────────────────────────────────────────
            "Normalized_EM":         round(nem_sum / n * 100, 2),        # %
            "Token_F1":              round(f1_sum / n * 100, 2),         # %
            # ── Answer-type breakdown ────────────────────────────────────────
            "by_answer_type": {
                k: round(v["vqa_acc"] / max(1, v["n"]) * 100, 2)
                for k, v in sorted(by_answer_type.items())
            },
            # ── Question-type top-10 ─────────────────────────────────────────
            "by_question_type_top10": {
                k: round(v["vqa_acc"] / max(1, v["n"]) * 100, 2)
                for k, v in sorted(
                    by_question_type.items(),
                    key=lambda x: -x[1]["n"],
                )[:10]
            },
        })

    # ── Efficiency metrics ────────────────────────────────────────────────────
    kept_ratio_actual = kept_tokens_sum / max(1, total_tokens_sum)
    total_time_s      = latency_sum
    throughput        = total_n / max(1e-6, total_time_s)
    ms_per_image      = total_time_s / max(1, total_n) * 1000

    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        peak_vram_mb = 0.0

    results.update({
        "Token_Keep_Ratio_Actual": round(kept_ratio_actual, 4),
        "Tokens_Kept_Avg":         round(kept_tokens_sum / max(1, total_n), 1),
        "Tokens_Total":            int(total_tokens_sum / max(1, total_n)),
        "Latency_ms_per_image":    round(ms_per_image, 1),
        "Throughput_img_per_sec":  round(throughput, 2),
        "Peak_VRAM_MB":            round(peak_vram_mb, 1),
    })

    return results, records


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print for paper table
# ─────────────────────────────────────────────────────────────────────────────
def print_paper_table(results: Dict):
    print(f"\n{'='*65}")
    print(f"  RESULTS  —  {results['split'].upper()} split")
    print(f"{'='*65}")
    print(f"  Checkpoint       : {results['checkpoint']}")
    print(f"  Samples evaluated: {results['num_samples']}")
    print(f"  Empty predictions: {results['empty_predictions']}")
    print()

    if "VQA_Accuracy" in results:
        print("  ┌─ ACCURACY ─────────────────────────────────────────────┐")
        print(f"  │  VQA Accuracy (soft)  : {results['VQA_Accuracy']:>6.2f} %              │")
        print(f"  │  Normalized Exact Match: {results['Normalized_EM']:>6.2f} %              │")
        print(f"  │  Token F1             : {results['Token_F1']:>6.2f} %              │")
        print("  └────────────────────────────────────────────────────────┘")
        print()

        print("  ┌─ BY ANSWER TYPE ────────────────────────────────────────┐")
        for atype, acc in results["by_answer_type"].items():
            print(f"  │  {atype:<12s}: {acc:>6.2f} %                             │")
        print("  └────────────────────────────────────────────────────────┘")
        print()

        print("  ┌─ BY QUESTION TYPE (top-10) ─────────────────────────────┐")
        for qtype, acc in results["by_question_type_top10"].items():
            print(f"  │  {qtype:<22s}: {acc:>6.2f} %                │")
        print("  └────────────────────────────────────────────────────────┘")
        print()

    print("  ┌─ EFFICIENCY ────────────────────────────────────────────┐")
    print(f"  │  Keep ratio (target)   : {results['keep_ratio']:.2f}                    │")
    print(f"  │  Keep ratio (actual)   : {results['Token_Keep_Ratio_Actual']:.4f}                  │")
    print(f"  │  Tokens kept / total   : {results['Tokens_Kept_Avg']:.0f} / {results['Tokens_Total']}                 │")
    print(f"  │  Latency               : {results['Latency_ms_per_image']:.1f} ms/image            │")
    print(f"  │  Throughput            : {results['Throughput_img_per_sec']:.2f} img/s               │")
    print(f"  │  Peak VRAM             : {results['Peak_VRAM_MB']:.0f} MB                      │")
    print("  └────────────────────────────────────────────────────────┘\n")


# ─────────────────────────────────────────────────────────────────────────────
# Ablation sweep
# ─────────────────────────────────────────────────────────────────────────────
def run_sweep(args):
    keep_ratios = args.sweep_ratios if args.sweep_ratios else [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    sweep_results = []

    for kr in keep_ratios:
        print(f"\n{'─'*55}")
        print(f"  Sweep  keep_ratio = {kr:.1f}")
        print(f"{'─'*55}")
        results, _ = run_eval(
            checkpoint_dir=args.checkpoint,
            split=args.split,
            keep_ratio=kr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            projector_hidden_dim=args.projector_hidden_dim,
            ia_hidden_dim=args.ia_hidden_dim,
            llava15_model_name=args.llava15_model_name,
            use_merging=args.use_merging,
            dynamic_budget_enabled=args.adaptive_budget,
            dynamic_budget_min_keep_ratio=args.dynamic_budget_min_keep_ratio,
            dynamic_budget_max_keep_ratio=args.dynamic_budget_max_keep_ratio,
        )
        sweep_results.append(results)

    # Print sweep table for paper
    print(f"\n{'='*65}")
    print("  SWEEP TABLE  (copy-paste into paper)")
    print(f"{'='*65}")
    header = f"{'keep_ratio':>12}  {'VQA Acc':>9}  {'Norm EM':>9}  {'Token F1':>9}  {'Kept%':>7}  {'ms/img':>7}"
    print(header)
    print("─" * len(header))
    for r in sweep_results:
        vqa  = r.get("VQA_Accuracy", float("nan"))
        nem  = r.get("Normalized_EM", float("nan"))
        f1   = r.get("Token_F1", float("nan"))
        kept = r["Token_Keep_Ratio_Actual"] * 100
        ms   = r["Latency_ms_per_image"]
        print(f"  {r['keep_ratio']:>10.2f}  {vqa:>9.2f}  {nem:>9.2f}  {f1:>9.2f}  {kept:>6.1f}%  {ms:>7.1f}")

    # Save sweep
    merge_tag = "_merge" if args.use_merging else ""
    sweep_path = os.path.join(OUTPUT_DIR, f"sweep_{args.split}{merge_tag}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n  Sweep saved → {sweep_path}")


def _clamp_budget_range(min_r: float, max_r: float) -> Tuple[float, float]:
    min_r = max(1e-3, min(0.999, float(min_r)))
    max_r = max(1e-3, min(1.0, float(max_r)))
    if max_r <= min_r:
        max_r = min(1.0, min_r + 0.01)
    return min_r, max_r


def calibrate_and_eval_target_tokens(args, target_tokens: int):
    """
    Calibrate adaptive budget range shift so actual avg kept tokens matches target.
    Uses a small calibration subset and then runs full evaluation with calibrated range.
    """
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be > 0, got {target_tokens}")

    low, high = -0.6, 0.6
    best = None
    target_keep_ratio = target_tokens / 576.0

    print(f"\n{'='*65}")
    print(f"  Calibrating adaptive budget for target tokens = {target_tokens}")
    print(f"{'='*65}")

    for it in range(args.calib_max_iters):
        shift = 0.5 * (low + high)
        min_r, max_r = _clamp_budget_range(
            args.dynamic_budget_min_keep_ratio + shift,
            args.dynamic_budget_max_keep_ratio + shift,
        )
        calib_results, _ = run_eval(
            checkpoint_dir=args.checkpoint,
            split=args.split,
            keep_ratio=args.keep_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.calib_samples,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            clip_max_length=args.clip_max_length,
            llm_max_length=args.llm_max_length,
            seed=args.seed,
            output_dir=args.output_dir,
            projector_hidden_dim=args.projector_hidden_dim,
            ia_hidden_dim=args.ia_hidden_dim,
            llava15_model_name=args.llava15_model_name,
            use_merging=args.use_merging,
            dynamic_budget_enabled=True,
            dynamic_budget_min_keep_ratio=min_r,
            dynamic_budget_max_keep_ratio=max_r,
        )
        actual_keep = float(calib_results["Token_Keep_Ratio_Actual"])
        err = abs(actual_keep - target_keep_ratio)
        if best is None or err < best["err"]:
            best = {
                "err": err,
                "shift": shift,
                "min_r": min_r,
                "max_r": max_r,
                "calib_results": calib_results,
            }

        print(
            f"  [calib {it+1}/{args.calib_max_iters}] "
            f"shift={shift:+.4f} range=({min_r:.3f}, {max_r:.3f}) "
            f"actual_keep={actual_keep:.4f} target_keep={target_keep_ratio:.4f}"
        )

        # Monotonic assumption: higher shift -> higher kept ratio.
        if actual_keep > target_keep_ratio:
            high = shift
        else:
            low = shift

    assert best is not None
    print(
        f"  Best calibration: shift={best['shift']:+.4f}, "
        f"range=({best['min_r']:.3f}, {best['max_r']:.3f}), "
        f"calib_tokens={best['calib_results']['Tokens_Kept_Avg']:.1f}"
    )

    final_results, final_records = run_eval(
        checkpoint_dir=args.checkpoint,
        split=args.split,
        keep_ratio=args.keep_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        clip_max_length=args.clip_max_length,
        llm_max_length=args.llm_max_length,
        seed=args.seed,
        output_dir=args.output_dir,
        projector_hidden_dim=args.projector_hidden_dim,
        ia_hidden_dim=args.ia_hidden_dim,
        llava15_model_name=args.llava15_model_name,
        use_merging=args.use_merging,
        dynamic_budget_enabled=True,
        dynamic_budget_min_keep_ratio=best["min_r"],
        dynamic_budget_max_keep_ratio=best["max_r"],
    )
    final_results["target_avg_tokens"] = int(target_tokens)
    final_results["calibrated_shift"] = round(best["shift"], 6)
    final_results["calibrated_min_keep_ratio"] = round(best["min_r"], 6)
    final_results["calibrated_max_keep_ratio"] = round(best["max_r"], 6)

    return final_results, final_records


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Paper-ready evaluation for PruningVLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default=os.path.join(CKPT_ROOT, "best_model_accuracy"),
                   help="Path to checkpoint directory")
    p.add_argument("--split", choices=["val", "train", "test"], default="val")
    p.add_argument("--keep_ratio", type=float, default=0.7,
                   help="Visual token keep ratio (0–1). Use 1.0 for no-pruning baseline.")
    p.add_argument("--sweep_keep_ratio", action="store_true",
                   help="Run accuracy-vs-efficiency sweep across keep ratios [1.0, 0.9, … 0.5]")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap number of eval samples (useful for quick sanity-check)")
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--num_beams",      type=int, default=1,
                   help="1=greedy (saves 3x KV cache vs beam=3, ~same accuracy for VQA)")
    p.add_argument("--clip_max_length", type=int, default=64)
    p.add_argument("--llm_max_length",  type=int, default=256)
    p.add_argument("--output_dir",  default=OUTPUT_DIR)
    p.add_argument("--projector_hidden_dim", type=int, default=2048)
    p.add_argument("--ia_hidden_dim",        type=int, default=512)
    p.add_argument("--llava15_model_name", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_merging", action="store_true",
                   help="Merge dropped tokens into nearest kept token (cosine sim). "
                        "Pure inference trick — no retraining needed.")
    p.add_argument("--sweep_ratios", type=float, nargs="+", default=None,
                   help="Custom keep_ratio list for sweep, e.g. --sweep_ratios 0.5 0.3 0.1 0.05")
    p.add_argument("--adaptive_budget", action="store_true",
                   help="Enable dynamic per-sample keep-ratio prediction during evaluation.")
    p.add_argument("--dynamic_budget_min_keep_ratio", type=float, default=0.2)
    p.add_argument("--dynamic_budget_max_keep_ratio", type=float, default=0.7)
    p.add_argument("--target_avg_tokens", type=int, nargs="+", default=None,
                   help="Calibrate adaptive budget to hit target average kept tokens (e.g. 192 128 64).")
    p.add_argument("--calib_samples", type=int, default=512,
                   help="Number of samples for adaptive calibration search.")
    p.add_argument("--calib_max_iters", type=int, default=8,
                   help="Binary-search iterations for adaptive calibration.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep_keep_ratio:
        run_sweep(args)
        return

    if args.target_avg_tokens:
        if not args.adaptive_budget:
            print("[Info] --target_avg_tokens implies --adaptive_budget; enabling it.")
            args.adaptive_budget = True

        all_results = []
        for tgt in args.target_avg_tokens:
            results, records = calibrate_and_eval_target_tokens(args, int(tgt))
            print_paper_table(results)

            tag = f"{args.split}_targetTok{int(tgt)}" + ("_merge" if args.use_merging else "")
            summary_path = os.path.join(args.output_dir, f"summary_{tag}.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            records_path = os.path.join(args.output_dir, f"records_{tag}.jsonl")
            with open(records_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  Summary saved → {summary_path}")
            print(f"  Per-sample records → {records_path}")
            all_results.append(results)

        compact_path = os.path.join(args.output_dir, f"adaptive_target_tokens_{args.split}.json")
        with open(compact_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Target-token sweep summary → {compact_path}")
        return

    results, records = run_eval(
        checkpoint_dir=args.checkpoint,
        split=args.split,
        keep_ratio=args.keep_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        clip_max_length=args.clip_max_length,
        llm_max_length=args.llm_max_length,
        seed=args.seed,
        output_dir=args.output_dir,
        projector_hidden_dim=args.projector_hidden_dim,
        ia_hidden_dim=args.ia_hidden_dim,
        llava15_model_name=args.llava15_model_name,
        use_merging=args.use_merging,
        dynamic_budget_enabled=args.adaptive_budget,
        dynamic_budget_min_keep_ratio=args.dynamic_budget_min_keep_ratio,
        dynamic_budget_max_keep_ratio=args.dynamic_budget_max_keep_ratio,
    )

    print_paper_table(results)

    # ── Save outputs ──────────────────────────────────────────────────────────
    tag = f"{args.split}_kr{args.keep_ratio:.2f}" + ("_merge" if args.use_merging else "")

    summary_path = os.path.join(args.output_dir, f"summary_{tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved → {summary_path}")

    records_path = os.path.join(args.output_dir, f"records_{tag}.jsonl")
    with open(records_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Per-sample records → {records_path}")

    # ── Test-set submission format (EvalAI) ───────────────────────────────────
    if args.split == "test":
        submission = [
            {"question_id": r["question_id"], "answer": r["prediction"]}
            for r in records
        ]
        sub_path = os.path.join(args.output_dir, "vqa_test_submission.json")
        with open(sub_path, "w", encoding="utf-8") as f:
            json.dump(submission, f)
        print(f"  EvalAI submission → {sub_path}")


if __name__ == "__main__":
    main()
