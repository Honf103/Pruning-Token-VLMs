import os
import json
import random
import re
from typing import List, Dict, Any, Optional

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    CLIPTokenizer,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, TaskType, get_peft_model

from models.pruning_vlm import PruningVLM
from utils.misc import set_seed
from eval import VQAv2EvalDataset, vqa_eval_collate_fn, generate_answers


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "datasets", "data", "vqa_v2")
CHECKPOINT_ROOT = os.path.join(BASE_DIR, "checkpoints")


def ensure_paths_exist(paths: List[str], header: str):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        msg = [header, "Missing required paths:"]
        msg.extend([f"  - {p}" for p in missing])
        raise FileNotFoundError("\n".join(msg))


CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hes": "he's",
    "im": "i'm", "isnt": "isn't", "itll": "it'll", "ive": "i've",
    "shouldve": "should've", "shouldnt": "shouldn't", "thats": "that's",
    "theres": "there's", "theyre": "they're", "theyve": "they've", "wasnt": "wasn't",
    "werent": "weren't", "whats": "what's", "wheres": "where's", "whos": "who's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "youre": "you're",
    "youve": "you've",
}

MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

ARTICLES = {"a", "an", "the"}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
PUNCT = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
    "_", "-", ">", "<", "@", "`", ",", "?", "!"
]


# ============================================================
# Utils
# ============================================================
def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True
    module.train()


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_trainable_parameters(model: nn.Module):
    total, trainable = count_params(model)
    ratio = 100.0 * trainable / total if total > 0 else 0.0
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Trainable ratio  : {ratio:.4f}%")


def normalize_vqa_answer(s: str) -> str:
    s = s.lower().replace("\n", " ").replace("\t", " ").strip()
    s = COMMA_STRIP.sub(r"\1\3", s)
    s = PERIOD_STRIP.sub("", s)

    for p in PUNCT:
        if (p + " " in s) or (" " + p in s) or (re.search(COMMA_STRIP, s) is not None):
            s = s.replace(p, "")
        else:
            s = s.replace(p, " ")

    words = []
    for word in s.split():
        word = MANUAL_MAP.get(word, word)
        word = CONTRACTIONS.get(word, word)
        if word not in ARTICLES:
            words.append(word)

    return re.sub(r"\s+", " ", " ".join(words)).strip()


def compute_vqa_soft_accuracy(pred: str, gt_answers: List[str]) -> float:
    pred_norm = normalize_vqa_answer(pred)
    gt_norm = [normalize_vqa_answer(x) for x in gt_answers]
    count = sum(1 for x in gt_norm if x == pred_norm)
    return min(1.0, count / 3.0)


def compute_budget_loss(
    soft_gates: torch.Tensor,
    target_keep_ratio,
):
    keep_per_sample = soft_gates.mean(dim=1)
    if isinstance(target_keep_ratio, torch.Tensor):
        target = target_keep_ratio.to(device=keep_per_sample.device, dtype=keep_per_sample.dtype)
    else:
        target = torch.full_like(keep_per_sample, float(target_keep_ratio))
    return F.mse_loss(keep_per_sample, target)



def linear_temp_schedule(epoch, num_epochs, start_temp=1.5, end_temp=0.3):
    if num_epochs <= 1:
        return end_temp
    t = epoch / float(num_epochs - 1)
    return start_temp + (end_temp - start_temp) * t


def linear_weight_warmup(epoch, num_epochs, warmup_ratio=0.4):
    """Linearly ramp a loss weight from 0 -> 1 during early training."""
    if num_epochs <= 1:
        return 1.0
    t = epoch / float(num_epochs - 1)
    if warmup_ratio <= 0:
        return 1.0
    return min(1.0, t / warmup_ratio)


def step_pruning_mode(
    step: int,
    total_steps: int,
    soft_stage_frac: float = 0.33,
    ste_stage_frac: float = 0.33,
) -> str:
    """
    Step-based pruning curriculum for Stage 2.
    Partitions total_steps into soft → STE → structural phases so the
    full curriculum runs within a single training epoch.
    """
    if total_steps <= 1:
        return "structural"
    t = step / float(total_steps - 1)
    if t < soft_stage_frac:
        return "soft"
    if t < soft_stage_frac + ste_stage_frac:
        return "ste"
    return "structural"


def step_keep_ratio(
    step: int,
    total_steps: int,
    start_ratio: float = 0.95,
    end_ratio: float = 0.7,
) -> float:
    """Linearly anneal keep_ratio across all training steps."""
    if total_steps <= 1:
        return end_ratio
    t = step / float(total_steps - 1)
    return start_ratio + (end_ratio - start_ratio) * t


def _normalize_ratio_values(ratios: Optional[List[float]]) -> List[float]:
    if not ratios:
        return []
    out = []
    for r in ratios:
        rr = float(r)
        if rr <= 0.0 or rr > 1.0:
            continue
        out.append(rr)
    # unique + sorted for stable logging / reproducibility
    return sorted(set(out))


def _sample_multi_ratio(
    ratio_values: List[float],
    ratio_probs: Optional[List[float]],
    progress: float,
    low_focus_start_ratio: float,
    low_focus_power: float,
) -> float:
    if len(ratio_values) == 1:
        return ratio_values[0]

    if ratio_probs is not None and len(ratio_probs) == len(ratio_values):
        weights = [max(0.0, float(w)) for w in ratio_probs]
        if sum(weights) <= 0.0:
            weights = [1.0 for _ in ratio_values]
    else:
        weights = [1.0 for _ in ratio_values]

    # Late training: bias to lower ratios so the scorer specializes in extreme pruning.
    if progress >= low_focus_start_ratio:
        adjusted = []
        for r, w in zip(ratio_values, weights):
            adjusted.append(w * (1.0 / max(r, 1e-4)) ** low_focus_power)
        weights = adjusted

    return random.choices(ratio_values, weights=weights, k=1)[0]


def ratio_adaptive_loss_scales(
    keep_ratio: float,
    low_threshold: float = 0.3,
    mid_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Increase regularization pressure for low keep-ratio training steps.
    This stabilizes extreme pruning where the student loses more visual context.
    """
    r = float(keep_ratio)
    if r < low_threshold:
        return {"budget": 1.5, "distill": 2.0}
    if r < mid_threshold:
        return {"budget": 1.2, "distill": 1.4}
    return {"budget": 1.0, "distill": 1.0}


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def build_question_adaptive_keep_ratio_targets(
    questions: List[str],
    answers: Optional[List[str]],
    min_keep_ratio: float,
    max_keep_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build lightweight pseudo keep-ratio targets from question/answer cues.
    This gives the dynamic-budget head a supervised signal without extra labels.
    """
    min_r = float(min_keep_ratio)
    max_r = float(max_keep_ratio)
    if not (0.0 < min_r <= max_r <= 1.0):
        raise ValueError(
            f"Expected 0 < min_keep_ratio <= max_keep_ratio <= 1, got {min_r}, {max_r}"
        )

    out = []
    for i, q in enumerate(questions):
        qn = str(q).strip().lower()
        a = "" if answers is None else str(answers[i]).strip().lower()

        # Base ratio around the center of target budget range.
        ratio = 0.5 * (min_r + max_r)
        q_tokens = qn.split()

        # Cheap binary yes/no questions often tolerate tighter budgets.
        if _contains_any(qn, ["is there", "are there", "do you", "does the", "can you", "is the"]) and a in {"yes", "no"}:
            ratio -= 0.10

        # Counting / reading / reasoning usually need more visual evidence.
        if _contains_any(qn, ["how many", "number of", "count"]):
            ratio += 0.18
        if _contains_any(qn, ["read", "text", "word", "letter", "sign", "logo", "written"]):
            ratio += 0.20
        if _contains_any(qn, ["why", "reason", "explain", "because", "how does", "how do"]):
            ratio += 0.14
        if _contains_any(qn, ["where", "left", "right", "top", "bottom", "position", "located"]):
            ratio += 0.07
        if _contains_any(qn, ["what color", "color of", "colour"]):
            ratio += 0.08

        # Longer instructions tend to be compositional.
        if len(q_tokens) >= 12:
            ratio += 0.05
        if len(q_tokens) >= 20:
            ratio += 0.05

        out.append(max(min_r, min(max_r, ratio)))

    return torch.tensor(out, device=device, dtype=dtype)


def load_llava15_pretrained_weights(
    model: "PruningVLM",
    llava_model_name: str = "llava-hf/llava-1.5-7b-hf",
    load_llm: bool = True,
    load_projector: bool = True,
):
    """
    Initialize PruningVLM từ pretrained LLaVA-1.5 checkpoint.

    Projector architecture now matches LLaVA-1.5 exactly:
        Linear(1024→4096) → GELU → Linear(4096→4096)
    Weights are copied directly via projector.load_from_llava().

    LLM (Vicuna-7B): load exact — LLaVA-1.5 instruction-tunes Vicuna further on
    image-text data, so these weights are strictly better than base Vicuna for VQA.
    """
    import gc
    try:
        from transformers import LlavaForConditionalGeneration
    except ImportError:
        raise ImportError(
            "transformers >= 4.37 required for LlavaForConditionalGeneration. "
            "Run: pip install -U transformers"
        )

    print(f"\n{'='*60}")
    print(f"Loading LLaVA-1.5 pretrained weights: {llava_model_name}")
    print("  First run downloads ~14GB  →  subsequent runs use HF cache.")
    print(f"{'='*60}")

    llava = LlavaForConditionalGeneration.from_pretrained(
        llava_model_name,
        torch_dtype=torch.float32,   # float32 for safe weight copy
        device_map="cpu",
    )

    # ── 1. LLM weights ────────────────────────────────────────────────────
    if load_llm:
        llm_src = llava.language_model.state_dict()
        llm_dst = model.llm.model.state_dict()
        compatible = {
            k: v.to(llm_dst[k].dtype)
            for k, v in llm_src.items()
            if k in llm_dst and llm_dst[k].shape == v.shape
        }
        missing = [k for k in llm_dst if k not in compatible]
        model.llm.model.load_state_dict(compatible, strict=False)
        print(f"  LLM : loaded {len(compatible)}/{len(llm_dst)} layers"
              + (f"  (skipped {len(missing)} — shape mismatch)" if missing else ""))

    # ── 2. Projector weights (direct copy — architectures match exactly) ───
    if load_projector:
        model.projector.load_from_llava(llava.multi_modal_projector)

    del llava
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{'='*60}\n")


def attach_lora_to_llm(
    llm_wrapper,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    if not hasattr(llm_wrapper, "model"):
        raise ValueError("Expected model.llm to have attribute `.model` for the HF causal LM.")

    base_llm = llm_wrapper.model

    for p in base_llm.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    peft_llm = get_peft_model(base_llm, lora_config)
    llm_wrapper.model = peft_llm
    return llm_wrapper


def save_lora_and_non_llm_trainables(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Only save LoRA adapter if LoRA was actually attached (PeftModel).
    # When freeze_llm=True the model is a plain LlamaForCausalLM — saving it
    # would dump the full ~14 GB 7B weights unnecessarily.
    try:
        from peft import PeftModel as _PeftModel
        _is_peft = isinstance(getattr(model.llm, "model", None), _PeftModel)
    except ImportError:
        _is_peft = False
    if _is_peft:
        model.llm.model.save_pretrained(os.path.join(save_dir, "llm_lora_adapter"))

    # Save all non-infrastructure params (projector + scorer).
    # Projector is always saved regardless of requires_grad — it may be frozen
    # during freeze_llm=True training but is still required for inference.
    # vision_encoder, text_encoder, and llm weights are excluded (too large /
    # sourced from pretrained HF checkpoints at eval time).
    _infra_prefixes = ("vision_encoder.", "text_encoder.", "llm.")
    trainable_non_llm_state = {}
    for name, param in model.named_parameters():
        if not any(name.startswith(p) for p in _infra_prefixes):
            trainable_non_llm_state[name] = param.detach().cpu()

    torch.save(
        {"trainable_non_llm_state_dict": trainable_non_llm_state},
        os.path.join(save_dir, "non_llm_trainables.pt"),
    )


def has_non_finite_gradients(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return True
    return False


# ============================================================
# VQAv2 Dataset
# ============================================================
class VQAv2Dataset(Dataset):
    """
    VQAv2 short-answer training.
    Mỗi sample trả ra:
    - question
    - answer
    - prompt_text
    """

    def __init__(
        self,
        questions_path: str,
        annotations_path: str,
        image_root: str,
        max_samples: Optional[int] = None,
        shuffle_before_select: bool = True,
        seed: int = 42,
        require_image: bool = True,
    ):
        super().__init__()
        self.image_root = image_root
        self.require_image = require_image

        with open(questions_path, "r", encoding="utf-8") as f:
            q_data = json.load(f)

        with open(annotations_path, "r", encoding="utf-8") as f:
            a_data = json.load(f)

        questions = q_data["questions"]
        annotations = a_data["annotations"]

        ann_map = {}
        for ann in annotations:
            qid = ann["question_id"]

            mc_answer = str(ann.get("multiple_choice_answer", "")).strip().lower()
            raw_answers = [
                str(x.get("answer", "")).strip().lower()
                for x in ann.get("answers", [])
                if str(x.get("answer", "")).strip()
            ]

            candidate_answers = []
            seen = set()
            for answer in raw_answers + ([mc_answer] if mc_answer else []):
                norm = normalize_vqa_answer(answer)
                if len(norm) == 0 or norm in seen:
                    continue
                candidate_answers.append(answer)
                seen.add(norm)

            if len(candidate_answers) == 0:
                continue

            ann_map[qid] = {
                "answer": mc_answer if mc_answer else candidate_answers[0],
                "candidate_answers": candidate_answers,
                "answers": ann.get("answers", []),
            }

        samples = []
        for q in questions:
            qid = q["question_id"]
            image_id = q["image_id"]
            question = str(q["question"]).strip()

            if qid not in ann_map:
                continue

            image_name = f"COCO_train2014_{image_id:012d}.jpg"
            image_path = os.path.join(image_root, image_name)

            if require_image and not os.path.exists(image_path):
                continue

            answer = ann_map[qid]["answer"]
            candidate_answers = ann_map[qid]["candidate_answers"]

            prompt_text = (
                    "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                    f"USER: <image>\n{question}\nASSISTANT:"
                )

            samples.append(
                {
                    "image_path": image_path,
                    "question": question,
                    "answer": answer,
                    "candidate_answers": candidate_answers,
                    "prompt_text": prompt_text,
                    "question_id": qid,
                    "image_id": image_id,
                }
            )

        if shuffle_before_select:
            rng = random.Random(seed)
            rng.shuffle(samples)

        if max_samples is not None:
            samples = samples[:max_samples]

        self.samples = samples
        print(f"[VQAv2Dataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        candidate_answers = item.get("candidate_answers", [])
        # Sử dụng canonical (multiple_choice) answer để tránh label noise từ random annotator.
        # Random choice tạo mâu thuẫn nhãn cho cùng question qua các epoch khác nhau.
        sampled_answer = item["answer"]

        return {
            "image": image,
            "question": item["question"],
            "answer": sampled_answer,
            "canonical_answer": item["answer"],
            "candidate_answers": candidate_answers,
            "prompt_text": item["prompt_text"],
            "question_id": item["question_id"],
            "image_id": item["image_id"],
        }


def vqa_collate_fn(batch: List[Dict[str, Any]]):
    return {
        "image": [x["image"] for x in batch],
        "question": [x["question"] for x in batch],
        "answer": [x["answer"] for x in batch],
        "canonical_answer": [x["canonical_answer"] for x in batch],
        "candidate_answers": [x["candidate_answers"] for x in batch],
        "prompt_text": [x["prompt_text"] for x in batch],
        "question_id": [x["question_id"] for x in batch],
        "image_id": [x["image_id"] for x in batch],
    }


@torch.no_grad()
def validate_vqa_generation(
    model: PruningVLM,
    clip_tokenizer,
    llm_tokenizer,
    device: str,
    keep_ratio: float,
    batch_size: int = 4,
    num_workers: int = 4,
    max_samples: int = 1000,
    image_size: int = 336,
    llm_max_length: int = 256,
    clip_max_length: int = 64,
    seed: int = 42,
):
    prev_training = model.training
    prev_keep_ratio = model.keep_ratio

    val_questions_path = os.path.join(DATA_ROOT, "v2_OpenEnded_mscoco_val2014_questions.json")
    val_annotations_path = os.path.join(DATA_ROOT, "v2_mscoco_val2014_annotations.json")
    val_image_root = os.path.join(DATA_ROOT, "val2014")
    ensure_paths_exist(
        [val_questions_path, val_annotations_path, val_image_root],
        "Validation dataset paths are not ready.",
    )

    dataset = VQAv2EvalDataset(
        questions_path=val_questions_path,
        annotations_path=val_annotations_path,
        image_root=val_image_root,
        max_samples=max_samples,
        shuffle_before_select=False,
        seed=seed,
        require_image=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=vqa_eval_collate_fn,
        drop_last=False,
    )

    model.eval()
    model.set_keep_ratio(keep_ratio)

    total_vqa_soft_acc = 0.0
    total_samples = 0
    empty_prediction_count = 0

    for batch in dataloader:
        images = prepare_images_for_model(batch["image"], device=device, image_size=image_size)
        questions = batch["question"]
        prompt_texts = batch["prompt_text"]
        gt_answers_list = batch["answers"]

        clip_tokens = clip_tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=clip_max_length,
        )
        clip_input_ids = clip_tokens.input_ids.to(device, non_blocking=True)
        clip_attention_mask = clip_tokens.attention_mask.to(device, non_blocking=True)

        prompt_tokens = llm_tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=llm_max_length,
        )
        prompt_input_ids = prompt_tokens.input_ids.to(device, non_blocking=True)
        prompt_attention_mask = prompt_tokens.attention_mask.to(device, non_blocking=True)

        gen_texts, _ = generate_answers(
            model=model,
            images=images,
            clip_input_ids=clip_input_ids,
            clip_attention_mask=clip_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            max_new_tokens=16,
            num_beams=3,
            use_hard_pruning=True,
        )

        for pred, gt_answers in zip(gen_texts, gt_answers_list):
            pred = str(pred).strip()
            if len(pred) == 0:
                empty_prediction_count += 1
            total_vqa_soft_acc += compute_vqa_soft_accuracy(pred, gt_answers)
            total_samples += 1

    model.set_keep_ratio(prev_keep_ratio)
    if prev_training:
        model.train()
        model.vision_encoder.eval()
        model.text_encoder.eval()

    return {
        "vqa_soft_accuracy": total_vqa_soft_acc / max(1, total_samples),
        "empty_prediction_rate": empty_prediction_count / max(1, total_samples),
        "num_samples": total_samples,
    }


# ============================================================
# Input / label prep
# ============================================================
def build_image_transform(image_size=336):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def prepare_images_for_model(images, device, image_size=336):
    """
    Chỉ resize + to tensor.
    Không normalize ở đây vì vision_encoder của bạn đã normalize bên trong.
    """
    tfm = build_image_transform(image_size=image_size)
    image_tensors = [tfm(img) for img in images]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors.to(device, non_blocking=True)


def build_llm_batch_from_prompt_answer(
    prompt_texts: List[str],
    answers: List[str],
    tokenizer,
    max_length: int = 256,
):
    """
    Build input_ids / attention_mask / labels an toàn:
    - tokenize prompt riêng
    - tokenize answer riêng, có prepend space và append eos
    - concat ở level token ids
    - labels = -100 cho prompt, giữ answer ids cho vùng answer
    """
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id must not be None.")

    if tokenizer.eos_token is None and tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have eos token or eos_token_id.")

    eos_text = tokenizer.eos_token if tokenizer.eos_token is not None else ""

    batch_input_ids = []
    batch_labels = []

    num_truncated = 0
    num_empty_answer_after_trunc = 0

    for prompt_text, answer in zip(prompt_texts, answers):
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=True,   # BOS <s> required by Vicuna
            truncation=False,
        )["input_ids"]

        answer_text = " " + str(answer).strip().lower()
        if eos_text:
            answer_text = answer_text + eos_text

        answer_ids = tokenizer(
            answer_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        if len(answer_ids) == 0:
            # trường hợp hiếm, ép ít nhất 1 eos nếu có
            if tokenizer.eos_token_id is not None:
                answer_ids = [tokenizer.eos_token_id]

        # Chừa chỗ cho ít nhất 1 token answer
        max_prompt_len = max_length - 1
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]
            num_truncated += 1

        remaining = max_length - len(prompt_ids)
        truncated_answer_ids = answer_ids[:remaining]

        if len(truncated_answer_ids) < len(answer_ids):
            num_truncated += 1

        if len(truncated_answer_ids) == 0:
            num_empty_answer_after_trunc += 1
            continue

        input_ids = prompt_ids + truncated_answer_ids
        labels = ([-100] * len(prompt_ids)) + truncated_answer_ids

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)

    if len(batch_input_ids) == 0:
        return None

    pad_id = tokenizer.pad_token_id
    max_seq_len = max(len(x) for x in batch_input_ids)

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for input_ids, labels in zip(batch_input_ids, batch_labels):
        pad_len = max_seq_len - len(input_ids)
        padded_input_ids.append(input_ids + [pad_id] * pad_len)
        padded_attention_mask.append([1] * len(input_ids) + [0] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)

    batch = {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "num_truncated": num_truncated,
        "num_empty_answer_after_trunc": num_empty_answer_after_trunc,
        "num_valid_samples": len(batch_input_ids),
    }
    return batch


# ============================================================
# Stage 1 — Feature Alignment
# ============================================================
def train_stage1(
    batch_size: int = 4,
    num_workers: int = 4,
    grad_accum_steps: int = 8,
    max_grad_norm: float = 1.0,
    projector_lr: float = 2e-4,   # lower when starting from LLaVA-1.5 weights
    weight_decay: float = 0.01,
    max_samples: Optional[int] = None,
    llm_max_length: int = 256,
    clip_max_length: int = 64,
    val_batch_size: int = 4,
    val_num_workers: int = 2,
    val_max_samples: int = 500,
    projector_hidden_dim: int = 2048,
    ia_hidden_dim: int = 512,
    enable_tf32: bool = True,
    resume_from: Optional[str] = None,
    save_dir: Optional[str] = None,
    llava15_init: bool = True,
    llava15_model_name: str = "llava-hf/llava-1.5-7b-hf",
    train_budget_head_only: bool = False,
):
    """
    Stage 1 — Feature Alignment (mirrors LLaVA-1.5 stage 1).

    Only the MLP projector is trained; everything else (CLIP encoders,
    LLM, all scoring modules) is frozen.  No token pruning (keep_ratio=1).
    Loss = LM CE only.

    Goal: teach the projector to map CLIP ViT-L/14 features into the
    LLM's input-embedding space before any VQA or pruning supervision.
    """
    if save_dir is None:
        save_dir = os.path.join(CHECKPOINT_ROOT, "stage1")
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if torch.cuda.is_available() and enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # When llava15_init=True, load the LLM directly from the LLaVA-1.5 checkpoint
    # inside LLaVALM.__init__ — this extracts only language_model and frees the rest,
    # avoiding a separate ~13 GB Vicuna download.
    _llm_source = llava15_model_name if llava15_init else "lmsys/vicuna-7b-v1.5"

    model = PruningVLM(
        clip_model_name="openai/clip-vit-large-patch14-336",
        llm_model_name=_llm_source,
        keep_ratio=1.0,
        llm_torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        gate_temperature=1.0,
        gate_score_scale=5.0,
        gate_threshold_mode="topk",
        llm_use_grad=False,
        llm_scale_visual_prefix=False,
        llm_visual_prefix_scale=1.0,
        projector_hidden_dim=projector_hidden_dim,
        ia_hidden_dim=ia_hidden_dim,
    ).to(device)

    # ── Initialize projector from LLaVA-1.5 pretrained weights ──────────
    # LLM was already loaded from LLaVA-1.5 above (load_llm=False avoids reloading).
    if llava15_init:
        load_llava15_pretrained_weights(
            model,
            llava_model_name=llava15_model_name,
            load_llm=False,
            load_projector=True,
        )
    else:
        print("[Stage1] llava15_init=False — projector trains from random init.")

    # ── Only projector is trainable ──────────────────────────────────────
    freeze_module(model.vision_encoder)
    freeze_module(model.text_encoder)
    freeze_module(model.cls_scorer)
    freeze_module(model.text_importance)
    freeze_module(model.instruction_aware)
    freeze_module(model.score_fusion)
    freeze_module(model.token_pruner)
    for p in model.llm.model.parameters():   # LLM frozen — no LoRA in stage 1
        p.requires_grad = False
    unfreeze_module(model.projector)
    if train_budget_head_only:
        # Keep pretrained LLaVA projector mapping fixed and warm up only budget head.
        for p in model.projector.proj.parameters():
            p.requires_grad = False
        for p in model.projector.budget_visual_proj.parameters():
            p.requires_grad = True
        if model.projector.budget_text_proj is not None:
            for p in model.projector.budget_text_proj.parameters():
                p.requires_grad = True
        for p in model.projector.budget_head.parameters():
            p.requires_grad = True
        print("[Stage1] Training budget head only (projector.proj frozen).")

    if resume_from is not None:
        _path = os.path.join(resume_from, "non_llm_trainables.pt")
        if os.path.exists(_path):
            _ckpt = torch.load(_path, map_location=device)
            _state = _ckpt["trainable_non_llm_state_dict"]
            _model_state = model.state_dict()
            _compatible = {
                k: v for k, v in _state.items()
                if k in _model_state and _model_state[k].shape == v.shape
            }
            model.load_state_dict(_compatible, strict=False)
            print(f"[Stage1 Resume] Loaded {len(_compatible)} keys from {_path}")

    print_trainable_parameters(model)

    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")
    llm_tokenizer = model.llm.tokenizer
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    train_questions_path = os.path.join(DATA_ROOT, "v2_OpenEnded_mscoco_train2014_questions.json")
    train_annotations_path = os.path.join(DATA_ROOT, "v2_mscoco_train2014_annotations.json")
    train_image_root = os.path.join(DATA_ROOT, "train2014")
    ensure_paths_exist(
        [train_questions_path, train_annotations_path, train_image_root],
        "Training dataset paths are not ready.",
    )

    dataset = VQAv2Dataset(
        questions_path=train_questions_path,
        annotations_path=train_annotations_path,
        image_root=train_image_root,
        max_samples=max_samples,
        shuffle_before_select=True,
        seed=42,
        require_image=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=vqa_collate_fn,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.projector.parameters(),
        lr=projector_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    total_update_steps = max(1, (len(dataloader) + grad_accum_steps - 1) // grad_accum_steps)
    warmup_steps = max(10, int(0.03 * total_update_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(torch.cuda.is_available() and not use_bf16))

    model.train()
    model.vision_encoder.eval()
    model.text_encoder.eval()
    model.cls_scorer.eval()
    model.text_importance.eval()
    model.instruction_aware.eval()
    model.score_fusion.eval()
    model.llm.model.eval()
    model.set_keep_ratio(1.0)
    model.set_dynamic_budget(False)

    total_loss = 0.0
    num_effective_steps = 0
    skipped = 0
    optimizer.zero_grad(set_to_none=True)

    print("=" * 60)
    print("Stage 1 — Feature Alignment")
    print(f"  dataset : {len(dataset)} samples   steps : {len(dataloader)}")
    print(f"  projector_lr={projector_lr}   grad_accum={grad_accum_steps}   bf16={use_bf16}")
    print("=" * 60)

    for step, batch in enumerate(dataloader):
        images       = prepare_images_for_model(batch["image"], device=device, image_size=336)
        questions    = batch["question"]
        answers      = batch["answer"]
        prompt_texts = batch["prompt_text"]

        clip_tokens = clip_tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=clip_max_length,
        )
        clip_input_ids      = clip_tokens.input_ids.to(device, non_blocking=True)
        clip_attention_mask = clip_tokens.attention_mask.to(device, non_blocking=True)

        llm_batch = build_llm_batch_from_prompt_answer(
            prompt_texts=prompt_texts,
            answers=answers,
            tokenizer=llm_tokenizer,
            max_length=llm_max_length,
        )
        if llm_batch is None:
            skipped += 1
            continue

        labels = llm_batch["labels"].to(device, non_blocking=True)
        if labels.ne(-100).sum() == 0:
            skipped += 1
            continue

        llm_input_ids      = llm_batch["input_ids"].to(device, non_blocking=True)
        llm_attention_mask = llm_batch["attention_mask"].to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            enabled=torch.cuda.is_available(),
        ):
            outputs = model(
                images=images,
                input_ids=clip_input_ids,
                attention_mask=clip_attention_mask,
                llm_input_ids=llm_input_ids,
                llm_attention_mask=llm_attention_mask,
                labels=labels,
                use_hard_pruning=False,
                train_pruning_mode="soft",  # keep_ratio=1.0 → gates≈1 → all tokens pass
            )
            lm_loss = outputs.get("loss")
            if lm_loss is None or not torch.isfinite(lm_loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            loss = lm_loss / grad_accum_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(dataloader))
        if should_step:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.projector.parameters(), max_grad_norm, error_if_nonfinite=False
            )
            if torch.isfinite(grad_norm):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
            else:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)

        total_loss += lm_loss.detach().item()
        num_effective_steps += 1

        if step % 50 == 0:
            print(
                f"[Stage1] Step {step}/{len(dataloader)} | "
                f"LM {lm_loss.item():.4f} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e} | "
                f"Skipped {skipped}"
            )

    avg_loss = total_loss / max(1, num_effective_steps)
    print(f"[Stage1] Done. avg_loss={avg_loss:.4f}   skipped={skipped}")

    # ── Validation ───────────────────────────────────────────────────────
    val_metrics = validate_vqa_generation(
        model=model,
        clip_tokenizer=clip_tokenizer,
        llm_tokenizer=llm_tokenizer,
        device=device,
        keep_ratio=1.0,
        batch_size=val_batch_size,
        num_workers=val_num_workers,
        max_samples=val_max_samples,
        image_size=336,
        llm_max_length=llm_max_length,
        clip_max_length=clip_max_length,
    )
    print(
        f"[Stage1 Val] vqa_soft_accuracy={val_metrics['vqa_soft_accuracy']:.4f} | "
        f"empty_pred_rate={val_metrics['empty_prediction_rate']:.4f}"
    )

    # ── Save checkpoint ───────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    trainable_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(
        {"trainable_non_llm_state_dict": trainable_state},
        os.path.join(save_dir, "non_llm_trainables.pt"),
    )
    torch.save(
        {
            "stage": "alignment_complete",
            "avg_loss": avg_loss,
            "val_vqa_soft_accuracy": val_metrics["vqa_soft_accuracy"],
            "projector_hidden_dim": projector_hidden_dim,
            "ia_hidden_dim": ia_hidden_dim,
        },
        os.path.join(save_dir, "training_state.pt"),
    )
    print(f"[Stage1] Checkpoint saved → {save_dir}")


# ============================================================
# Stage 2 — Visual Instruction Tuning + Token Pruning
# ============================================================
def train_stage2(
    num_epochs: int = 1,
    batch_size: int = 2,
    num_workers: int = 4,
    grad_accum_steps: int = 4,
    max_grad_norm: float = 1.0,
    adapter_lr: float = 2e-5,
    lora_lr: float = 2e-4,
    weight_decay: float = 0.01,
    keep_ratio: float = 0.7,
    keep_ratio_start: float = 0.95,   # step curriculum start (step 0)
    soft_stage_ratio: float = 0.33,
    ste_stage_ratio: float = 0.33,
    lambda_budget: float = 0.05,
    budget_warmup_ratio: float = 0.0,  # 0 = no warmup; budget active from step 1
    lambda_distill: float = 0.5,
    start_temp: float = 1.5,
    end_temp: float = 0.5,
    use_ste: bool = False,
    max_samples: Optional[int] = None,
    llm_max_length: int = 256,
    clip_max_length: int = 64,
    val_batch_size: int = 4,
    val_num_workers: int = 2,
    val_max_samples: int = 1000,
    val_every_n_epochs: int = 1,
    dynamic_budget_enabled: bool = False,
    dynamic_budget_min_keep_ratio: float = 0.45,
    dynamic_budget_max_keep_ratio: float = 0.65,
    dynamic_budget_supervision_mode: str = "heuristic",  # "heuristic" | "self"
    multi_ratio_enabled: bool = False,
    multi_ratio_values: Optional[List[float]] = None,
    multi_ratio_probs: Optional[List[float]] = None,
    multi_ratio_low_focus_start_ratio: float = 0.6,
    multi_ratio_low_focus_power: float = 1.0,
    use_ratio_adaptive_loss: bool = True,
    dynamic_budget_start_ratio: float = 0.5,
    enable_tf32: bool = True,
    use_autocast_fp16: bool = False,
    resume_from: Optional[str] = None,
    stage1_checkpoint: Optional[str] = None,  # Stage 1 projector weights to load
    llava15_init: bool = True,
    llava15_model_name: str = "llava-hf/llava-1.5-7b-hf",
    projector_hidden_dim: int = 2048,
    ia_hidden_dim: int = 512,
    freeze_llm: bool = True,
    projector_unfreeze_threshold: float = 0.25,
    projector_finetune_lr: float = 5e-6,
    best_checkpoint_name: str = "best_model",
    latest_checkpoint_name: str = "latest",
):
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if torch.cuda.is_available() and enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # When llava15_init=True, load LLM directly from the LLaVA-1.5 checkpoint
    # (extracts only language_model, frees vision+projector) — no separate Vicuna download.
    _llm_source = llava15_model_name if llava15_init else "lmsys/vicuna-7b-v1.5"

    model = PruningVLM(
        clip_model_name="openai/clip-vit-large-patch14-336",
        llm_model_name=_llm_source,
        keep_ratio=keep_ratio,
        alpha=0.5,
        learnable_alpha=True,
        llm_torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        gate_temperature=start_temp,
        gate_score_scale=5.0,
        use_ste=use_ste,
        gate_threshold_mode="topk",
        llm_use_grad=False,
        llm_scale_visual_prefix=False,
        llm_visual_prefix_scale=1.0,
        projector_hidden_dim=projector_hidden_dim,
        ia_hidden_dim=ia_hidden_dim,
        dynamic_budget_enabled=dynamic_budget_enabled,
        dynamic_budget_min_keep_ratio=dynamic_budget_min_keep_ratio,
        dynamic_budget_max_keep_ratio=dynamic_budget_max_keep_ratio,
        use_merging=True,
    ).to(device)

    # ── Stage 2: CLIP encoders always frozen ────────────────────────────
    freeze_module(model.vision_encoder)
    freeze_module(model.text_encoder)

    if freeze_llm:
        # freeze_llm=True: LLM + projector both frozen.
        # Use original LLaVA-1.5 projector (already loaded by PruningVLM.__init__
        # via _llava_projector). Do NOT overwrite with Stage 1 projector — Stage 1
        # fine-tunes projector on VQA-v2 which biases features and drops Cognition.
        # Only load scorer weights from Stage 1 (if any were saved there).
        if stage1_checkpoint is not None:
            _s1_path = os.path.join(stage1_checkpoint, "non_llm_trainables.pt")
            if os.path.exists(_s1_path):
                _s1 = torch.load(_s1_path, map_location=device)
                _s1_state = _s1["trainable_non_llm_state_dict"]
                _model_state = model.state_dict()
                # Load ONLY scorer weights — skip projector keys
                _scorer_prefixes = ("cls_scorer.", "instruction_aware.",
                                    "text_importance.", "score_fusion.")
                _compatible = {
                    k: v for k, v in _s1_state.items()
                    if any(k.startswith(pfx) for pfx in _scorer_prefixes)
                    and k in _model_state and _model_state[k].shape == v.shape
                }
                model.load_state_dict(_compatible, strict=False)
                print(f"[Stage2] freeze_llm=True: loaded {len(_compatible)} scorer "
                      f"weights from Stage 1 (projector kept as LLaVA-1.5 pretrained).")
        # Freeze LLM always
        for p in model.llm.model.parameters():
            p.requires_grad = False
        # Conditionally unfreeze projector for aggressive pruning regimes
        _effective_min_ratio = min(float(r) for r in multi_ratio_values if float(r) > 0) if multi_ratio_values else float(keep_ratio)
        _should_unfreeze_projector = (
            projector_unfreeze_threshold > 0.0
            and _effective_min_ratio < projector_unfreeze_threshold
        )
        if _should_unfreeze_projector:
            unfreeze_module(model.projector)
            print(
                f"[Stage2] Projector UNFROZEN (min ratio {_effective_min_ratio:.2f} "
                f"< threshold {projector_unfreeze_threshold:.2f}) — "
                f"will train with lr={projector_finetune_lr:.1e}."
            )
        else:
            freeze_module(model.projector)
        # Always train scorer
        unfreeze_module(model.instruction_aware)
        unfreeze_module(model.cls_scorer)
        unfreeze_module(model.text_importance)
        unfreeze_module(model.score_fusion)
        if _should_unfreeze_projector:
            print("[Stage2] Training scorer + projector (LLM frozen).")
        else:
            print("[Stage2] LLM + projector frozen — training scorer only (~7M params).")
    else:
        # LoRA path: load Stage 1 checkpoint (projector + scorer) as warm start
        if stage1_checkpoint is not None:
            _s1_path = os.path.join(stage1_checkpoint, "non_llm_trainables.pt")
            if os.path.exists(_s1_path):
                _s1 = torch.load(_s1_path, map_location=device)
                _s1_state = _s1["trainable_non_llm_state_dict"]
                _model_state = model.state_dict()
                _compatible = {
                    k: v for k, v in _s1_state.items()
                    if k in _model_state and _model_state[k].shape == v.shape
                }
                model.load_state_dict(_compatible, strict=False)
                print(f"[Stage2] LoRA path: loaded {len(_compatible)} weights from Stage 1.")
        elif llava15_init:
            load_llava15_pretrained_weights(
                model, llava_model_name=llava15_model_name,
                load_llm=False, load_projector=True,
            )
        unfreeze_module(model.projector)
        unfreeze_module(model.instruction_aware)
        unfreeze_module(model.cls_scorer)
        unfreeze_module(model.text_importance)
        unfreeze_module(model.score_fusion)
        # LoRA path: LLM adapts via low-rank updates alongside projector + scorer.
        # WARNING: may cause catastrophic forgetting of reasoning/cognition tasks.
        model.llm = attach_lora_to_llm(
            model.llm, r=16, lora_alpha=32, lora_dropout=0.05,
        )
        print("[Stage2] LoRA attached — training projector + scorer + LLM (LoRA).")

    model = model.to(device)
    
    start_epoch = 0
    _pending_opt_sched_state: Optional[dict] = None

    if resume_from is not None:
        print(f"Resuming from: {resume_from}")
        non_llm_path = os.path.join(resume_from, "non_llm_trainables.pt")
        if os.path.exists(non_llm_path):
            non_llm_ckpt = torch.load(non_llm_path, map_location=device)
            ckpt_state = non_llm_ckpt["trainable_non_llm_state_dict"]
            model_state = model.state_dict()

            # Filter out any key whose tensor shape doesn't match the current model
            # (happens when architecture changed between checkpoints, e.g. projector)
            compatible = {}
            skipped = []
            for k, v in ckpt_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    compatible[k] = v
                else:
                    skipped.append(
                        f"{k}: ckpt {tuple(v.shape) if hasattr(v,'shape') else '?'}"
                        f" vs model {tuple(model_state[k].shape) if k in model_state else 'missing'}"
                    )
            if skipped:
                print(f"[Resume] Skipped {len(skipped)} mismatched/missing keys "
                      f"(will be trained from scratch):")
                for s in skipped:
                    print(f"  • {s}")
            model.load_state_dict(compatible, strict=False)
            print(f"[Resume] Loaded {len(compatible)} compatible keys.")
        
        lora_dir = os.path.join(resume_from, "llm_lora_adapter")
        if os.path.exists(lora_dir):
            from peft import PeftModel
            model.llm.model.load_adapter(lora_dir, adapter_name="default")

        # Load start_epoch and pending optimizer/scheduler state if available
        _ts_path = os.path.join(resume_from, "training_state.pt")
        if os.path.exists(_ts_path):
            _ts = torch.load(_ts_path, map_location="cpu")
            start_epoch = int(_ts.get("epoch", 0))
            if "optimizer_state_dict" in _ts:
                _pending_opt_sched_state = _ts
            print(f"[Resume] start_epoch={start_epoch}")

        print("Resumed successfully.")

    if hasattr(model.llm.model, "print_trainable_parameters"):
        model.llm.model.print_trainable_parameters()

    print_trainable_parameters(model)

    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")

    llm_tokenizer = model.llm.tokenizer if hasattr(model.llm, "tokenizer") else None
    if llm_tokenizer is None:
        raise ValueError("model.llm.tokenizer is required.")

    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    val_clip_tokenizer = clip_tokenizer
    val_llm_tokenizer = llm_tokenizer

    train_questions_path = os.path.join(DATA_ROOT, "v2_OpenEnded_mscoco_train2014_questions.json")
    train_annotations_path = os.path.join(DATA_ROOT, "v2_mscoco_train2014_annotations.json")
    train_image_root = os.path.join(DATA_ROOT, "train2014")
    ensure_paths_exist(
        [train_questions_path, train_annotations_path, train_image_root],
        "Training dataset paths are not ready.",
    )

    dataset = VQAv2Dataset(
        questions_path=train_questions_path,
        annotations_path=train_annotations_path,
        image_root=train_image_root,
        max_samples=max_samples,
        shuffle_before_select=True,
        seed=42,
        require_image=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=vqa_collate_fn,
        drop_last=False,
    )

    print(f"Loaded dataset with {len(dataset)} samples")

    lora_params = []
    no_decay_params = []   # scalars / biases — no weight decay
    non_llm_params = []
    projector_decay_params = []
    projector_nodecay_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(p)
        elif name.startswith("projector."):
            # Projector uses its own low LR when unfrozen for low-ratio finetune
            if p.ndim <= 1 or name.endswith(".bias"):
                projector_nodecay_params.append(p)
            else:
                projector_decay_params.append(p)
        elif p.ndim <= 1 or name.endswith(".bias") or "score_fusion.alpha" in name:
            # scalars (alpha), biases, and 1-d params should not be weight-decayed
            no_decay_params.append(p)
        else:
            non_llm_params.append(p)

    param_groups = [
        {"params": non_llm_params, "lr": adapter_lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": adapter_lr, "weight_decay": 0.0},
    ]
    if projector_decay_params:
        param_groups.append({"params": projector_decay_params, "lr": projector_finetune_lr, "weight_decay": weight_decay, "name": "projector_decay"})
    if projector_nodecay_params:
        param_groups.append({"params": projector_nodecay_params, "lr": projector_finetune_lr, "weight_decay": 0.0, "name": "projector_nodecay"})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
    )

    total_update_steps = max(1, (len(dataloader) * num_epochs + grad_accum_steps - 1) // grad_accum_steps)
    warmup_steps = max(10, int(0.03 * total_update_steps))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(torch.cuda.is_available() and not use_bf16)
    )

    ratio_pool = _normalize_ratio_values(multi_ratio_values)
    if multi_ratio_enabled and not ratio_pool:
        ratio_pool = _normalize_ratio_values([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, keep_ratio])
    if not ratio_pool:
        ratio_pool = [float(keep_ratio)]

    if multi_ratio_probs is not None and len(multi_ratio_probs) != len(ratio_pool):
        print(
            f"[Stage2] Ignoring multi_ratio_probs because len={len(multi_ratio_probs)} "
            f"!= len(ratio_pool)={len(ratio_pool)}"
        )
        multi_ratio_probs = None

    # Apply pending optimizer/scheduler state from checkpoint (if any)
    if _pending_opt_sched_state is not None:
        try:
            optimizer.load_state_dict(_pending_opt_sched_state["optimizer_state_dict"])
            print("[Resume] Loaded optimizer state.")
        except Exception as e:
            print(f"[Resume] Could not load optimizer state (will train with fresh optimizer): {e}")
        if "scheduler_state_dict" in _pending_opt_sched_state:
            try:
                scheduler.load_state_dict(_pending_opt_sched_state["scheduler_state_dict"])
                print("[Resume] Loaded scheduler state.")
            except Exception as e:
                print(f"[Resume] Could not load scheduler state: {e}")
        _pending_opt_sched_state = None

    model.train()
    model.vision_encoder.eval()
    model.text_encoder.eval()

    best_loss = float("inf")
    best_val_vqa_soft_acc = float("-inf")
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)

    skipped_empty_batch = 0
    skipped_invalid_loss = 0
    total_truncated_samples = 0
    total_empty_answer_after_trunc = 0

    total_steps = (num_epochs - start_epoch) * len(dataloader)

    if dynamic_budget_supervision_mode not in {"heuristic", "self"}:
        raise ValueError(
            "dynamic_budget_supervision_mode must be one of ['heuristic', 'self'], "
            f"got {dynamic_budget_supervision_mode}"
        )

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        num_effective_steps = 0

        current_temp = linear_temp_schedule(
            epoch=epoch,
            num_epochs=num_epochs,
            start_temp=start_temp,
            end_temp=end_temp,
        )
        model.set_gate_temperature(current_temp)

        # Budget warmup scale (epoch-level; returns 1.0 when num_epochs<=1)
        budget_warmup_scale = linear_weight_warmup(
            epoch=epoch,
            num_epochs=num_epochs,
            warmup_ratio=budget_warmup_ratio,
        )
        current_lambda_budget = lambda_budget * budget_warmup_scale

        # Dynamic budget: enable only after warm-up fraction of training
        current_epoch_ratio = (epoch - start_epoch) / max(1, num_epochs - start_epoch)
        should_use_dynamic_budget = (
            dynamic_budget_enabled and (current_epoch_ratio >= dynamic_budget_start_ratio)
        )
        model.set_dynamic_budget(should_use_dynamic_budget)

        print(
            f"Epoch {epoch + 1}: temp={current_temp:.3f}  "
            f"keep_ratio {keep_ratio_start:.3f}→{keep_ratio:.3f} (step-scheduled)  "
            f"lambda_budget_eff={current_lambda_budget:.4f}  "
            f"dynamic_budget={'ON' if should_use_dynamic_budget else 'OFF'}  "
            f"budget_sup={dynamic_budget_supervision_mode}  "
            f"multi_ratio={'ON' if multi_ratio_enabled else 'OFF'}"
        )

        for step, batch in enumerate(dataloader):
            # ── Per-step curriculum (advances across the full training run) ──
            _global_step = (epoch - start_epoch) * len(dataloader) + step
            _progress = _global_step / max(1, total_steps - 1)
            current_pruning_mode = step_pruning_mode(
                _global_step, total_steps,
                soft_stage_frac=soft_stage_ratio,
                ste_stage_frac=ste_stage_ratio,
            )
            scheduled_keep_ratio = step_keep_ratio(
                _global_step, total_steps,
                start_ratio=keep_ratio_start,
                end_ratio=keep_ratio,
            )
            if multi_ratio_enabled:
                current_keep_ratio = _sample_multi_ratio(
                    ratio_values=ratio_pool,
                    ratio_probs=multi_ratio_probs,
                    progress=_progress,
                    low_focus_start_ratio=multi_ratio_low_focus_start_ratio,
                    low_focus_power=multi_ratio_low_focus_power,
                )
            else:
                current_keep_ratio = scheduled_keep_ratio

            if use_ratio_adaptive_loss:
                loss_scales = ratio_adaptive_loss_scales(current_keep_ratio)
            else:
                loss_scales = {"budget": 1.0, "distill": 1.0}

            current_lambda_budget_step = current_lambda_budget * loss_scales["budget"]
            current_lambda_distill_step = lambda_distill * loss_scales["distill"]
            model.set_keep_ratio(current_keep_ratio)

            images = prepare_images_for_model(
                batch["image"],
                device=device,
                image_size=336,
            )
            questions = batch["question"]
            answers = batch["answer"]
            prompt_texts = batch["prompt_text"]

            # text query cho scorer branch
            clip_tokens = clip_tokenizer(
                questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=clip_max_length,
            )
            clip_input_ids = clip_tokens.input_ids.to(device, non_blocking=True)
            clip_attention_mask = clip_tokens.attention_mask.to(device, non_blocking=True)

            # llm input_ids / labels an toàn
            llm_batch = build_llm_batch_from_prompt_answer(
                prompt_texts=prompt_texts,
                answers=answers,
                tokenizer=llm_tokenizer,
                max_length=llm_max_length,
            )

            if llm_batch is None:
                skipped_empty_batch += 1
                if step % 20 == 0:
                    print(f"[Warning] Skip step {step}: entire batch has no valid answer tokens after truncation.")
                continue

            total_truncated_samples += llm_batch["num_truncated"]
            total_empty_answer_after_trunc += llm_batch["num_empty_answer_after_trunc"]

            llm_input_ids = llm_batch["input_ids"].to(device, non_blocking=True)
            llm_attention_mask = llm_batch["attention_mask"].to(device, non_blocking=True)
            labels = llm_batch["labels"].to(device, non_blocking=True)

            # Nếu có sample bị drop trong llm_batch do truncation hết answer,
            # thì batch image/clip vẫn còn size cũ -> cần đồng bộ lại.
            # Cách an toàn: nếu số sample valid < batch ban đầu thì rebuild image và clip theo sample valid.
            if llm_batch["num_valid_samples"] != len(prompt_texts):
                valid_indices = []
                for i, (prompt_text, answer) in enumerate(zip(prompt_texts, answers)):
                    prompt_ids = llm_tokenizer(
                        prompt_text,
                        add_special_tokens=True,   # BOS <s> required by Vicuna
                        truncation=False,
                    )["input_ids"]

                    answer_text = " " + str(answer).strip().lower()
                    if llm_tokenizer.eos_token is not None:
                        answer_text += llm_tokenizer.eos_token

                    answer_ids = llm_tokenizer(
                        answer_text,
                        add_special_tokens=False,
                        truncation=False,
                    )["input_ids"]

                    if len(answer_ids) == 0 and llm_tokenizer.eos_token_id is not None:
                        answer_ids = [llm_tokenizer.eos_token_id]

                    max_prompt_len = llm_max_length - 1
                    if len(prompt_ids) > max_prompt_len:
                        prompt_ids = prompt_ids[:max_prompt_len]

                    remaining = llm_max_length - len(prompt_ids)
                    truncated_answer_ids = answer_ids[:remaining]
                    if len(truncated_answer_ids) > 0:
                        valid_indices.append(i)

                images = images[valid_indices]
                clip_input_ids = clip_input_ids[valid_indices]
                clip_attention_mask = clip_attention_mask[valid_indices]

            valid_target_count = (labels != -100).sum().item()
            if valid_target_count == 0:
                skipped_empty_batch += 1
                if step % 20 == 0:
                    print(f"[Warning] Skip step {step}: labels are all -100.")
                continue

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=(torch.cuda.is_available() and (use_bf16 or use_autocast_fp16)),
            ):
                outputs = model(
                    images=images,
                    input_ids=clip_input_ids,
                    attention_mask=clip_attention_mask,
                    llm_input_ids=llm_input_ids,
                    llm_attention_mask=llm_attention_mask,
                    labels=labels,
                    return_intermediates=True,
                    use_hard_pruning=False,
                    train_pruning_mode=current_pruning_mode,
                    compute_distill_loss=(current_lambda_distill_step > 0),
                )

                lm_loss = outputs.get("loss", None)
                if lm_loss is None:
                    raise ValueError(
                        "Model did not return loss. "
                        "Expected LLaVALM to compute aligned loss internally."
                    )

                if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                    skipped_invalid_loss += 1
                    print(f"[Warning] Skip step {step}: lm_loss is invalid ({lm_loss.item()}).")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                soft_gates = outputs["soft_gates"]
                fused_scores = outputs["fused_scores"]

                if soft_gates is None:
                    raise ValueError("Model must return `soft_gates` in return_intermediates=True mode.")

                budget_loss = compute_budget_loss(
                    soft_gates=soft_gates,
                    target_keep_ratio=current_keep_ratio,
                )

                dynamic_keep_ratio = outputs.get("dynamic_keep_ratio", None)

                if should_use_dynamic_budget and dynamic_keep_ratio is not None:
                    if dynamic_budget_supervision_mode == "heuristic":
                        budget_target = build_question_adaptive_keep_ratio_targets(
                            questions=questions,
                            answers=answers,
                            min_keep_ratio=dynamic_budget_min_keep_ratio,
                            max_keep_ratio=dynamic_budget_max_keep_ratio,
                            device=soft_gates.device,
                            dtype=soft_gates.dtype,
                        )
                    else:
                        budget_target = dynamic_keep_ratio.detach()

                    budget_loss = compute_budget_loss(
                        soft_gates=soft_gates,
                        target_keep_ratio=budget_target,
                    )

                distill_loss = outputs.get("distill_loss")
                if distill_loss is None:
                    distill_loss = fused_scores.new_tensor(0.0)

                full_loss = (
                    lm_loss
                    + current_lambda_budget_step * budget_loss
                    + current_lambda_distill_step * distill_loss
                )

                if torch.isnan(full_loss) or torch.isinf(full_loss):
                    skipped_invalid_loss += 1
                    print(f"[Warning] Skip step {step}: full_loss is invalid ({full_loss.item()}).")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss = full_loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(dataloader))

            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                    error_if_nonfinite=False,
                )
                bad_grad = (not torch.isfinite(grad_norm)) or has_non_finite_gradients(model)
                if bad_grad:
                    skipped_invalid_loss += 1
                    optimizer.zero_grad(set_to_none=True)
                    if step % 20 == 0:
                        print(f"[Warning] Skip step {step}: non-finite gradients (grad_norm={grad_norm}).")
                    continue

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            total_loss += full_loss.detach().item()
            num_effective_steps += 1

            if step % 20 == 0:
                num_visual_tokens = (
                    outputs["visual_for_llm"].size(1)
                    if "visual_for_llm" in outputs and outputs["visual_for_llm"] is not None
                    else -1
                )

                soft_keep_ratio = (
                    outputs["soft_gates"].mean().item()
                    if "soft_gates" in outputs and outputs["soft_gates"] is not None
                    else -1.0
                )

                hard_keep_ratio = (
                    outputs["hard_mask"].mean().item()
                    if "hard_mask" in outputs and outputs["hard_mask"] is not None
                    else -1.0
                )

                alpha_value = (
                    torch.clamp(model.score_fusion.alpha.detach(), 0.0, 1.0).item()
                    if hasattr(model.score_fusion, "alpha")
                    else -1.0
                )

                score_mean = outputs["fused_scores"].mean().item()
                score_std = outputs["fused_scores"].std().item()
                score_min = outputs["fused_scores"].min().item()
                score_max = outputs["fused_scores"].max().item()
                gate_std = outputs["soft_gates"].std().item()

                lora_lr_log = (
                    f"LR(lora) {optimizer.param_groups[1]['lr']:.2e} | "
                    if len(optimizer.param_groups) > 1
                    else ""
                )
                dyn_keep_log = (
                    f"DynKeepMean {dynamic_keep_ratio.mean().item():.4f} | "
                    if dynamic_keep_ratio is not None
                    else ""
                )
                print(
                    f"Epoch {epoch + 1} | "
                    f"Step {step}/{len(dataloader)} | "
                    f"Loss {full_loss.item():.4f} | "
                    f"LM {lm_loss.item():.4f} | "
                    f"Budget {budget_loss.item():.4f} | "
                    f"Distill {distill_loss.item():.4f} | "
                    f"Keep(schedule/sample) {scheduled_keep_ratio:.3f}/{current_keep_ratio:.3f} | "
                    f"Lambda(budget/distill) {current_lambda_budget_step:.4f}/{current_lambda_distill_step:.4f} | "
                    f"SoftKeep {soft_keep_ratio:.4f} | "
                    f"HardKeep {hard_keep_ratio:.4f} | "
                    f"Temp {current_temp:.3f} | "
                    f"PruningMode {current_pruning_mode} | "
                    f"Alpha {alpha_value:.4f} | "
                    f"Visual tokens {num_visual_tokens} | "
                    f"LR(adapter) {optimizer.param_groups[0]['lr']:.2e} | "
                    f"{lora_lr_log}"
                    f"ScoreMean {score_mean:.4f} | "
                    f"ScoreStd {score_std:.4f} | "
                    f"ScoreMin {score_min:.4f} | "
                    f"ScoreMax {score_max:.4f} | "
                    f"GateStd {gate_std:.4f} | "
                    f"{dyn_keep_log}"
                    f"ValidTarget {valid_target_count} | "
                    f"TruncSoFar {total_truncated_samples} | "
                    f"EmptyAnsAfterTruncSoFar {total_empty_answer_after_trunc}"
                )

        avg_loss = total_loss / max(1, num_effective_steps)
        print(f"Epoch {epoch + 1} finished | avg_loss = {avg_loss:.4f}")
        print(
            f"Skipped empty batches: {skipped_empty_batch} | "
            f"Skipped invalid loss: {skipped_invalid_loss} | "
            f"Total truncated samples: {total_truncated_samples} | "
            f"Total empty-answer-after-trunc samples: {total_empty_answer_after_trunc}"
        )
        print(f"✅ Finished Epoch")

        should_run_val = ((epoch + 1) % max(1, val_every_n_epochs) == 0) or ((epoch + 1) == num_epochs)
        val_metrics = None
        if should_run_val:
            val_metrics = validate_vqa_generation(
                model=model,
                clip_tokenizer=val_clip_tokenizer,
                llm_tokenizer=val_llm_tokenizer,
                device=device,
                keep_ratio=keep_ratio,
                batch_size=val_batch_size,
                num_workers=val_num_workers,
                max_samples=val_max_samples,
                image_size=336,
                llm_max_length=llm_max_length,
                clip_max_length=clip_max_length,
                seed=42,
            )
            print(
                f"[Val] epoch {epoch + 1} | "
                f"vqa_soft_accuracy={val_metrics['vqa_soft_accuracy']:.4f} | "
                f"empty_prediction_rate={val_metrics['empty_prediction_rate']:.4f} | "
                f"samples={val_metrics['num_samples']}"
            )
        else:
            print(f"[Val] epoch {epoch + 1} skipped (val_every_n_epochs={val_every_n_epochs}).")

        should_save_best = False
        if val_metrics is not None and val_metrics["vqa_soft_accuracy"] > best_val_vqa_soft_acc:
            best_val_vqa_soft_acc = val_metrics["vqa_soft_accuracy"]
            best_loss = avg_loss
            should_save_best = True
        elif (
            val_metrics is not None
            and
            val_metrics["vqa_soft_accuracy"] == best_val_vqa_soft_acc
            and avg_loss < best_loss
        ):
            best_loss = avg_loss
            should_save_best = True
        elif val_metrics is None and avg_loss < best_loss:
            best_loss = avg_loss
            should_save_best = True

        if should_save_best:

            best_dir = os.path.join(CHECKPOINT_ROOT, best_checkpoint_name)
            os.makedirs(best_dir, exist_ok=True)

            # chỉ lưu metadata + training states cần thiết
            torch.save(
                {
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                    "best_loss": best_loss,
                    "best_val_vqa_soft_accuracy": best_val_vqa_soft_acc,
                    "val_vqa_soft_accuracy": (
                        val_metrics["vqa_soft_accuracy"] if val_metrics is not None else None
                    ),
                    "val_empty_prediction_rate": (
                        val_metrics["empty_prediction_rate"] if val_metrics is not None else None
                    ),
                    "val_num_samples": (
                        val_metrics["num_samples"] if val_metrics is not None else 0
                    ),
                    "skipped_empty_batch": skipped_empty_batch,
                    "skipped_invalid_loss": skipped_invalid_loss,
                    "total_truncated_samples": total_truncated_samples,
                    "total_empty_answer_after_trunc": total_empty_answer_after_trunc,
                    "keep_ratio": keep_ratio,
                    "soft_stage_ratio": soft_stage_ratio,
                    "ste_stage_ratio": ste_stage_ratio,
                    "budget_warmup_ratio": budget_warmup_ratio,
                    "dynamic_budget_enabled": dynamic_budget_enabled,
                    "dynamic_budget_min_keep_ratio": dynamic_budget_min_keep_ratio,
                    "dynamic_budget_max_keep_ratio": dynamic_budget_max_keep_ratio,
                    "dynamic_budget_supervision_mode": dynamic_budget_supervision_mode,
                    "train_pruning_mode": current_pruning_mode,
                    "llm_max_length": llm_max_length,
                    "clip_max_length": clip_max_length,
                    # save optimizer/scheduler for crash recovery
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                os.path.join(best_dir, "training_state.pt"),
            )

            # lưu đúng phần trainable thôi
            save_lora_and_non_llm_trainables(model, best_dir)

            print(f"🔥 Saved best checkpoint by val VQA soft accuracy to: {best_dir}")

        # Always save a latest checkpoint after each epoch for crash recovery
        latest_dir = os.path.join(CHECKPOINT_ROOT, latest_checkpoint_name)
        os.makedirs(latest_dir, exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "best_val_vqa_soft_accuracy": best_val_vqa_soft_acc,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(latest_dir, "training_state.pt"),
        )
        save_lora_and_non_llm_trainables(model, latest_dir)
        print(f"[Checkpoint] Saved latest checkpoint (epoch {epoch + 1}) → {latest_dir}")


if __name__ == "__main__":
    import os as _os
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    STAGE1_DIR = os.path.join(CHECKPOINT_ROOT, "state_1_fastbest")
    BEST_STAGE2_DIR_NAME = "best_model_fastbest"
    LATEST_STAGE2_DIR_NAME = "latest_fastbest"
    STAGE2_LATEST_DIR = os.path.join(CHECKPOINT_ROOT, LATEST_STAGE2_DIR_NAME)
    _stage1_done = os.path.exists(os.path.join(STAGE1_DIR, "training_state.pt"))
    _stage2_resume_from = (
        STAGE2_LATEST_DIR
        if os.path.exists(os.path.join(STAGE2_LATEST_DIR, "training_state.pt"))
        else None
    )

    stage1_max_samples = 100_000
    stage2_max_samples = 200_000
    stage2_num_epochs = 2
    val_max_samples = 500

    print("=" * 60)
    print("Training target: SOTA-oriented accuracy under ~1 day budget")
    print(f"Stage1 checkpoint: {STAGE1_DIR}")
    print(f"Stage2 best checkpoint dir: {os.path.join(CHECKPOINT_ROOT, BEST_STAGE2_DIR_NAME)}")
    print(f"Stage2 latest checkpoint dir: {os.path.join(CHECKPOINT_ROOT, LATEST_STAGE2_DIR_NAME)}")
    print(f"Stage2 resume from: {_stage2_resume_from}")
    print(f"Stage1 max_samples: {stage1_max_samples}")
    print(f"Stage2 max_samples: {stage2_max_samples}")
    print(f"Stage2 num_epochs: {stage2_num_epochs}")
    print("=" * 60)

    # ── Stage 1: Projector Alignment ───────────────────────────────────────────
    # Static-first setup: keep Stage 1 focused on projector alignment rather than
    # dynamic-budget supervision. This gives Stage 2 a stronger visual prefix before
    # multi-ratio pruning starts.
    # Skipped automatically if checkpoint already exists.
    if _stage1_done:
        print(f"[Stage1] Checkpoint found at {STAGE1_DIR} — skipping Stage 1.")
    else:
        train_stage1(
            save_dir=STAGE1_DIR,
            batch_size=3,
            num_workers=4,
            grad_accum_steps=8,
            max_grad_norm=1.0,
            projector_lr=5e-5,
            weight_decay=0.01,
            max_samples=stage1_max_samples,
            llm_max_length=256,
            clip_max_length=64,
            val_batch_size=4,
            val_num_workers=2,
            val_max_samples=val_max_samples,
            projector_hidden_dim=2048,
            ia_hidden_dim=512,
            enable_tf32=True,
            resume_from=None,
            llava15_init=True,
            llava15_model_name="llava-hf/llava-1.5-7b-hf",
            train_budget_head_only=False,
        )

    # ── Stage 2: Pruning-Aware Accuracy Tuning ───────────────────────────────
    # Target: keep token pruning active and competitive at low budgets.
    # Strategy: train scorer + projector + LoRA-LLM jointly on full data with a
    # multi-ratio curriculum, while keeping budget/distillation losses enabled.
    train_stage2(
        stage1_checkpoint=STAGE1_DIR,
        llava15_init=True,
        llava15_model_name="llava-hf/llava-1.5-7b-hf",
        freeze_llm=False,
        num_epochs=stage2_num_epochs,
        batch_size=2,
        num_workers=4,
        grad_accum_steps=8,
        max_grad_norm=0.5,
        adapter_lr=1e-5,
        lora_lr=1e-4,
        weight_decay=0.01,
        keep_ratio=0.333,
        keep_ratio_start=1.0,
        soft_stage_ratio=0.40,
        ste_stage_ratio=0.30,
        lambda_budget=0.03,
        budget_warmup_ratio=0.0,
        lambda_distill=0.8,
        start_temp=1.5,
        end_temp=0.5,
        max_samples=stage2_max_samples,
        llm_max_length=256,
        clip_max_length=64,
        val_batch_size=4,
        val_num_workers=2,
        val_max_samples=val_max_samples,
        val_every_n_epochs=1,
        dynamic_budget_enabled=False,
        dynamic_budget_min_keep_ratio=0.20,
        dynamic_budget_max_keep_ratio=0.70,
        dynamic_budget_supervision_mode="heuristic",
        multi_ratio_enabled=True,
        multi_ratio_values=[0.056, 0.111, 0.223, 0.4, 0.5, 1.0],
        multi_ratio_probs=[1.5, 1.5, 1.4, 1.2, 1.0, 1.0],
        multi_ratio_low_focus_start_ratio=0.70,
        multi_ratio_low_focus_power=0.5,
        use_ratio_adaptive_loss=True,
        dynamic_budget_start_ratio=1.0,
        projector_unfreeze_threshold=0.40,
        projector_finetune_lr=5e-6,
        enable_tf32=True,
        resume_from=_stage2_resume_from,
        projector_hidden_dim=2048,
        ia_hidden_dim=512,
        best_checkpoint_name=BEST_STAGE2_DIR_NAME,
        latest_checkpoint_name=LATEST_STAGE2_DIR_NAME,
    )